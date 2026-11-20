"""Task 2 - recurrent neural networks (course Ch4) with PyTorch.

Sentiment analysis is the "many-to-one" case of the course: a sequence of
words goes in, one sentiment comes out. Three architectures are compared:

* ``rnn``      - Embedding -> simple RNN (tanh) -> Dense (softmax);
* ``lstm``     - Embedding -> LSTM -> Dense. The LSTM is the course's answer
                 to vanishing gradients on long sequences;
* ``cnn_lstm`` - Embedding -> Conv1D + max-pooling -> LSTM -> Dense, the
                 hybrid model of Othman et al. (2021) shown in Ch3: the
                 convolution extracts local features (n-grams), the LSTM the
                 long-distance dependencies.

The embedding layer is initialised with a word2vec model trained on the
training tweets (Ch3 "plongement sémantique") and fine-tuned. Exploding
gradients are limited by gradient clipping, overfitting by dropout and early
stopping on the validation macro-F1. Class imbalance is handled with a
class-weighted cross-entropy.

PyTorch is used rather than Keras because TensorFlow has no stable release
for Python 3.14 (the project's virtual environment); the layers are the
same as in the course's Keras examples (Embedding, SimpleRNN/LSTM, Dense).
"""
import copy
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .config import LABELS, RANDOM_STATE
from .features import train_word2vec
from .preprocessing import TextNormalizer

PAD, UNK = 0, 1


class Vocabulary:
    """Tokenizer + word index ("Tokenisation et séquencement" in Ch4)."""

    def __init__(self, tokenizer=None, min_count=2, max_len=50):
        self.tokenizer = tokenizer or TextNormalizer()
        self.min_count = min_count
        self.max_len = max_len

    def fit(self, texts):
        counts = Counter(w for t in texts for w in self.tokenizer(t))
        words = [w for w, c in counts.most_common() if c >= self.min_count]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def encode(self, texts):
        """Index sequences, truncated / right-padded to ``max_len``."""
        out = np.full((len(texts), self.max_len), PAD, dtype=np.int64)
        lengths = np.ones(len(texts), dtype=np.int64)
        for r, t in enumerate(texts):
            ids = [self.stoi.get(w, UNK) for w in self.tokenizer(t)][: self.max_len]
            if ids:
                out[r, : len(ids)] = ids
                lengths[r] = len(ids)
        return out, lengths


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, arch="lstm", emb_dim=100, hidden=128, n_classes=3,
                 dropout=0.4, embeddings=None, bidirectional=False, conv_filters=128,
                 kernel_size=3):
        super().__init__()
        self.arch = arch
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embeddings))
        self.dropout = nn.Dropout(dropout)
        rnn_in = emb_dim
        if arch == "cnn_lstm":
            self.conv = nn.Conv1d(emb_dim, conv_filters, kernel_size, padding=kernel_size // 2)
            self.pool = nn.MaxPool1d(2)
            rnn_in = conv_filters
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "cnn_lstm": nn.LSTM}[arch]
        kwargs = {"nonlinearity": "tanh"} if arch == "rnn" else {}
        self.rnn = rnn_cls(rnn_in, hidden, batch_first=True, bidirectional=bidirectional, **kwargs)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x, lengths):
        e = self.dropout(self.embedding(x))                       # (B, T, E)
        if self.arch == "cnn_lstm":
            c = torch.relu(self.conv(e.transpose(1, 2)))            # (B, F, T)
            e = self.pool(c).transpose(1, 2)                         # (B, T/2, F)
            lengths = torch.clamp(lengths // 2, min=1)
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[0] if isinstance(h, tuple) else h                      # LSTM returns (h, c)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.rnn.bidirectional else h[-1]
        return self.out(self.dropout(h))                             # many-to-one


class RNNSentimentClassifier:
    """scikit-learn style wrapper: ``fit(texts, labels)`` / ``predict(texts)``."""

    def __init__(self, arch="lstm", emb_dim=100, hidden=128, max_len=50, epochs=10,
                 batch_size=128, lr=2e-3, dropout=0.4, clip=1.0, patience=2,
                 bidirectional=False, pretrained=True, tokenizer=None,
                 random_state=RANDOM_STATE, verbose=True):
        self.arch, self.emb_dim, self.hidden, self.max_len = arch, emb_dim, hidden, max_len
        self.epochs, self.batch_size, self.lr, self.dropout = epochs, batch_size, lr, dropout
        self.clip, self.patience, self.bidirectional = clip, patience, bidirectional
        self.pretrained, self.tokenizer = pretrained, tokenizer
        self.random_state, self.verbose = random_state, verbose

    # labels -1/0/1 <-> class indices 0/1/2
    @staticmethod
    def _to_idx(y):
        return np.searchsorted(LABELS, np.asarray(y))

    def _embedding_matrix(self, texts):
        docs = [self.vocab_.tokenizer(t) for t in texts]
        w2v = train_word2vec(docs, vector_size=self.emb_dim, seed=self.random_state)
        rng = np.random.default_rng(self.random_state)
        mat = rng.normal(0, 0.1, (len(self.vocab_.itos), self.emb_dim)).astype(np.float32)
        mat[PAD] = 0
        hits = 0
        for i, w in enumerate(self.vocab_.itos):
            if w in w2v.wv.key_to_index:
                mat[i] = w2v.wv[w]
                hits += 1
        self.embedding_coverage_ = hits / len(self.vocab_.itos)
        return mat

    def _batches(self, X, L, y=None, shuffle=False):
        idx = np.arange(len(X))
        if shuffle:
            self.rng_.shuffle(idx)
        for s in range(0, len(idx), self.batch_size):
            b = idx[s:s + self.batch_size]
            xb, lb = torch.as_tensor(X[b]), torch.as_tensor(L[b])
            yield (xb, lb) if y is None else (xb, lb, torch.as_tensor(y[b]))

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        torch.manual_seed(self.random_state)
        self.rng_ = np.random.default_rng(self.random_state)
        texts = list(texts)
        self.vocab_ = Vocabulary(self.tokenizer, max_len=self.max_len).fit(texts)
        X, L = self.vocab_.encode(texts)
        y = self._to_idx(labels)
        emb = self._embedding_matrix(texts) if self.pretrained else None
        self.model_ = SentimentNet(len(self.vocab_.itos), self.arch, self.emb_dim, self.hidden,
                                   dropout=self.dropout, embeddings=emb,
                                   bidirectional=self.bidirectional)
        counts = np.bincount(y, minlength=3)
        weights = torch.as_tensor(len(y) / (3 * counts), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        has_val = val_texts is not None
        if has_val:
            Xv, Lv = self.vocab_.encode(list(val_texts))
            yv = np.asarray(val_labels)
        best, best_state, wait, self.history_ = -1, None, 0, []
        for epoch in range(1, self.epochs + 1):
            t = time.time()
            self.model_.train()
            total = 0.0
            for xb, lb, yb in self._batches(X, L, y, shuffle=True):
                opt.zero_grad()
                loss = loss_fn(self.model_(xb, lb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.clip)
                opt.step()
                total += loss.item() * len(yb)
            record = {"epoch": epoch, "train_loss": total / len(y), "seconds": time.time() - t}
            if has_val:
                record["val_macro_f1"] = f1_score(yv, self._predict_encoded(Xv, Lv), average="macro")
            self.history_.append(record)
            if self.verbose:
                print(f"    [{self.arch}] epoch {epoch}: " +
                      ", ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"))
            if has_val:
                if record["val_macro_f1"] > best:
                    best, best_state, wait = record["val_macro_f1"], copy.deepcopy(
                        self.model_.state_dict()), 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _predict_encoded(self, X, L):
        self.model_.eval()
        preds = [self.model_(xb, lb).argmax(1).numpy() for xb, lb in self._batches(X, L)]
        return np.asarray(LABELS)[np.concatenate(preds)]

    def predict(self, texts):
        X, L = self.vocab_.encode(list(texts))
        return self._predict_encoded(X, L)

    def save(self, path):
        torch.save({"params": {k: v for k, v in self.__dict__.items()
                               if not k.endswith("_") and k != "tokenizer"},
                    "itos": self.vocab_.itos, "state": self.model_.state_dict()}, path)

    @classmethod
    def load(cls, path, tokenizer=None):
        """Rebuild a classifier saved with :meth:`save` (no retraining)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        clf = cls(tokenizer=tokenizer, **ckpt["params"])
        clf.vocab_ = Vocabulary(tokenizer, max_len=clf.max_len)
        clf.vocab_.itos = ckpt["itos"]
        clf.vocab_.stoi = {w: i for i, w in enumerate(ckpt["itos"])}
        clf.model_ = SentimentNet(len(ckpt["itos"]), clf.arch, clf.emb_dim, clf.hidden,
                                  dropout=clf.dropout, bidirectional=clf.bidirectional)
        clf.model_.load_state_dict(ckpt["state"])
        clf.model_.eval()
        return clf
