"""Course feature extractors (Ch3): Delta TF-IDF and weighted word2vec.

Delta TF-IDF "améliorée" (Othman et al., 2019), smoothing alpha = 0.5::

    V(w, t) = C(w, t) * log2( |N| * (P_w + alpha) / (|P| * (N_w + alpha)) )

C(w, t) is the count of w in tweet t, |P| / |N| the number of positive /
negative tweets, P_w / N_w the number of positive / negative tweets that
contain w. (Putting alpha inside the parentheses reproduces the slide's
table: with |N|/|P| = 0.59, (P_w, N_w) = (100, 0), (2, 0), (100, 1) give
6.90, 1.57, 5.32.)

With three classes the course formula is applied either to positive vs
negative only (``scheme="pos_neg"``, neutral tweets are ignored when the
weights are estimated), or once per class against the two others
(``scheme="ovr"``, one block of features per class).
"""
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing import TextNormalizer


def delta_idf(X_counts, y, positive=1, negative=-1, alpha=0.5):
    """Per-term improved Delta IDF, log2(|N|(P_w+a) / (|P|(N_w+a)))."""
    y = np.asarray(y)
    pos = y == positive if np.isscalar(positive) else np.isin(y, positive)
    neg = y == negative if np.isscalar(negative) else np.isin(y, negative)
    present = (X_counts > 0).astype(np.float64)
    p_w = np.asarray(present[pos].sum(axis=0)).ravel()
    n_w = np.asarray(present[neg].sum(axis=0)).ravel()
    n_pos, n_neg = pos.sum(), neg.sum()
    return np.log2(n_neg * (p_w + alpha) / (n_pos * (n_w + alpha)))


class DeltaTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Count vectorizer re-weighted by the improved Delta TF-IDF.

    Parameters
    ----------
    scheme : {"pos_neg", "ovr"}
    alpha : float
        Smoothing parameter of the improved version (0.5 in the course).
    sublinear_tf : bool
        Use 1 + log(count) instead of the raw count C(w, t).
    normalize : bool
        L2-normalise each row (keeps linear models well conditioned).
    """

    def __init__(self, tokenizer=None, ngram_range=(1, 2), min_df=2,
                 scheme="pos_neg", alpha=0.5, sublinear_tf=True, normalize=True):
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.scheme = scheme
        self.alpha = alpha
        self.sublinear_tf = sublinear_tf
        self.normalize = normalize

    def fit(self, X, y):
        self.vectorizer_ = CountVectorizer(
            tokenizer=self.tokenizer or TextNormalizer(), lowercase=False,
            token_pattern=None, ngram_range=self.ngram_range, min_df=self.min_df)
        counts = self.vectorizer_.fit_transform(X)
        y = np.asarray(y)
        if self.scheme == "pos_neg":
            self.weights_ = [delta_idf(counts, y, 1, -1, self.alpha)]
        elif self.scheme == "ovr":
            self.weights_ = [delta_idf(counts, y, c, [o for o in (-1, 0, 1) if o != c],
                                       self.alpha) for c in (-1, 0, 1)]
        else:
            raise ValueError("scheme must be 'pos_neg' or 'ovr'")
        return self

    def transform(self, X):
        counts = self.vectorizer_.transform(X).astype(np.float64)
        if self.sublinear_tf:
            counts.data = 1.0 + np.log(counts.data)
        blocks = [counts @ sp.diags(w) for w in self.weights_]
        out = sp.hstack(blocks).tocsr()
        if self.normalize:
            norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
            norms[norms == 0] = 1.0
            out = sp.diags(1.0 / norms) @ out
        return out

    def term_scores(self):
        """DataFrame of terms with their positive-vs-negative Delta IDF."""
        import pandas as pd
        return pd.DataFrame({"term": self.vectorizer_.get_feature_names_out(),
                             "delta_idf": self.weights_[0] if self.scheme == "pos_neg"
                             else self.weights_[2] - self.weights_[0]})


class WeightedWord2Vec(BaseEstimator, TransformerMixin):
    """Tweet vector = Delta-TF-IDF weighted mean of word2vec vectors (course).

        V_t = sum_i v_{w_i} * wgt(w_i, t) / sum_i |wgt(w_i, t)|

    word2vec is trained on the training tweets only (gensim). The weight is
    the improved Delta TF-IDF (positive vs negative), so that words of
    opposite polarity pull the tweet vector in opposite directions. The
    course formula divides by the sum of the weights; because those weights
    are signed we divide by the sum of their absolute values, which avoids
    dividing by a number close to zero. ``add_mean=True`` appends the plain
    (unweighted) mean of the vectors, which keeps neutral words informative.
    """

    def __init__(self, tokenizer=None, vector_size=100, window=5, min_count=2,
                 epochs=10, alpha=0.5, add_mean=True, w2v_model=None, random_state=42):
        self.tokenizer = tokenizer
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.alpha = alpha
        self.add_mean = add_mean
        self.w2v_model = w2v_model
        self.random_state = random_state

    def fit(self, X, y):
        tok = self.tokenizer or TextNormalizer()
        docs = [tok(t) for t in X]
        if self.w2v_model is not None:
            self.wv_ = self.w2v_model.wv
        else:
            self.wv_ = train_word2vec(docs, self.vector_size, self.window,
                                      self.min_count, self.epochs, self.random_state).wv
        vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        rows, cols = [], []
        for r, d in enumerate(docs):
            for w in set(d):
                rows.append(r)
                cols.append(vocab[w])
        present = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab)))
        weights = delta_idf(present, y, 1, -1, self.alpha)
        self.word_weight_ = {w: weights[i] for w, i in vocab.items()}
        return self

    def transform(self, X):
        tok = self.tokenizer or TextNormalizer()
        dim = self.wv_.vector_size
        out = np.zeros((len(X), dim * (2 if self.add_mean else 1)), dtype=np.float32)
        for r, text in enumerate(X):
            words = [w for w in tok(text) if w in self.wv_.key_to_index]
            if not words:
                continue
            vecs = np.stack([self.wv_[w] for w in words])
            wgt = np.array([self.word_weight_.get(w, 0.0) for w in words])
            denom = np.abs(wgt).sum()
            if denom > 0:
                out[r, :dim] = (vecs * wgt[:, None]).sum(axis=0) / denom
            if self.add_mean:
                out[r, dim:] = vecs.mean(axis=0)
        return out


def train_word2vec(docs, vector_size=100, window=5, min_count=2, epochs=10, seed=42):
    """Skip-gram word2vec trained on tokenised tweets."""
    from gensim.models import Word2Vec
    return Word2Vec(sentences=docs, vector_size=vector_size, window=window,
                    min_count=min_count, sg=1, epochs=epochs, seed=seed, workers=1)
