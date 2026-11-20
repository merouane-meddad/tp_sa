"""Text cleaning and tokenisation (course Ch3, "Pré-traitement").

Two levels are provided:

* :func:`clean_text` - light normalisation used everywhere (lowercase,
  URLs -> ``URL``, mentions -> ``USER``, ``#`` removed, emoticons replaced
  by ``EMOPOS`` / ``EMONEG`` so that the tokenizer does not throw them away,
  letter repetitions shortened: "soooo" -> "soo");
* :class:`TextNormalizer` - the three course steps on top of it: stop word
  removal, lemmatisation, stemming. Negation words are never treated as stop
  words, because "not good" and "good" have opposite polarities.
"""
import re
from functools import lru_cache

import pandas as pd

from .patterns import EMO_NEG_RE, EMO_POS_RE, HASHTAG_RE, MENTION_RE, URL_RE

_REPEAT_RE = re.compile(r"(\w)\1{2,}")
_SPACE_RE = re.compile(r"\s+")
# words (with inner apostrophe), special tokens, and single emoji / symbols
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|EMOPOS|EMONEG|URL|USER|[^\w\s]", re.UNICODE)
_EMOJI_LIKE_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿❤♥]")

NEGATIONS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "cannot", "without", "ain't", "aren't", "can't", "couldn't",
    "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't",
    "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't",
    "weren't", "won't", "wouldn't",
}


def clean_text(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" URL ", text)
    text = EMO_POS_RE.sub(" EMOPOS ", text)
    text = EMO_NEG_RE.sub(" EMONEG ", text)
    text = MENTION_RE.sub(" USER ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = text.replace("’", "'").replace("‘", "'")
    # lowercase everything except the special tokens
    text = " ".join(t if t in {"URL", "USER", "EMOPOS", "EMONEG"} else t.lower()
                    for t in text.split())
    text = _REPEAT_RE.sub(r"\1\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str):
    """Split a cleaned text into words, special tokens and emoji.

    Plain punctuation is dropped (it is studied as metadata in Task 3),
    emoji are kept as tokens.
    """
    return [t for t in TOKEN_RE.findall(text)
            if t[0].isalnum() or _EMOJI_LIKE_RE.match(t)]


@lru_cache(maxsize=1)
def _stop_words():
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words("english"))
    except LookupError:  # nltk data not downloaded: fall back to sklearn's list
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        words = set(ENGLISH_STOP_WORDS)
    words |= {w.replace("'", "") for w in words}
    keep = NEGATIONS | {w.replace("'", "") for w in NEGATIONS}
    # NLTK also lists truncated negations ("didn", "wasn", ...): keep them too
    keep |= {w[:-2] for w in NEGATIONS if w.endswith("n't")}
    return frozenset(words - keep)


@lru_cache(maxsize=1)
def _lemmatizer():
    from nltk.stem import WordNetLemmatizer
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import SnowballStemmer
    return SnowballStemmer("english")


@lru_cache(maxsize=200_000)
def lemmatize(word: str) -> str:
    lem = _lemmatizer()
    return lem.lemmatize(lem.lemmatize(word, pos="v"), pos="n")


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    return _stemmer().stem(word)


class TextNormalizer:
    """Callable tokenizer: clean -> tokenize -> stop words -> lemma / stem.

    Parameters
    ----------
    stop_words : bool
        Remove English stop words (negations are kept).
    reduce : {None, "lemma", "stem"}
        Reduce words to their lemma (WordNet) or stem (Snowball).
    """

    SPECIAL = {"URL", "USER", "EMOPOS", "EMONEG"}

    def __init__(self, stop_words=False, reduce=None):
        if reduce not in (None, "lemma", "stem"):
            raise ValueError("reduce must be None, 'lemma' or 'stem'")
        self.stop_words = stop_words
        self.reduce = reduce

    def __call__(self, text):
        tokens = tokenize(clean_text(text))
        if self.stop_words:
            sw = _stop_words()
            tokens = [t for t in tokens if t not in sw]
        if self.reduce:
            f = lemmatize if self.reduce == "lemma" else stem
            tokens = [t if t in self.SPECIAL or not t.isalpha() else f(t) for t in tokens]
        return tokens

    def __repr__(self):
        return f"TextNormalizer(stop_words={self.stop_words}, reduce={self.reduce!r})"


def preprocess_dataframe(df: pd.DataFrame, normalizer=None) -> pd.DataFrame:
    """Return a copy of a tweet dataframe with its text cleaned.

    With a ``normalizer`` the text is also normalised and re-joined with spaces.
    """
    processed = df.copy()
    if normalizer is None:
        processed["text"] = processed["text"].map(clean_text)
    else:
        processed["text"] = processed["text"].map(lambda t: " ".join(normalizer(t)))
    return processed
