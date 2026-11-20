"""Loading of the raw tweet files.

The raw files are tab separated (idf, polarity, text, optional empty field).
They contain several encoding artefacts that are repaired here, because they
are not part of what the author wrote:

* about 24 % of the training texts are wrapped in CSV quotes, with inner
  quotes doubled (``"He said ""hi"" today"``);
* about 11 % contain literal JSON escapes (``\\u2019`` for ``'``, ``\\u002c``
  for ``,``), which otherwise show up as the "words" ``u2019`` / ``u002c``;
* about 4 % contain HTML entities (``&amp;``, ``&lt;``, ``&quot;``);
* some contain escaped quotes or slashes (``\\"``, ``\\/``);
* about 10 % of the TEST texts are "mojibake": UTF-8 decoded as
  Windows-1252, so that emoji appear as ``ðŸ˜‚`` and "…" as ``â€¦``.
"""
import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_FILE, TRAIN_FILE, VAL_SIZE

_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_MOJIBAKE_RE = re.compile(r"â€|ðŸ|Ã[\x80-\xbf]|Â[\x80-\xbf ]|â[\x80-\x9f]")


def _cp1252_byte(ch):
    try:
        return ch.encode("cp1252")
    except UnicodeEncodeError:
        if ord(ch) < 256:          # bytes undefined in cp1252 kept as latin-1
            return bytes([ord(ch)])
        raise


def fix_mojibake(text: str) -> str:
    """Undo UTF-8 text wrongly decoded as Windows-1252 ("ðŸ˜‚" -> "😂")."""
    if not _MOJIBAKE_RE.search(text):
        return text
    try:
        return b"".join(_cp1252_byte(ch) for ch in text).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def fix_encoding(text: str) -> str:
    """Undo CSV quoting, JSON escapes and HTML entities in a raw tweet."""
    text = fix_mojibake(str(text).strip("\r\n"))
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')
    text = text.replace('\\"', '"').replace("\\/", "/")
    text = _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
    # A few texts were quoted twice, which leaves runs of quotes behind.
    text = re.sub(r'"{2,}', '"', text)
    text = html.unescape(text)
    return text.strip()


def load_tweets(path, sep="\t", repair=True):
    """Load a tweets file with columns: idf, polarity, text.

    The optional (always empty) field after the text is ignored, and empty
    lines are skipped. With ``repair=True`` the text goes through
    :func:`fix_encoding`.
    """
    df = pd.read_csv(
        path, sep=sep, header=None, names=["idf", "polarity", "text"],
        usecols=[0, 1, 2], quoting=3, encoding="utf-8", on_bad_lines="warn",
        dtype={"idf": "int64", "text": "string"},
    )
    df = df.dropna(subset=["polarity", "text"])
    df["polarity"] = df["polarity"].astype(int)
    df["text"] = df["text"].astype(str)
    if repair:
        df["text"] = df["text"].map(fix_encoding)
    return df.reset_index(drop=True)


def load_train(repair=True):
    return load_tweets(TRAIN_FILE, repair=repair)


def load_test(repair=True):
    return load_tweets(TEST_FILE, repair=repair)


def duplicate_report(df):
    """Count repeated IDs / texts and how many of them carry conflicting labels."""
    by_id = df.groupby("idf")["polarity"].nunique()
    by_text = df.groupby("text")["polarity"].nunique()
    return pd.Series({
        "rows": len(df),
        "exact_duplicate_rows": int(df.duplicated(["idf", "polarity", "text"]).sum()),
        "ids_seen_more_than_once": int((df["idf"].value_counts() > 1).sum()),
        "ids_with_conflicting_labels": int((by_id > 1).sum()),
        "texts_seen_more_than_once": int((df["text"].value_counts() > 1).sum()),
        "texts_with_conflicting_labels": int((by_text > 1).sum()),
        "texts_shorter_than_3_chars": int((df["text"].str.len() < 3).sum()),
    })


def deduplicate(df):
    """Remove duplicates and contradictory labels before modelling.

    * exact duplicate rows are collapsed to one;
    * IDs or texts annotated with different polarities are dropped entirely
      (we cannot know which label is right);
    * remaining repeated texts are kept once, so that the same tweet cannot
      be in both the training and the validation fold;
    * texts shorter than 3 characters (e.g. the tweet "0") are dropped.
    """
    out = df.drop_duplicates(["idf", "polarity", "text"])
    bad_ids = out.groupby("idf")["polarity"].nunique()
    bad_texts = out.groupby("text")["polarity"].nunique()
    out = out[~out["idf"].isin(bad_ids[bad_ids > 1].index)]
    out = out[~out["text"].isin(bad_texts[bad_texts > 1].index)]
    out = out.drop_duplicates("text").drop_duplicates("idf")
    out = out[out["text"].str.len() >= 3]
    return out.reset_index(drop=True)


def train_val_split(df, val_size=VAL_SIZE):
    """Stratified train/validation split shared by every model."""
    return train_test_split(df, test_size=val_size, stratify=df["polarity"],
                            random_state=RANDOM_STATE)


def load_model_data():
    """Return (train, val, test) DataFrames used by all the Task 2/3 models."""
    train, val = train_val_split(deduplicate(load_train()))
    return train.reset_index(drop=True), val.reset_index(drop=True), load_test()
