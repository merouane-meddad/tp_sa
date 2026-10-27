"""Text cleaning and tokenisation."""
import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" USER ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a tweet dataframe with its text cleaned."""
    processed = df.copy()
    processed["text"] = processed["text"].map(clean_text)
    return processed
