"""Text cleaning and tokenisation."""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" USER ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
