"""Task 3 - metadata features: URLs, emoticons, punctuation, irony cues, censored insults."""
import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOTICON_RE = re.compile(r"(?::|;|=|8)[-o^']?[)(DPp/\\|]|<3|\^_\^")
CENSORED_RE = re.compile(r"\b\w*[*@#$%]{2,}\w*\b")  # e.g. f**k, s#it
IRONY_CUES = re.compile(r"\b(yeah right|sure|great|thanks a lot)\b|#irony|#sarcasm|/s\b", re.I)


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    t = df["text"].astype(str)
    out = df.copy()
    out["n_urls"] = t.str.count(URL_RE)
    out["n_emoticons"] = t.str.count(EMOTICON_RE)
    out["n_exclam"] = t.str.count("!")
    out["n_question"] = t.str.count(r"\?")
    out["n_ellipsis"] = t.str.count(r"\.{3,}")
    out["n_caps_words"] = t.str.count(r"\b[A-Z]{3,}\b")
    out["n_censored"] = t.str.count(CENSORED_RE)
    out["irony_cue"] = t.str.contains(IRONY_CUES).astype(int)
    return out
