"""Loading of the raw tweet files."""
import pandas as pd
from .config import TRAIN_FILE, TEST_FILE


def load_tweets(path, sep="\t"):
    """Load a tweets file with columns: idf, polarity, text.

    ASSUMPTION: tab-separated. Check the real file format and adjust `sep`.
    """
    df = pd.read_csv(
        path, sep=sep, header=None, names=["idf", "polarity", "text"],
        quoting=3, encoding="utf-8", on_bad_lines="warn",
    )
    df["polarity"] = df["polarity"].astype(int)
    return df


def load_train():
    return load_tweets(TRAIN_FILE)


def load_test():
    return load_tweets(TEST_FILE)
