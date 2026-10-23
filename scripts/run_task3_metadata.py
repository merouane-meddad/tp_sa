from tweet_sentiment.config import TABLES
from tweet_sentiment.data import load_train
from tweet_sentiment.metadata import extract_metadata

meta = extract_metadata(load_train())
cols = [c for c in meta.columns if c.startswith("n_") or c == "irony_cue"]
summary = meta.groupby("polarity")[cols].mean()
print(summary)
summary.to_csv(TABLES / "metadata_by_polarity.csv")
