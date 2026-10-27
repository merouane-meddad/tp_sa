from tweet_sentiment.config import PROCESSED
from tweet_sentiment.data import load_test, load_train
from tweet_sentiment.preprocessing import preprocess_dataframe


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for split, load in (("train", load_train), ("test", load_test)):
        df = preprocess_dataframe(load())
        output = PROCESSED / f"{split}_processed.csv"
        df.to_csv(output, index=False, encoding="utf-8")
        print(f"Saved {len(df):,} {split} tweets to {output}")


if __name__ == "__main__":
    main()
