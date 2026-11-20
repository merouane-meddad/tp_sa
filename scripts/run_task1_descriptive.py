"""Task 1 - descriptive analysis: writes tables/task1_* and figures/task1_*."""
from tweet_sentiment.config import FIGURES, TABLES, ensure_output_dirs
from tweet_sentiment.data import load_test, load_train
from tweet_sentiment import descriptive as d


def main():
    ensure_output_dirs()
    train, test = load_train(), load_test()
    print(f"train: {len(train):,} tweets, test: {len(test):,} tweets")

    quality, dups = d.data_quality(train)
    print(quality, dups, sep="\n\n")
    dups.to_csv(TABLES / "task1_duplicates.csv", header=["value"])

    classes = d.class_table(train, test)
    print(classes.round(2))
    classes.to_csv(TABLES / "task1_class_distribution.csv")
    d.plot_class_distribution(train, FIGURES / "task1_class_distribution.png", test)

    d.length_stats(train).to_csv(TABLES / "task1_length_stats.csv")
    print(d.length_tests(train))
    d.length_tests(train).to_csv(TABLES / "task1_length_tests.csv")
    d.plot_lengths(train, FIGURES / "task1_lengths.png")

    for n, name in ((1, "unigrams"), (2, "bigrams")):
        tables = d.top_ngrams(train, n=n)
        for c, t in tables.items():
            t.to_csv(TABLES / f"task1_top_{name}_{c}.csv", index=False)
        d.plot_top_terms(tables, f"Most frequent {name} (stop words removed)",
                         FIGURES / f"task1_top_{name}.png")

    pos, neg = d.polar_terms(train)
    pos.to_csv(TABLES / "task1_polar_terms_positive.csv", index=False)
    neg.to_csv(TABLES / "task1_polar_terms_negative.csv", index=False)
    d.plot_polar_terms(pos, neg, FIGURES / "task1_polar_terms.png")
    d.plot_wordclouds(train, FIGURES / "task1_wordclouds.png")

    d.tags_table(train).to_csv(TABLES / "task1_hashtags_mentions.csv", index=False)
    rates = d.marker_rates(train)
    print(rates.round(1))
    rates.to_csv(TABLES / "task1_marker_rates.csv")
    d.plot_marker_rates(rates, FIGURES / "task1_marker_rates.png")


if __name__ == "__main__":
    main()
