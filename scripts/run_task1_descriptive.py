from tweet_sentiment.config import FIGURES, TABLES
from tweet_sentiment.data import load_train
from tweet_sentiment.descriptive import class_distribution, length_stats, plot_class_distribution

df = load_train()
print(df.head())
class_distribution(df).to_csv(TABLES / "class_distribution.csv")
length_stats(df).to_csv(TABLES / "length_stats.csv")
plot_class_distribution(df, FIGURES / "class_distribution.png")
