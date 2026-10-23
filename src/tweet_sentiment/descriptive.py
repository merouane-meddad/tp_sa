"""Task 1 - descriptive statistics and plots."""
import matplotlib.pyplot as plt
import seaborn as sns


def class_distribution(df):
    return df["polarity"].value_counts(normalize=True).sort_index()


def length_stats(df):
    d = df.assign(n_chars=df["text"].str.len(), n_words=df["text"].str.split().str.len())
    return d.groupby("polarity")[["n_chars", "n_words"]].describe()


def plot_class_distribution(df, out_path):
    ax = sns.countplot(x="polarity", data=df)
    ax.set_title("Polarity distribution")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# TODO: top words / n-grams per class, word clouds, hashtags, mentions
