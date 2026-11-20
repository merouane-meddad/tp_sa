"""Task 1 - descriptive statistics and plots.

All the analyses of ``notebooks/descriptive_analysis.ipynb`` live here so
that ``scripts/run_task1_descriptive.py`` reproduces every table and figure.
"""
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

from .config import LABEL_NAMES, LABELS  # noqa: E402
from .data import duplicate_report  # noqa: E402
from .features import DeltaTfidfVectorizer  # noqa: E402
from .metadata import extract_metadata  # noqa: E402
from .preprocessing import TextNormalizer  # noqa: E402

PALETTE = {-1: "#C44E52", 0: "#8C8C8C", 1: "#4C72B0"}
sns.set_theme(style="whitegrid", context="notebook")


def add_lengths(df):
    return df.assign(n_chars=df["text"].str.len(), n_words=df["text"].str.split().str.len())


# ---------------------------------------------------------------- quality
def data_quality(df):
    quality = pd.DataFrame({
        "dtype": df[["idf", "polarity", "text"]].dtypes.astype(str),
        "missing_values": df[["idf", "polarity", "text"]].isna().sum(),
    })
    return quality, duplicate_report(df)


# ---------------------------------------------------------------- classes
def class_distribution(df):
    return df["polarity"].value_counts(normalize=True).sort_index()


def class_table(train, test=None):
    out = pd.DataFrame({"train_tweets": train["polarity"].value_counts().sort_index()})
    out["train_pct"] = 100 * out["train_tweets"] / out["train_tweets"].sum()
    if test is not None:
        out["test_tweets"] = test["polarity"].value_counts().sort_index()
        out["test_pct"] = 100 * out["test_tweets"] / out["test_tweets"].sum()
    out.index = [f"{c} ({LABEL_NAMES[c]})" for c in out.index]
    return out


def plot_class_distribution(train, out_path, test=None):
    data = [train.assign(split="train")]
    if test is not None:
        data.append(test.assign(split="test"))
    d = pd.concat(data)
    pct = (d.groupby("split")["polarity"].value_counts(normalize=True).mul(100)
           .rename("pct").reset_index())
    ax = sns.barplot(data=pct, x="polarity", y="pct", hue="split",
                     palette=["#4C72B0", "#DD8452"])
    ax.set(title="Polarity distribution, train vs test",
           xlabel="Polarity (-1 negative, 0 neutral, +1 positive)", ylabel="Tweets (%)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------- lengths
def length_stats(df):
    return add_lengths(df).groupby("polarity")[["n_chars", "n_words"]].describe()


def length_tests(df):
    """Kruskal-Wallis test and effect size (epsilon squared) on lengths."""
    from scipy import stats
    d = add_lengths(df)
    rows = []
    for col in ("n_chars", "n_words"):
        groups = [d.loc[d["polarity"] == c, col] for c in LABELS]
        h, p = stats.kruskal(*groups)
        rows.append({"variable": col, "kruskal_H": h, "p_value": p,
                     "epsilon_squared": h / (len(d) - 1)})
    return pd.DataFrame(rows).set_index("variable")


def plot_lengths(df, out_path):
    d = add_lengths(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, lab in zip(axes, ("n_chars", "n_words"), ("Characters", "Words")):
        sns.boxplot(data=d, x="polarity", y=col, hue="polarity", palette=PALETTE,
                    showfliers=False, legend=False, ax=ax)
        ax.set(title=f"{lab} per tweet", xlabel="Polarity", ylabel=lab)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------- words
def top_ngrams(df, n=1, k=15, normalizer=None):
    """Most frequent n-grams per class (stop words removed), per 1 000 tweets."""
    normalizer = normalizer or TextNormalizer(stop_words=True)
    vec = CountVectorizer(tokenizer=normalizer, lowercase=False, token_pattern=None,
                          ngram_range=(n, n), min_df=5)
    counts = vec.fit_transform(df["text"])
    terms = vec.get_feature_names_out()
    labels = df["polarity"].to_numpy()
    tables = {}
    for c in LABELS:
        cc = np.asarray(counts[labels == c].sum(axis=0)).ravel()
        top = cc.argsort()[::-1]
        top = [i for i in top if not set(terms[i].split()) & {"URL", "USER", "rt", "via"}][:k]
        tables[c] = pd.DataFrame({"term": terms[top], "occurrences": cc[top].astype(int),
                                  "per_1k_tweets": 1000 * cc[top] / (labels == c).sum()})
    return tables


def polar_terms(df, k=20):
    """Terms with the highest / lowest improved Delta TF-IDF (course Ch3)."""
    vec = DeltaTfidfVectorizer(tokenizer=TextNormalizer(stop_words=True),
                               ngram_range=(1, 1), min_df=20, scheme="pos_neg")
    vec.fit(df["text"], df["polarity"])
    scores = vec.term_scores()
    scores = scores[~scores["term"].isin({"URL", "USER"})]
    return (scores.nlargest(k, "delta_idf").reset_index(drop=True),
            scores.nsmallest(k, "delta_idf").reset_index(drop=True))


def plot_top_terms(tables, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, c in zip(axes, LABELS):
        t = tables[c].sort_values("per_1k_tweets")
        ax.barh(t["term"], t["per_1k_tweets"], color=PALETTE[c])
        ax.set(title=f"{c} ({LABEL_NAMES[c]})", xlabel="Occurrences per 1 000 tweets")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_polar_terms(pos, neg, out_path):
    d = pd.concat([neg.iloc[::-1], pos.iloc[::-1]])
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(d["term"], d["delta_idf"],
            color=[PALETTE[1] if v > 0 else PALETTE[-1] for v in d["delta_idf"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Most polar words (improved Delta TF-IDF, positive vs negative)",
           xlabel="log2( |N|(P_w+0.5) / (|P|(N_w+0.5)) )")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_wordclouds(df, out_path, normalizer=None):
    from wordcloud import WordCloud
    normalizer = normalizer or TextNormalizer(stop_words=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, c in zip(axes, LABELS):
        freq = Counter(w for t in df.loc[df["polarity"] == c, "text"] for w in normalizer(t)
                       if w.isalpha() and w not in {"URL", "USER", "EMOPOS", "EMONEG"})
        wc = WordCloud(width=600, height=400, background_color="white",
                       random_state=0, colormap={-1: "Reds", 0: "Greys", 1: "Blues"}[c])
        ax.imshow(wc.generate_from_frequencies(freq), interpolation="bilinear")
        ax.set_title(f"{c} ({LABEL_NAMES[c]})")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------- hashtags
TAG_PATTERNS = {"hashtag": re.compile(r"(?<!\w)#(\w+)"), "mention": re.compile(r"(?<!\w)@(\w+)")}


def tags_table(df, k=10):
    rows = []
    for c, g in df.groupby("polarity"):
        for kind, pattern in TAG_PATTERNS.items():
            values = [v.lower() for s in g["text"] for v in pattern.findall(s)]
            rows.append({"polarity": c, "kind": kind,
                         "tweets_with_any_pct": 100 * g["text"].map(lambda s: bool(pattern.search(s))).mean(),
                         "top_values": ", ".join(f"{v} ({n})" for v, n in Counter(values).most_common(k))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- markers
MARKERS = {"URL": "n_urls", "ASCII emoticon": ["n_emo_pos", "n_emo_neg"], "emoji": "n_emoji",
           "exclamation": "n_exclam", "question mark": "n_question", "ellipsis": "n_ellipsis",
           "all-caps word": "n_caps_words"}


def marker_rates(df):
    """% of tweets of each class containing each marker at least once."""
    meta = extract_metadata(df)
    rows = {}
    for name, cols in MARKERS.items():
        cols = [cols] if isinstance(cols, str) else cols
        present = meta[cols].sum(axis=1) > 0
        rows[name] = present.groupby(meta["polarity"]).mean() * 100
    return pd.DataFrame(rows).T


def plot_marker_rates(rates, out_path):
    d = rates.reset_index(names="marker").melt(id_vars="marker", var_name="polarity",
                                               value_name="pct")
    ax = sns.barplot(data=d, x="marker", y="pct", hue="polarity", palette=PALETTE)
    ax.set(title="Tweets containing each marker", xlabel="", ylabel="Tweets (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
