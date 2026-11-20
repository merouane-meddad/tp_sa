"""Task 3 - metadata features: URLs, emoticons / emoji, punctuation, irony
cues, censored insults - and their statistical comparison across classes.

Every detector works on the raw (encoding-repaired) text, because the
cleaner used by the models removes most of these markers.
"""
from collections import Counter

import emoji
import numpy as np
import pandas as pd
from scipy import stats

from .patterns import (CAPS_WORD_RE, CENSOR_CANDIDATE_RE, ELONGATED_RE, EMO_NEG_RE,
                       EMO_POS_RE, EMOJI_NEG, EMOJI_POS, HASHTAG_RE, IRONY_RE,
                       MASKED_WORD_RE, MENTION_RE, SCARE_QUOTES_RE, SWEAR_RE, URL_RE,
                       is_censored_swear)

# Feature groups, used for the ablation study in Task 3.
FEATURE_GROUPS = {
    "urls": ["n_urls"],
    "emoticons": ["n_emo_pos", "n_emo_neg", "n_emoji", "n_emoji_pos", "n_emoji_neg"],
    "punctuation": ["n_exclam", "n_question", "n_ellipsis", "multi_exclam",
                    "multi_question", "interrobang"],
    "emphasis": ["n_caps_words", "n_elongated"],
    "irony": ["irony_marker", "n_scare_quotes"],
    "insults": ["n_censored", "n_swear"],
    "twitter": ["n_hashtags", "n_mentions"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


# Characters that the emoji library counts as emoji but that are ordinary
# typographic symbols in tweets (mostly in neutral, advertising tweets).
NOT_EMOJI = {"™", "©", "®", "‼", "↔", "↩", "▶", "◀", "‼️", "™️", "©️", "®️"}


def _emoji_chars(text):
    return [e["emoji"] for e in emoji.emoji_list(text) if e["emoji"] not in NOT_EMOJI]


def _first_char(e):
    return e[0]


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with one numeric column per metadata feature."""
    t = df["text"].astype(str)
    no_url = t.str.replace(URL_RE, " ", regex=True)
    emojis = t.map(_emoji_chars)
    out = df.copy()

    out["n_urls"] = t.str.count(URL_RE)
    out["n_emo_pos"] = no_url.str.count(EMO_POS_RE)
    out["n_emo_neg"] = no_url.str.count(EMO_NEG_RE)
    out["n_emoji"] = emojis.map(len)
    out["n_emoji_pos"] = emojis.map(lambda es: sum(_first_char(e) in EMOJI_POS for e in es))
    out["n_emoji_neg"] = emojis.map(lambda es: sum(_first_char(e) in EMOJI_NEG for e in es))

    out["n_exclam"] = no_url.str.count("!")
    out["n_question"] = no_url.str.count(r"\?")
    out["n_ellipsis"] = no_url.str.count(r"\.{3,}|…")
    out["multi_exclam"] = no_url.str.contains(r"!{2,}").astype(int)
    out["multi_question"] = no_url.str.contains(r"\?{2,}").astype(int)
    out["interrobang"] = no_url.str.contains(r"\?!|!\?").astype(int)

    no_handles = no_url.str.replace(MENTION_RE, " ", regex=True).str.replace(r"#\w+", " ", regex=True)
    out["n_caps_words"] = no_handles.str.count(CAPS_WORD_RE)
    out["n_elongated"] = no_handles.str.count(ELONGATED_RE)

    out["irony_marker"] = t.str.contains(IRONY_RE).astype(int)
    out["n_scare_quotes"] = t.str.count(SCARE_QUOTES_RE)

    out["n_censored"] = no_url.map(
        lambda s: sum(is_censored_swear(m) for m in CENSOR_CANDIDATE_RE.findall(s))
        + len(MASKED_WORD_RE.findall(s)))
    out["n_swear"] = no_url.str.count(SWEAR_RE)

    out["n_hashtags"] = t.str.count(HASHTAG_RE)
    out["n_mentions"] = t.str.count(MENTION_RE)
    return out


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def compare_across_classes(meta: pd.DataFrame, features=FEATURES):
    """One row per feature: presence rate and mean per class + tests.

    * chi-square test of independence between "feature present (>0)" and the
      polarity, with Cramér's V as effect size (0.1 small, 0.3 medium);
    * Kruskal-Wallis test on the counts (non-parametric: counts are very
      skewed, so an ANOVA would be inappropriate).
    With ~50 000 tweets almost every p-value is tiny, so the effect size is
    what tells whether a difference matters.
    """
    rows = []
    for f in features:
        present = (meta[f] > 0)
        table = pd.crosstab(present, meta["polarity"])
        row = {"feature": f}
        for c in (-1, 0, 1):
            row[f"pct_present_{c}"] = 100 * present[meta["polarity"] == c].mean()
        for c in (-1, 0, 1):
            row[f"mean_{c}"] = meta.loc[meta["polarity"] == c, f].mean()
        if table.shape[0] == 2:
            chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
            row.update(chi2=chi2, chi2_p=p, cramers_v=cramers_v(table))
        groups = [meta.loc[meta["polarity"] == c, f] for c in (-1, 0, 1)]
        try:
            row["kruskal_p"] = stats.kruskal(*groups).pvalue
        except ValueError:  # all values identical
            row["kruskal_p"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def emoji_polarity_table(df, min_count=20, alpha=0.5):
    """Most frequent emoji / emoticons with their class distribution.

    ``log2_pos_neg`` is the Delta-IDF of the course applied to the symbol:
    > 0 means more frequent in positive tweets, < 0 in negative ones.
    """
    rows = []
    n_pos = (df["polarity"] == 1).sum()
    n_neg = (df["polarity"] == -1).sum()
    symbols = df["text"].map(lambda s: set(_emoji_chars(s))
                             | set(EMO_POS_RE.findall(URL_RE.sub(" ", s)))
                             | set(EMO_NEG_RE.findall(URL_RE.sub(" ", s))))
    counts = {c: Counter(e for es in symbols[df["polarity"] == c] for e in es) for c in (-1, 0, 1)}
    total = counts[-1] + counts[0] + counts[1]
    for sym, n in total.most_common():
        if n < min_count:
            break
        p, ng = counts[1][sym], counts[-1][sym]
        rows.append({"symbol": sym, "tweets": n, "negative": ng, "neutral": counts[0][sym],
                     "positive": p,
                     "log2_pos_neg": np.log2(n_neg * (p + alpha) / (n_pos * (ng + alpha)))})
    return pd.DataFrame(rows)


def polarity_contradictions(meta):
    """How often a tweet's label contradicts its emoticons / emoji (an
    imperfect proxy for irony, or for an emoticon that softens a complaint)."""
    pos_sym = (meta["n_emo_pos"] + meta["n_emoji_pos"]) > 0
    neg_sym = (meta["n_emo_neg"] + meta["n_emoji_neg"]) > 0
    rows = []
    for c in (-1, 0, 1):
        m = meta["polarity"] == c
        rows.append({"polarity": c, "tweets": int(m.sum()),
                     "pct_with_positive_symbol": 100 * pos_sym[m].mean(),
                     "pct_with_negative_symbol": 100 * neg_sym[m].mean(),
                     "pct_with_irony_marker": 100 * meta.loc[m, "irony_marker"].mean()})
    return pd.DataFrame(rows).set_index("polarity")


def build_text_meta_model(normalizer=None, C=2.0, features=FEATURES, use_text=True):
    """Logistic regression on TF-IDF n-grams + scaled metadata features.

    The model expects a DataFrame with a ``text`` column and the metadata
    columns produced by :func:`extract_metadata`. Counts are log-scaled
    (log1p) then standardised so that they are comparable with TF-IDF values.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    from .config import RANDOM_STATE
    from .models import tfidf

    parts = []
    if use_text:
        parts.append(("text", tfidf(normalizer), "text"))
    if features:
        parts.append(("meta", Pipeline([("log", FunctionTransformer(np.log1p)),
                                        ("scale", StandardScaler())]), list(features)))
    return Pipeline([
        ("features", ColumnTransformer(parts)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
