"""Regenerates descriptive_analysis.ipynb from the functions of descriptive.py.

Run from the repository root:  python notebooks/build_descriptive_notebook.py
then execute the notebook (Jupyter or `jupyter nbconvert --execute --inplace`).
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
INTERP = Path(__file__).with_name("descriptive_interpretations.md")


def interpretations():
    """Interpretation paragraphs keyed by '## key' headers in the .md file."""
    out, key = {}, None
    if INTERP.exists():
        for line in INTERP.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                key = line[3:].strip()
                out[key] = []
            elif key:
                out[key].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


I = interpretations()


def interp(key):
    return md("**Interpretation.** " + I.get(key, "_to be written after execution_"))


cells = [
    md("# Descriptive analysis of tweet sentiment data (Task 1)\n\n"
       "This notebook describes the **training split** (class balance, length, frequent and polar "
       "words, hashtags and mentions, visible markers) and compares its class balance with the "
       "test split. The test texts are not otherwise analysed, so they stay untouched for the "
       "evaluation of the models.\n\n"
       "All computations come from `tweet_sentiment.descriptive`, so "
       "`scripts/run_task1_descriptive.py` reproduces the same tables and figures in `reports/`.\n\n"
       "**Text repair.** The loader repairs encoding artefacts of the raw files before any "
       "analysis: CSV quotes around ~24 % of the texts (with doubled inner quotes), literal JSON "
       "escapes in ~11 % (`\\u2019` for an apostrophe, `\\u002c` for a comma, which previously "
       "appeared as the frequent \"words\" `u2019` and `u002c`), HTML entities in ~4 % "
       "(`&amp;`), escaped quotes and slashes. In the test file, ~10 % of the texts are \"mojibake\" "
       "(UTF-8 read as Windows-1252: `ðŸ˜‚` instead of 😂), also repaired."),
    code("""from pathlib import Path
import sys

import pandas as pd
from IPython.display import Image, display

root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (p / "src" / "tweet_sentiment").is_dir())
sys.path.insert(0, str(root / "src"))

from tweet_sentiment import descriptive as d
from tweet_sentiment.config import FIGURES, ensure_output_dirs
from tweet_sentiment.data import load_test, load_train, load_tweets, TRAIN_FILE

ensure_output_dirs()
pd.set_option("display.max_colwidth", 120)
tweets, test = load_train(), load_test()
print(f"Loaded {len(tweets):,} training tweets and {len(test):,} test tweets.")
display(tweets.head())"""),
    code("""# Effect of the repair on a few raw lines
raw = load_tweets(TRAIN_FILE, repair=False)
changed = raw["text"] != tweets["text"]
print(f"{changed.mean():.1%} of the training texts were modified by the repair")
display(pd.DataFrame({"raw": raw.loc[changed, "text"].head(4).values,
                      "repaired": tweets.loc[changed, "text"].head(4).values}))"""),

    md("## 1. Data quality and structure\n\n"
       "Schema, missing values, repeated IDs and texts. Repeated tweets are a problem for "
       "modelling when they carry **different labels** (annotation conflicts) or when copies end up "
       "on both sides of a train/validation split. The descriptive statistics below keep every row; "
       "the models use `data.deduplicate()`, which drops the conflicting tweets and keeps one copy "
       "of the others."),
    code("""quality, dups = d.data_quality(tweets)
display(quality)
display(dups.rename("count").to_frame())
conflicts = tweets[tweets.groupby("text")["polarity"].transform("nunique") > 1]
display(conflicts.sort_values("text").head(6))"""),
    interp("quality"),

    md("## 2. Sentiment class balance (train vs test)"),
    code("""display(d.class_table(tweets, test).round(2))
d.plot_class_distribution(tweets, FIGURES / "task1_class_distribution.png", test)
display(Image(FIGURES / "task1_class_distribution.png"))"""),
    interp("classes"),

    md("## 3. Tweet length by sentiment\n\nLength is computed on the repaired raw text. "
       "A Kruskal-Wallis test checks whether the length distributions differ; the effect size "
       "ε² (share of the rank variance explained by the class) says whether the difference matters."),
    code("""display(d.length_stats(tweets).round(1))
display(d.length_tests(tweets))
d.plot_lengths(tweets, FIGURES / "task1_lengths.png")
display(Image(FIGURES / "task1_lengths.png"))"""),
    interp("lengths"),

    md("## 4. Frequent words and two-word phrases\n\n"
       "Counts on the normalised tokens (lowercase, URLs/mentions replaced, English stop words "
       "removed **except negations** such as *not*, *no*, *don't*, which carry sentiment). Values "
       "are occurrences per 1 000 tweets of the class, so classes of different sizes are comparable."),
    code("""for n, name in ((1, "unigrams"), (2, "bigrams")):
    tables = d.top_ngrams(tweets, n=n, k=12)
    d.plot_top_terms(tables, f"Most frequent {name}", FIGURES / f"task1_top_{name}.png")
    display(Image(FIGURES / f"task1_top_{name}.png"))
    display(pd.concat({c: t.round(1) for c, t in tables.items()}, axis=1))"""),
    interp("words"),

    md("## 5. Most polar words (improved Delta TF-IDF, course Ch3)\n\n"
       "Frequency alone mostly shows topics. The improved Delta TF-IDF of the course scores each "
       "word by log2(|N|(P_w + 0.5) / (|P|(N_w + 0.5))), where P_w and N_w are the numbers of "
       "positive and negative tweets containing it: > 0 for words typical of positive tweets, "
       "< 0 for negative ones. Words seen in fewer than 20 tweets are ignored."),
    code("""pos, neg = d.polar_terms(tweets, k=20)
d.plot_polar_terms(pos, neg, FIGURES / "task1_polar_terms.png")
display(Image(FIGURES / "task1_polar_terms.png"))
d.plot_wordclouds(tweets, FIGURES / "task1_wordclouds.png")
display(Image(FIGURES / "task1_wordclouds.png"))"""),
    interp("polar"),

    md("## 6. Hashtags and mentions\n\nExtracted from the raw text (the cleaner removes them)."),
    code("display(d.tags_table(tweets).round(1))"),
    interp("tags"),

    md("## 7. URLs, emoticons, emoji and punctuation\n\n"
       "Percentage of tweets of each class containing the marker at least once. The detectors "
       "are shared with Task 3 (`patterns.py`).\n\n"
       "**Correction.** The first version of this section reported 23-35 % of tweets with an "
       "emoticon. That pattern matched `:/` inside every `https://` and strings like `8p` inside "
       "t.co links, so the \"emoticon\" rate was essentially the URL rate. Emoticons are now "
       "searched after removing URLs and must stand as a separate symbol."),
    code("""rates = d.marker_rates(tweets)
display(rates.round(1))
d.plot_marker_rates(rates, FIGURES / "task1_marker_rates.png")
display(Image(FIGURES / "task1_marker_rates.png"))"""),
    interp("markers"),

    md("## 8. Summary and limitations"),
    md(I.get("summary", "_to be written after execution_")),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("descriptive_analysis.ipynb")
nbf.write(nb, out)
print("written", out)
