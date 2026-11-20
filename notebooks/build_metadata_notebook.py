"""Regenerates metadata_analysis.ipynb (Task 3).

Run from the repository root:  python notebooks/build_metadata_notebook.py
then execute the notebook (Jupyter or `jupyter nbconvert --execute --inplace`).
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Tweet metadata analysis (Task 3)\n\n"
       "The subject lets us choose among URLs, emoticons, punctuation, irony, censored insults, "
       "etc. We study **all five**, plus emphasis (capital letters, elongated words) and "
       "uncensored swear words for comparison. For each family:\n\n"
       "1. how the marker is detected, and whether the detector is reliable (manual checks);\n"
       "2. how often it occurs in each polarity class, with a statistical test;\n"
       "3. whether it helps the classifier of Task 2 (ablation study).\n\n"
       "Detectors work on the raw text (encoding repaired by the loader), because the model "
       "cleaner removes most of these markers. They are defined in `patterns.py` and "
       "`metadata.py`, shared with Task 1 and with the model tokenizer. "
       "`scripts/run_task3_metadata.py` reproduces the tables and figures."),
    code("""from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (p / "src" / "tweet_sentiment").is_dir())
sys.path.insert(0, str(root / "src"))

from tweet_sentiment.config import FIGURES, TABLES, ensure_output_dirs
from tweet_sentiment.data import load_model_data, load_test, load_train
from tweet_sentiment.evaluation import scores
from tweet_sentiment.metadata import (FEATURE_GROUPS, FEATURES, build_text_meta_model,
                                      compare_across_classes, emoji_polarity_table,
                                      extract_metadata, polarity_contradictions)
from tweet_sentiment.models import PREPROCESSING_VARIANTS

RECOMPUTE_ABLATION = False   # True: refit the 17 models of section 7 (~6 min)

ensure_output_dirs()
pd.set_option("display.max_colwidth", 140)
sns.set_theme(style="whitegrid")
train, test = extract_metadata(load_train()), extract_metadata(load_test())
print(f"{len(train):,} training tweets, {len(test):,} test tweets, {len(FEATURES)} features")
for group, cols in FEATURE_GROUPS.items():
    print(f"  {group:12s} {cols}")"""),

    md("## 1. Overview: association of each feature with the polarity\n\n"
       "For each feature: percentage of tweets of each class where it is present, chi-square test "
       "of independence between *present / absent* and the class, and **Cramér's V** as effect "
       "size (0.1 small, 0.3 medium, 0.5 large). With 50 000 tweets almost every p-value is tiny, "
       "so the effect size is what matters. Kruskal-Wallis tests on the counts (non-parametric, "
       "counts being very skewed) give the same conclusions."),
    code("""import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Kruskal on constant emoji counts
comp = compare_across_classes(train)
cols = ["pct_present_-1", "pct_present_0", "pct_present_1", "cramers_v", "chi2_p", "kruskal_p"]
display(comp[cols].sort_values("cramers_v", ascending=False)
        .style.format({c: "{:.2f}" for c in cols[:3]} | {"cramers_v": "{:.3f}",
                      "chi2_p": "{:.1e}", "kruskal_p": "{:.1e}"}))
import importlib.util
spec = importlib.util.spec_from_file_location("t3", root / "scripts" / "run_task3_metadata.py")
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
t3.plot_presence(comp, FIGURES / "task3_metadata_presence.png")
display(Image(FIGURES / "task3_metadata_presence.png"))"""),
    md("**Interpretation.** Only a few markers have a real association with the polarity: the "
       "**exclamation mark** (V = 0.25), **swear words** (0.22), **positive emoticons** (0.16), "
       "repeated `!!` (0.13), **URLs** (0.11) and negative emoticons (0.10). The others are "
       "either evenly spread (capital words, hashtags, mentions) or too rare to matter (irony "
       "markers, censored insults, V ≈ 0.01-0.02). Emoji give no value at all on the training "
       "file: it contains none (section 3). The following sections look at each family."),

    md("## 2. URLs"),
    code("""url = pd.DataFrame({
    "pct_with_url": (train["n_urls"] > 0).groupby(train["polarity"]).mean() * 100,
    "mean_urls": train.groupby("polarity")["n_urls"].mean(),
    "test_pct_with_url": (test["n_urls"] > 0).groupby(test["polarity"]).mean() * 100})
display(url.round(2))
display(train.loc[(train["n_urls"] > 0) & (train["polarity"] == 0), "text"].sample(5, random_state=1))"""),
    md("**Interpretation.** A third of the neutral tweets contain a link, against 20 % of the "
       "negative and 26 % of the positive ones. A tweet with a URL is often a headline or a shared "
       "article, written without personal opinion. All links are shortened (t.co), so the domain "
       "cannot be used. In the test file links are twice as frequent (53 % of tweets), and present "
       "in 40-60 % of every class: the association learned on the training file is weaker there."),

    md("## 3. Emoticons and emoji\n\n"
       "ASCII emoticons are searched **after removing URLs** (otherwise `:/` matches every "
       "`https://`) and must stand as a separate symbol: positive `:) :-) :D ;) :P <3 xD (:` and "
       "negative `:( :'( :/ :| D: </3 -_-`. Emoji are found with the `emoji` library; a short "
       "list of emoji with a conventional polarity (😍 ❤ 👍 vs 😢 😡 💔) gives `n_emoji_pos` and "
       "`n_emoji_neg`."),
    code("""emo = emoji_polarity_table(train, min_count=15)
print("Training file - most frequent emoticons (log2_pos_neg = Delta-IDF of the course, > 0 positive)")
display(emo.head(12).round(2))
print(f"Tweets with an emoji in the training file: {(train['n_emoji'] > 0).sum()}")
emo_test = emoji_polarity_table(test, min_count=15)
print("Test file - most frequent emoji / emoticons")
display(emo_test.head(14).round(2))"""),
    md("**Interpretation.**\n\n"
       "- ASCII emoticons have a **very clear polarity**: `:)` appears in 707 positive tweets "
       "against 30 negative ones (82 % positive), `:(` is 67 % negative, and their Delta-IDF is "
       "about ±3.6. But they are rare (7 % of the positive tweets, 3 % of the negative ones).\n"
       "- The **training file contains no emoji** (they were apparently removed when it was built), "
       "while 4-15 % of the test tweets have one. A model trained on this file cannot learn them.\n"
       "- In the test file, some emoji are as clear as emoticons (😍, ❤️, 😊 almost always "
       "positive), but the most frequent one, **😂, is spread over the three classes** "
       "(23 negative, 43 neutral, 41 positive tweets) and 😭 is split evenly: laughter is also used "
       "to mock, and \"crying\" often expresses being moved or amused. The meaning of an emoji "
       "depends on the context, which links this family to irony (section 5)."),

    md("## 4. Punctuation and emphasis"),
    code("""punct = ["n_exclam", "multi_exclam", "n_question", "multi_question", "interrobang",
         "n_ellipsis", "n_caps_words", "n_elongated"]
display(comp.loc[punct, ["pct_present_-1", "pct_present_0", "pct_present_1", "cramers_v"]].round(3))
fig, ax = plt.subplots(figsize=(7, 3.5))
d = (train.assign(exclam=train["n_exclam"].clip(upper=4))
     .groupby("exclam")["polarity"].value_counts(normalize=True).unstack() * 100)
d.plot.bar(stacked=True, color=["#C44E52", "#8C8C8C", "#4C72B0"], ax=ax, rot=0)
ax.set(xlabel="Number of '!' in the tweet (4 = 4 or more)", ylabel="Tweets (%)",
       title="Polarity by number of exclamation marks")
ax.legend(["negative", "neutral", "positive"], bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.show()"""),
    md("**Interpretation.** The **exclamation mark** is the strongest metadata of the dataset: "
       "present in 33 % of the positive tweets against 13 % of the others, and the share of "
       "positive tweets grows with the number of `!`. It mostly marks enthusiasm "
       "(*\"can't wait!!\"*), more rarely anger. **Question marks** go the other way: less "
       "frequent in positive tweets (9 % vs 14-15 %), typical of neutral questions and of "
       "indignant ones. Ellipses, capital words and elongated words (*soooo*) are spread almost "
       "evenly over the classes (V < 0.07): capital words in particular are often acronyms "
       "(NFL, USA) rather than shouting."),

    md("## 5. Irony\n\n"
       "Irony cannot be read from a single word. We use explicit markers only: the hashtags "
       "`#sarcasm`, `#irony`, `#not`, the `/s` tag, and fixed phrases (*yeah right*, *thanks a "
       "lot*, *said no one ever*, *I love how*…). Words like *great* or *sure* are not used on "
       "their own: they are far more often literal. We also count \"scare quotes\" (1-3 quoted "
       "words) and look at tweets whose emoticons contradict their label."),
    code("""print(f"Tweets with an explicit irony marker: {train['irony_marker'].sum()} "
      f"({train['irony_marker'].mean():.2%})")
display(train.loc[train["irony_marker"] == 1, ["polarity", "text"]].sample(12, random_state=0))
display(polarity_contradictions(train).round(2))"""),
    md("**Manual check.** Of the 12 tweets above, **5 are really ironic**: the `#Sarcasm` and "
       "`#not` tweets, *\"thanks a lot Frank Ocean\"*, *\"so much fun\"* about a politician's speeches "
       "and *\"I love how … have failed to have one yet\"*. Phrases such as "
       "*so much fun* or *love it when* are mostly sincere. Note also that the annotators did not "
       "label ironic tweets consistently: a `#sarcasm` tweet can be labelled negative while a "
       "`#not` one is labelled neutral.\n\n"
       "Positive emoticons in negative tweets (0.8 % of them) are not irony either in most "
       "cases: they soften a complaint or express mixed feelings (sample in "
       "`task3_validation_samples.csv`).\n\n"
       "**Interpretation.** Explicit irony is **too rare** in this corpus (0.04-0.13 % of the "
       "tweets) to be measured or used: V = 0.01. Irony exists, as the error analysis of Task 2 "
       "showed (*\"Happy … oppression day\"*), but it is implicit and would need a model of the "
       "contrast between words and context, not a list of markers."),

    md("## 6. Censored insults and swear words\n\n"
       "A censored insult is a token mixing letters and masking symbols (`f**k`, `sh*t`, `b!tch`, "
       "`a$$`) or a phrase like *f-word*. A symbol-only rule matches many false positives "
       "(`A$AP`, `#rap`, `is@the`, `great**`), so each candidate is accepted only if, once the "
       "symbols are treated as wildcards, it matches a known swear word. Uncensored swear words "
       "are counted with a small lexicon for comparison."),
    code("""cens = train[train["n_censored"] > 0]
print(f"Tweets with a censored insult: {len(cens)} ({len(cens) / len(train):.2%})")
display(cens["polarity"].value_counts().sort_index().rename("tweets").to_frame())
display(cens[["polarity", "text"]].head(8))
display(comp.loc[["n_censored", "n_swear"], ["pct_present_-1", "pct_present_0", "pct_present_1",
                                             "cramers_v"]].round(3))"""),
    md("**Interpretation.** Censored insults are **very rare** (48 tweets out of 50 000: 19 "
       "negative, 14 neutral, 15 positive). Relative to class size they are more frequent in "
       "negative tweets (0.24 % vs 0.06-0.08 %), but some are friendly or enthusiastic swearing "
       "(*\"Holy s**t\"*). Users of this "
       "corpus do not censor themselves much: **uncensored swear words** appear in 13 % of the "
       "negative tweets against about 2 % of the others (V = 0.22), the second strongest marker "
       "after the exclamation mark. The text models already see these words (*fuck*, *shit*, "
       "*stupid* are among the strongest negative ME weights in Task 2)."),

    md("## 7. Do the metadata help the classifier?\n\n"
       "Logistic regression on the TF-IDF n-grams of Task 2 (same preprocessing and C), with the "
       "metadata features added (log-scaled and standardised). We compare the text alone, text + "
       "all metadata, text + one family, text + all but one family, and the metadata alone. "
       "Scores on the validation part (used for decisions) and on the test file."),
    code("""path = TABLES / "task3_classifier_contribution.csv"
if RECOMPUTE_ABLATION or not path.exists():
    t3.classifier_contribution(
        PREPROCESSING_VARIANTS[json.loads((TABLES / "task2_best_params.json").read_text())["preprocessing"]],
        json.loads((TABLES / "task2_best_params.json").read_text())["me_C"]).to_csv(path)
contrib = pd.read_csv(path, index_col=0)
display(contrib[["val_macro_f1", "val_delta_vs_text", "test_macro_f1", "test_delta_vs_text"]]
        .round(4).sort_values("val_delta_vs_text", ascending=False))"""),
    code("""d = contrib.loc[[i for i in contrib.index if i.startswith("text + ") and "except" not in i]]
d = d[["val_delta_vs_text", "test_delta_vs_text"]].mul(100).sort_values("val_delta_vs_text")
ax = d.plot.barh(figsize=(8, 4), color=["#4C72B0", "#DD8452"])
ax.axvline(0, color="black", lw=0.8)
ax.set(xlabel="Change in macro-F1 vs text only (points)", ylabel="",
       title="Adding one family of metadata to the TF-IDF model")
ax.legend(ax.containers, ["validation", "test"])
plt.tight_layout()
plt.savefig(FIGURES / "task3_ablation.png", dpi=120)
plt.show()"""),
    md("**Interpretation.**\n\n"
       "- With all metadata the model gains **+1.0 point of macro-F1 on validation** and +0.3 on "
       "test.\n"
       "- **Punctuation brings almost all of the gain** (+1.2 on validation, +0.5 on test): the "
       "tokenizer drops `!` and `?`, so this information is new for the model.\n"
       "- Emoticons bring nothing: they are already in the text as `EMOPOS` / `EMONEG` tokens. "
       "URLs, hashtags and mentions are also already visible to the text model (as `URL`, words "
       "and `USER`).\n"
       "- Insults (+0.4) and irony markers (+0.2) help slightly on validation but not on test: "
       "these gains are within the noise of a 4 943-tweet validation set.\n"
       "- The metadata **alone** reach a macro-F1 of 0.45 (0.33 for a random guess): they carry "
       "real but limited information.\n"
       "- The smaller gain on test is consistent with the shift between the files (more URLs and "
       "hashtags, emoji that were never seen in training)."),

    md("## 8. Train / test differences"),
    code("""shift = pd.read_csv(TABLES / "task3_train_test_presence.csv", index_col=0)
display(shift.round(1).assign(ratio=lambda d: (d["test_pct_any_class"] / d["train_pct_any_class"]).round(2)))"""),
    md("**Interpretation.** URLs (28 % → 53 %), hashtags (19 % → 39 %) and emoji (0 % → 6.5 %) are "
       "much more frequent in the test file, posted one to five years later, while exclamation marks "
       "(21 % → 13 %) and ASCII emoticons (4 % → 0.6 %, replaced by emoji) are rarer. Metadata habits change over "
       "time, which limits how well these features transfer from one period to another."),

    md("## 9. Summary\n\n"
       "| Family | Main finding |\n|---|---|\n"
       "| URLs | more frequent in neutral tweets (33 % vs 20-26 %); links are shortened, no domain |\n"
       "| Emoticons / emoji | very clear polarity but rare; no emoji in the training file; 😂 and 😭 "
       "are ambiguous |\n"
       "| Punctuation | `!` is the strongest marker (33 % of positive tweets vs 13 %, V = 0.25); "
       "`?` less frequent in positive tweets |\n"
       "| Irony | explicit markers in 0.08 % of tweets, half of them literal; irony is mostly implicit |\n"
       "| Censored insults | 48 tweets, relatively more frequent in negative ones; uncensored swear words are far more "
       "frequent (13 % of negative tweets, V = 0.22) |\n"
       "| Classifier | +1.0 point on validation, +0.3 on test, almost all from punctuation |\n\n"
       "The detectors are rule-based heuristics checked by hand on samples; they show associations "
       "in this corpus, not causes."),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("metadata_analysis.ipynb")
nbf.write(nb, out)
print("written", out)
