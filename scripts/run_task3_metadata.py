"""Task 3 - metadata analysis: URLs, emoticons / emoji, punctuation, irony
cues, censored insults.

1. extract the features (training file, and test file for comparison);
2. compare them across polarity classes (chi-square + Cramér's V,
   Kruskal-Wallis);
3. polarity of emoticons / emoji, label / symbol contradictions (irony);
4. samples of detected irony markers and censored insults for manual checks;
5. contribution to the classifier: TF-IDF model with and without the
   metadata, ablation by feature group, metadata-only model.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from tweet_sentiment.config import FIGURES, TABLES, ensure_output_dirs  # noqa: E402
from tweet_sentiment.data import load_model_data, load_test, load_train  # noqa: E402
from tweet_sentiment.evaluation import scores  # noqa: E402
from tweet_sentiment.metadata import (FEATURE_GROUPS, FEATURES, build_text_meta_model,  # noqa: E402
                                      compare_across_classes, emoji_polarity_table,
                                      extract_metadata, polarity_contradictions)
from tweet_sentiment.models import PREPROCESSING_VARIANTS  # noqa: E402


def plot_presence(comp, out_path):
    pct = comp[[f"pct_present_{c}" for c in (-1, 0, 1)]]
    pct.columns = ["negative", "neutral", "positive"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [3, 1.3]})
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set(title="Tweets containing the feature (%)", xlabel="", ylabel="")
    v = comp["cramers_v"].fillna(0)
    # heatmap rows are centred on 0.5, 1.5, ... from the top: align the bars
    axes[1].barh([i + 0.5 for i in range(len(v))], v.values, height=0.7, color="#4C72B0")
    axes[1].set_ylim(len(v), 0)
    axes[1].set(title="Association with polarity\n(Cramér's V)", yticks=[])
    for x in (0.1, 0.3):
        axes[1].axvline(x, ls="--", color="grey", lw=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def classifier_contribution(normalizer, C):
    train, val, test = load_model_data()
    train, val, test = (extract_metadata(d) for d in (train, val, test))
    configs = {"text only (TF-IDF)": dict(features=[]),
               "text + all metadata": dict(features=FEATURES),
               "metadata only": dict(features=FEATURES, use_text=False)}
    for group, cols in FEATURE_GROUPS.items():
        configs[f"text + {group}"] = dict(features=cols)
        configs[f"text + all except {group}"] = dict(features=[f for f in FEATURES if f not in cols])
    rows = []
    for name, cfg in configs.items():
        model = build_text_meta_model(normalizer, C, **cfg)
        model.fit(train, train["polarity"])
        v = scores(val["polarity"], model.predict(val))
        t = scores(test["polarity"], model.predict(test))
        rows.append({"model": name, "val_macro_f1": v["macro_f1"], "val_f1_pn": v["f1_pn"],
                     "test_macro_f1": t["macro_f1"], "test_macro_recall": t["macro_recall"],
                     "test_f1_pn": t["f1_pn"]})
        print(f"  {name:36s} val macro-F1 {v['macro_f1']:.4f} | test macro-F1 {t['macro_f1']:.4f}")
    out = pd.DataFrame(rows).set_index("model")
    base = out.loc["text only (TF-IDF)"]
    out["val_delta_vs_text"] = out["val_macro_f1"] - base["val_macro_f1"]
    out["test_delta_vs_text"] = out["test_macro_f1"] - base["test_macro_f1"]
    return out


def main():
    ensure_output_dirs()
    train, test = extract_metadata(load_train()), extract_metadata(load_test())

    print("1-2. Features compared across classes (training file)")
    comp = compare_across_classes(train)
    comp.to_csv(TABLES / "task3_metadata_by_polarity.csv")
    print(comp[["pct_present_-1", "pct_present_0", "pct_present_1", "cramers_v"]].round(3))
    plot_presence(comp, FIGURES / "task3_metadata_presence.png")
    test_comp = compare_across_classes(test)
    test_comp.to_csv(TABLES / "task3_metadata_by_polarity_test.csv")
    shift = pd.DataFrame({"train_pct_any_class": (train[FEATURES] > 0).mean() * 100,
                          "test_pct_any_class": (test[FEATURES] > 0).mean() * 100})
    shift.to_csv(TABLES / "task3_train_test_presence.csv")

    print("\n3. Emoticon / emoji polarity and contradictions")
    emo_train = emoji_polarity_table(train, min_count=15)
    emo_train.to_csv(TABLES / "task3_emoticon_polarity_train.csv", index=False)
    emo_test = emoji_polarity_table(test, min_count=15)
    emo_test.to_csv(TABLES / "task3_emoji_polarity_test.csv", index=False)
    print(emo_train.head(12).round(2).to_string(index=False))
    contra = polarity_contradictions(train)
    contra.to_csv(TABLES / "task3_symbol_label_contradictions.csv")
    print(contra.round(2))

    print("\n4. Samples for manual validation")
    samples = pd.concat([
        train[train["irony_marker"] == 1].sample(30, random_state=0).assign(detector="irony_marker"),
        train[train["n_censored"] > 0].assign(detector="censored_insult"),
        train[(train["polarity"] == -1) & (train["n_emo_pos"] > 0)]
        .sample(20, random_state=0).assign(detector="negative_with_positive_emoticon"),
    ])[["detector", "polarity", "text"]]
    samples.to_csv(TABLES / "task3_validation_samples.csv", index=False)
    print(samples["detector"].value_counts())

    print("\n5. Contribution to the classifier (logistic regression)")
    params_file = TABLES / "task2_best_params.json"
    saved = json.loads(params_file.read_text()) if params_file.exists() else {}
    normalizer = PREPROCESSING_VARIANTS[saved.get("preprocessing", "lemma (stop words kept)")]
    contrib = classifier_contribution(normalizer, saved.get("me_C", 2.0))
    contrib.to_csv(TABLES / "task3_classifier_contribution.csv")


if __name__ == "__main__":
    main()
