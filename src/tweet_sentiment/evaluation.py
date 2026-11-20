"""Shared evaluation: metrics, confusion matrices, error analysis.

Metrics reported for every model:

* ``macro_f1``      - mean of the per-class F1 (main metric of the project);
* ``macro_recall``  - mean of the per-class recall, the official SemEval-2017
  Task 4A metric, robust to the different class balance of the test set;
* ``f1_pn``         - mean F1 of the positive and negative classes only
  (secondary SemEval metric);
* ``accuracy``      - shown for reference only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,  # noqa: E402
                             precision_recall_fscore_support, recall_score)

from .config import LABEL_NAMES, LABELS  # noqa: E402


def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=LABELS),
        "f1_pn": (f[0] + f[2]) / 2,
        "accuracy": accuracy_score(y_true, y_pred),
    }
    for i, c in enumerate(LABELS):
        name = LABEL_NAMES[c][:3]
        out.update({f"precision_{name}": p[i], f"recall_{name}": r[i], f"f1_{name}": f[i]})
    return out


def plot_confusion(y_true, y_pred, title, out_path, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABELS, normalize="true", values_format=".2f",
        display_labels=[LABEL_NAMES[c] for c in LABELS], cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    if own:
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()


def plot_confusions(results, y_true, out_path, ncols=3):
    """Grid of row-normalised confusion matrices, ``results = {name: y_pred}``."""
    n = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, y_pred) in zip(axes, results.items()):
        plot_confusion(y_true, y_pred, name, None, ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_model_comparison(table, out_path, metrics=("macro_f1", "macro_recall", "f1_pn")):
    d = table.reset_index(names="model").melt(id_vars="model", value_vars=list(metrics),
                                              var_name="metric", value_name="score")
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(table) + 2))
    sns.barplot(data=d, y="model", x="score", hue="metric", ax=ax)
    ax.set(xlim=(0, 1), xlabel="Test score", ylabel="", title="Model comparison on the test set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def error_sample(df, y_pred, n_per_cell=5, random_state=0):
    """A few misclassified tweets for each (true, predicted) pair."""
    d = df.assign(predicted=y_pred)
    wrong = d[d["polarity"] != d["predicted"]]
    return (wrong.sample(frac=1, random_state=random_state)
            .groupby(["polarity", "predicted"]).head(n_per_cell)
            .sort_values(["polarity", "predicted"])[["polarity", "predicted", "text"]])


def top_features(pipeline, k=15, vec_step="vec", clf_step="clf"):
    """Most weighted n-grams per class for a linear model on TF-IDF features."""
    vec, clf = pipeline.named_steps[vec_step], pipeline.named_steps[clf_step]
    terms = vec.get_feature_names_out()
    rows = {}
    for i, c in enumerate(clf.classes_):
        order = np.argsort(clf.coef_[i])[::-1][:k]
        rows[f"{c} ({LABEL_NAMES[c]})"] = [f"{terms[j]} ({clf.coef_[i][j]:.2f})" for j in order]
    return pd.DataFrame(rows)
