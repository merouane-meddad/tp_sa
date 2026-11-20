"""Task 2 - classical models (course Ch3) and MLP (course Ch4).

1. compare the preprocessing variants (stop words / lemma / stem) with
   5-fold cross-validation;
2. tune the main hyper-parameters (grid search, 5-fold CV, macro-F1);
3. fit every model on the training split, evaluate on the validation split
   and, once, on the test file;
4. save models, tables (reports/tables/task2_*) and figures.

Usage:  python scripts/run_task2_models.py [--quick]
``--quick`` skips steps 1-2 and uses the tuned values stored in the code.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from tweet_sentiment.config import (CV_FOLDS, FIGURES, MODELS, RANDOM_STATE, TABLES,
                                    ensure_output_dirs)
from tweet_sentiment.data import load_model_data
from tweet_sentiment.evaluation import (error_sample, plot_confusions, plot_model_comparison,
                                        scores, top_features)
from tweet_sentiment.models import (GRID_TO_PARAM, PARAM_GRIDS, PREPROCESSING_VARIANTS,
                                    build_models, maxent, tfidf)
from sklearn.pipeline import Pipeline

CV = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def compare_preprocessing(train):
    rows = []
    for name, norm in PREPROCESSING_VARIANTS.items():
        pipe = Pipeline([("vec", tfidf(norm)), ("clf", maxent())])
        t = time.time()
        cv = cross_val_score(pipe, train["text"], train["polarity"], cv=CV,
                             scoring="f1_macro", n_jobs=2)
        vocab = len(tfidf(norm).fit(train["text"]).vocabulary_)
        rows.append({"preprocessing": name, "cv_macro_f1_mean": cv.mean(),
                     "cv_macro_f1_std": cv.std(), "vocabulary_size": vocab})
        print(f"  {name:28s} macro-F1 = {cv.mean():.4f} ± {cv.std():.4f}  "
              f"(vocabulary {vocab:,}, {time.time() - t:.0f}s)")
    return pd.DataFrame(rows).set_index("preprocessing")


def tune(train, normalizer):
    best_params, rows = {}, []
    models = build_models(normalizer)
    for name, grid in PARAM_GRIDS.items():
        t = time.time()
        gs = GridSearchCV(models[name], grid, scoring="f1_macro", cv=CV, n_jobs=2)
        gs.fit(train["text"], train["polarity"])
        print(f"  {name:36s} best {gs.best_params_}  CV macro-F1 = {gs.best_score_:.4f} "
              f"({time.time() - t:.0f}s)")
        for params, mean, std in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"],
                                     gs.cv_results_["std_test_score"]):
            rows.append({"model": name, "params": json.dumps(params), "cv_macro_f1": mean,
                         "cv_std": std})
        for grid_key, value in gs.best_params_.items():
            best_params[GRID_TO_PARAM[(name, grid_key)]] = value
    # the vote and the word2vec SVM reuse the tuned SVM / ME / NB settings
    best_params["w2v_C"] = best_params.get("svm_C", 0.1)
    return best_params, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip CV comparison and tuning")
    args = parser.parse_args()
    ensure_output_dirs()

    train, val, test = load_model_data()
    print(f"train {len(train):,} | validation {len(val):,} | test {len(test):,} tweets")

    normalizer = PREPROCESSING_VARIANTS["clean only"]
    params = {}
    if not args.quick:
        print("\n1. Preprocessing comparison (ME / logistic regression, TF-IDF 1-2 grams)")
        prep = compare_preprocessing(train)
        prep.to_csv(TABLES / "task2_preprocessing_comparison.csv")
        best_prep = prep["cv_macro_f1_mean"].idxmax()
        normalizer = PREPROCESSING_VARIANTS[best_prep]
        print(f"  -> kept: {best_prep}")

        print("\n2. Hyper-parameter search")
        params, grid = tune(train, normalizer)
        grid.to_csv(TABLES / "task2_grid_search.csv", index=False)
        (TABLES / "task2_best_params.json").write_text(
            json.dumps({"preprocessing": best_prep, **params}, indent=2, default=str))
    elif (TABLES / "task2_best_params.json").exists():
        saved = json.loads((TABLES / "task2_best_params.json").read_text())
        normalizer = PREPROCESSING_VARIANTS[saved.pop("preprocessing")]
        params = saved

    print("\n3. Fit on train, evaluate on validation and test")
    rows, test_preds = [], {}
    for name, model in build_models(normalizer, params).items():
        t = time.time()
        if name.startswith("MLP"):
            w = train["polarity"].map(len(train) / (3 * train["polarity"].value_counts()))
            model.fit(train["text"], train["polarity"], clf__sample_weight=w.to_numpy())
        else:
            model.fit(train["text"], train["polarity"])
        fit_s = time.time() - t
        val_s, test_pred = scores(val["polarity"], model.predict(val["text"])), model.predict(test["text"])
        test_s = scores(test["polarity"], test_pred)
        test_preds[name] = test_pred
        rows.append({"model": name, "fit_seconds": fit_s,
                     **{f"val_{k}": v for k, v in val_s.items()},
                     **{f"test_{k}": v for k, v in test_s.items()}})
        print(f"  {name:36s} val macro-F1 {val_s['macro_f1']:.4f} | test macro-F1 "
              f"{test_s['macro_f1']:.4f}, macro-recall {test_s['macro_recall']:.4f} ({fit_s:.0f}s)")
        safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
        joblib.dump(model, MODELS / f"task2_{safe}.joblib")
        if name.startswith("ME"):
            top_features(model, k=20).to_csv(TABLES / "task2_maxent_top_features.csv", index=False)

    results = pd.DataFrame(rows).set_index("model")
    results.to_csv(TABLES / "task2_classical_results.csv")
    np.save(MODELS / "task2_classical_test_predictions.npy", test_preds, allow_pickle=True)

    test_cols = results.filter(like="test_").rename(columns=lambda c: c[5:])
    plot_model_comparison(test_cols, FIGURES / "task2_classical_comparison.png")
    plot_confusions(test_preds, test["polarity"], FIGURES / "task2_classical_confusions.png")
    best = results["val_macro_f1"].idxmax()
    print(f"\nBest model on validation: {best}")
    error_sample(test, test_preds[best]).to_csv(TABLES / "task2_error_sample.csv", index=False)
    print(test_cols[["macro_f1", "macro_recall", "f1_pn", "accuracy"]].round(4))


if __name__ == "__main__":
    main()
