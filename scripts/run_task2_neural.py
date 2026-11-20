"""Task 2 - recurrent networks (course Ch4): simple RNN, LSTM, CNN+LSTM.

Trains each architecture on the training split with early stopping on the
validation split, evaluates once on the test file, then merges the results
with those of run_task2_models.py into task2_all_models.csv.

Usage:  python scripts/run_task2_neural.py [--archs rnn lstm cnn_lstm] [--epochs 10]
"""
import argparse

import numpy as np
import pandas as pd

from tweet_sentiment.advanced_models import RNNSentimentClassifier
from tweet_sentiment.config import FIGURES, MODELS, TABLES, ensure_output_dirs
from tweet_sentiment.data import load_model_data
from tweet_sentiment.evaluation import plot_confusions, plot_model_comparison, scores

NAMES = {"rnn": "simple RNN (word2vec init)", "lstm": "LSTM (word2vec init)",
         "cnn_lstm": "CNN + LSTM (word2vec init)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=["rnn", "lstm", "cnn_lstm"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    ensure_output_dirs()
    train, val, test = load_model_data()

    rows, preds, histories = [], {}, []
    for arch in args.archs:
        print(f"\n== {NAMES[arch]}")
        clf = RNNSentimentClassifier(arch=arch, epochs=args.epochs)
        clf.fit(train["text"], train["polarity"], val["text"], val["polarity"])
        print(f"   vocabulary {len(clf.vocab_.itos):,} words, "
              f"{clf.embedding_coverage_:.1%} initialised from word2vec")
        val_s = scores(val["polarity"], clf.predict(val["text"]))
        preds[NAMES[arch]] = clf.predict(test["text"])
        test_s = scores(test["polarity"], preds[NAMES[arch]])
        rows.append({"model": NAMES[arch],
                     "fit_seconds": sum(h["seconds"] for h in clf.history_),
                     "epochs_run": len(clf.history_),
                     **{f"val_{k}": v for k, v in val_s.items()},
                     **{f"test_{k}": v for k, v in test_s.items()}})
        histories += [{"model": NAMES[arch], **h} for h in clf.history_]
        print(f"   val macro-F1 {val_s['macro_f1']:.4f} | test macro-F1 {test_s['macro_f1']:.4f}, "
              f"macro-recall {test_s['macro_recall']:.4f}")
        clf.save(MODELS / f"task2_{arch}.pt")

    neural = pd.DataFrame(rows).set_index("model")
    neural.to_csv(TABLES / "task2_neural_results.csv")
    pd.DataFrame(histories).to_csv(TABLES / "task2_neural_training_history.csv", index=False)
    plot_confusions(preds, test["polarity"], FIGURES / "task2_neural_confusions.png")

    # merge with the classical models
    classical_path = TABLES / "task2_classical_results.csv"
    if classical_path.exists():
        allm = pd.concat([pd.read_csv(classical_path, index_col="model"), neural])
        allm = allm.sort_values("val_macro_f1", ascending=False)
        allm.to_csv(TABLES / "task2_all_models.csv")
        plot_model_comparison(allm.filter(like="test_").rename(columns=lambda c: c[5:]),
                              FIGURES / "task2_all_models_comparison.png")
        cols = ["val_macro_f1", "test_macro_f1", "test_macro_recall", "test_f1_pn", "test_accuracy"]
        print("\n", allm[cols].round(4))
        classical_preds = MODELS / "task2_classical_test_predictions.npy"
        if classical_preds.exists():
            preds = {**np.load(classical_preds, allow_pickle=True).item(), **preds}
            np.save(MODELS / "task2_all_test_predictions.npy", preds, allow_pickle=True)


if __name__ == "__main__":
    main()
