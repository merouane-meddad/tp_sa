"""Regenerates models_analysis.ipynb (Task 2).

Run from the repository root:  python notebooks/build_models_notebook.py
then execute the notebook (Jupyter or `jupyter nbconvert --execute --inplace`).

The notebook reuses what scripts/run_task2_models.py and run_task2_neural.py
saved (cross-validation tables, tuned parameters, fitted models), so it runs
in a few minutes. Set RETRAIN = True in the first cell to refit everything.
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Sentiment classification with machine learning models (Task 2)\n\n"
       "This notebook compares the models of the course on the same data and the same protocol:\n\n"
       "- **Ch3, *Analyse des sentiments***: preprocessing (stop words, lemmatisation, stemming), "
       "text representations (TF-IDF, improved Delta TF-IDF, word2vec weighted by Delta TF-IDF), "
       "classifiers Naive Bayes (NB), linear SVM, Maximum Entropy (ME = logistic regression) and "
       "their majority vote;\n"
       "- **Ch4, *Perceptron multicouches* and RNN**: a multilayer perceptron, a simple RNN, an "
       "LSTM and the CNN + LSTM hybrid shown in Ch3.\n\n"
       "**Protocol.** The training file is deduplicated (conflicting labels removed) and split "
       "into 90 % train / 10 % validation, stratified, seed 42. Preprocessing and hyper-parameters "
       "are chosen by 5-fold cross-validation on the train part; the RNNs use early stopping on the "
       "validation part. The test file is used only for the final comparison.\n\n"
       "**Metrics.** The test set has a different class balance from the training set "
       "(32 % negative vs 16 %), so accuracy is misleading. We report the **macro-F1** (main "
       "metric), the **macro-recall** (official metric of SemEval-2017 Task 4A, which used this "
       "test set) and **F1-PN**, the mean F1 of the positive and negative classes.\n\n"
       "The heavy steps (cross-validation, grid search, RNN training) are run by "
       "`scripts/run_task2_models.py` and `scripts/run_task2_neural.py`. By default this notebook "
       "reads their saved tables and models; set `RETRAIN = True` to refit everything here "
       "(about 30 minutes on a laptop CPU)."),
    code("""from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (p / "src" / "tweet_sentiment").is_dir())
sys.path.insert(0, str(root / "src"))

from tweet_sentiment.config import FIGURES, MODELS, TABLES, ensure_output_dirs
from tweet_sentiment.data import load_model_data
from tweet_sentiment.evaluation import plot_confusions, plot_model_comparison, scores, top_features
from tweet_sentiment.features import DeltaTfidfVectorizer, delta_idf
from tweet_sentiment.models import PREPROCESSING_VARIANTS, build_models

RETRAIN = False   # True: rerun the CV / grid search and refit every model here

ensure_output_dirs()
pd.set_option("display.max_colwidth", 140)
sns.set_theme(style="whitegrid")
train, val, test = load_model_data()
print(f"train {len(train):,} | validation {len(val):,} | test {len(test):,} tweets")
display(pd.DataFrame({s: d["polarity"].value_counts(normalize=True).sort_index().mul(100).round(1)
                      for s, d in (("train", train), ("validation", val), ("test", test))}))"""),

    md("## 1. Preprocessing (course Ch3)\n\n"
       "The course lists three cleaning steps: stop word removal, lemmatisation and stemming. "
       "They are implemented in `TextNormalizer`, on top of a light cleaning (lowercase, URLs → "
       "`URL`, mentions → `USER`, `#` removed, emoticons → `EMOPOS` / `EMONEG`, \"soooo\" → "
       "\"soo\"). Negations (*not*, *no*, *don't*…) are never removed as stop words, since "
       "Task 1 showed that they are 2-3 times more frequent in negative tweets."),
    code("""example = train.loc[train["text"].str.len().between(90, 120)].iloc[3, 2]
print("raw:", example, "\\n")
for name, norm in PREPROCESSING_VARIANTS.items():
    print(f"{name:26s}", norm(example))"""),
    md("Each variant is evaluated with the same model (ME / logistic regression on TF-IDF uni- and "
       "bigrams) by 5-fold cross-validation on the training part."),
    code("""if RETRAIN:
    import subprocess
    subprocess.run([sys.executable, str(root / "scripts" / "run_task2_models.py")], check=True, cwd=root)
prep = pd.read_csv(TABLES / "task2_preprocessing_comparison.csv", index_col=0)
display(prep.round(4))
ax = prep["cv_macro_f1_mean"].plot.barh(xerr=prep["cv_macro_f1_std"], color="#4C72B0", figsize=(7, 3))
ax.set(xlim=(0.62, 0.65), xlabel="CV macro-F1 (mean ± std over 5 folds)", ylabel="")
plt.show()"""),
    md("**Interpretation.** Removing stop words shrinks the vocabulary by a third (96 615 → 66 889 "
       "terms) but slightly *lowers* the score (0.633 vs 0.638). In tweets of about 20 words, many "
       "\"empty\" words carry sentiment or contrast (*but*, *so*, *too*, *very*, *against*), and "
       "TF-IDF already down-weights the truly uninformative ones, as the course notes. "
       "Lemmatisation helps a little (0.643 when stop words are kept), stemming about as much. "
       "The differences are of the order of the standard deviation between folds (0.003-0.006), "
       "so the choice matters less than the model. We keep **lemmatisation without stop word "
       "removal** for all the models below."),

    md("## 2. Improved Delta TF-IDF (course Ch3)\n\n"
       "Each term receives the weight\n\n"
       "$$V(w,t) = C(w,t) \\times \\log_2\\left(\\frac{|N|\\,(P_w + 0.5)}{|P|\\,(N_w + 0.5)}\\right)$$\n\n"
       "where $C(w,t)$ is the count of $w$ in tweet $t$, $|P|, |N|$ the numbers of positive and negative "
       "tweets and $P_w, N_w$ the numbers of them containing $w$. The smoothing 0.5 avoids infinite "
       "weights for words absent from one class. The cell below checks the implementation against "
       "the table of the course slide (t1, t2, t3)."),
    code("""import scipy.sparse as sp
n_pos, n_neg = 1000, 590                       # |N| / |P| = 0.59 as on the slide
y = np.array([1] * n_pos + [-1] * n_neg)
cols = []
for p_w, n_w in ((100, 0), (2, 0), (100, 1)):
    c = np.zeros(len(y)); c[:p_w] = 1; c[n_pos:n_pos + n_w] = 1; cols.append(c)
print("course table : 6.90, 1.57, 5.32")
print("our weights  :", np.round(delta_idf(sp.csr_matrix(np.column_stack(cols)), y), 2))"""),
    md("With three classes the formula can be applied to positive vs negative only "
       "(`pos_neg`, neutral tweets ignored when estimating the weights) or once per class against "
       "the two others (`ovr`, three blocks of features). The grid search below chooses between "
       "the two."),

    md("## 3. Hyper-parameter search\n\n5-fold cross-validation on the training part, macro-F1."),
    code("""grid = pd.read_csv(TABLES / "task2_grid_search.csv")
best = grid.loc[grid.groupby("model")["cv_macro_f1"].idxmax()]
display(best.round(4))
params = json.loads((TABLES / "task2_best_params.json").read_text())
print(params)"""),
    md("**Interpretation.** The one-vs-rest Delta TF-IDF (`ovr`) beats the positive-vs-negative "
       "version: the neutral class, which is the largest, gets its own weights. The SVM prefers "
       "strong regularisation (small C), which is usual with ~90 000 sparse n-gram features for "
       "44 000 tweets."),

    md("## 4. Classical models and multilayer perceptron\n\n"
       "Models fitted on the training part with the tuned parameters (loaded from `models/` "
       "unless `RETRAIN = True`), evaluated on the validation part and on the test file."),
    code("""names = list(build_models().keys())
models = {}
for name in names:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
    path = MODELS / f"task2_{safe}.joblib"
    if path.exists() and not RETRAIN:
        models[name] = joblib.load(path)
    else:
        raise FileNotFoundError(f"{path} missing: run scripts/run_task2_models.py first")

rows, test_preds = [], {}
for name, model in models.items():
    v = scores(val["polarity"], model.predict(val["text"]))
    test_preds[name] = model.predict(test["text"])
    t = scores(test["polarity"], test_preds[name])
    rows.append({"model": name, "val_macro_f1": v["macro_f1"], "test_macro_f1": t["macro_f1"],
                 "test_macro_recall": t["macro_recall"], "test_f1_pn": t["f1_pn"],
                 "test_accuracy": t["accuracy"], "test_recall_neg": t["recall_neg"],
                 "test_recall_neu": t["recall_neu"], "test_recall_pos": t["recall_pos"]})
classical = pd.DataFrame(rows).set_index("model").sort_values("val_macro_f1", ascending=False)
display(classical.round(3))"""),
    code("""plot_confusions(test_preds, test["polarity"], FIGURES / "task2_classical_confusions.png")
display(Image(FIGURES / "task2_classical_confusions.png"))"""),
    md("**Interpretation.**\n\n"
       "- The **linear SVM on Delta TF-IDF** is the best classical model (test macro-F1 0.617, "
       "+1.1 points over the same SVM on plain TF-IDF): weighting each n-gram by how unevenly it is "
       "spread over the classes, as the course suggests, helps the classifier.\n"
       "- SVM, ME and their **majority vote** are within half a point of each other. The vote does "
       "not beat its best member: NB, SVM and ME see the same TF-IDF features and make largely the "
       "same mistakes, so voting has little to correct.\n"
       "- **Naive Bayes** is the weakest TF-IDF model (0.585), as expected from its independence "
       "assumption, but it is by far the fastest to train.\n"
       "- The **word2vec tweet vector** weighted by Delta TF-IDF (formula of the course) loses the "
       "n-gram information: averaging 20 word vectors blurs the sentence, and word2vec trained on "
       "44 000 tweets only gives weak vectors. Its neutral recall drops to 0.44.\n"
       "- The **multilayer perceptron** (one hidden layer of 256 ReLU units, 30 000 n-grams selected "
       "by χ²) overfits within a few epochs (early stopping after 7) and stays below the linear "
       "models: with sparse n-grams, a linear separator is already a strong model.\n"
       "- The confusion matrices show the same pattern for every model: the main error is "
       "**neutral ↔ polarised** (21-37 % of negative and positive tweets predicted neutral), while "
       "negative ↔ positive confusions are rarer (5-14 %)."),
    code("""me = models["ME / logistic regression (TF-IDF)"]
display(top_features(me, k=15))"""),
    md("**Interpretation.** The largest ME weights confirm Task 1. Negative and positive tweets "
       "are recognised through evaluative words, swear words and emoticons (`EMONEG`, *fuck*, "
       "*suck*, *worst*, *not*, *don't* vs `EMOPOS`, *happy*, *love*, *great*, *can't wait*); "
       "lemmatisation merges *excited* / *exciting* into *excite*. The neutral class is defined "
       "differently: by **topics** (brands and celebrities such as *gucci*, *shawn*, *murray*, "
       "*naruto*, *ps4*), by question forms (*do you*, *or*) and by links (`URL`). A model that "
       "relies on topic names transfers poorly to a test set about other topics."),

    md("## 5. Recurrent networks (course Ch4)\n\n"
       "Sentiment analysis is the **many-to-one** case of the course: a sequence of words goes in, "
       "one class comes out. All three networks share the same input pipeline: tokens → indices "
       "(sequences cut or padded to 50 tokens) → an **Embedding** layer of dimension 100 initialised "
       "with a word2vec model trained on the training tweets (fine-tuned during training) → "
       "recurrent part → **Dense** softmax layer over the 3 classes.\n\n"
       "| Model | Recurrent part |\n|---|---|\n"
       "| simple RNN | `RNN(128, tanh)` |\n"
       "| LSTM | `LSTM(128)`, gates that keep information over long sequences |\n"
       "| CNN + LSTM | `Conv1D(128, kernel 3) + MaxPool(2)` then `LSTM(128)` (Othman et al., 2021) |\n\n"
       "Training: Adam (lr 2e-3), batch 128, class-weighted cross-entropy, dropout 0.4, gradient "
       "clipping at 1 (against exploding gradients), early stopping on validation macro-F1 "
       "(patience 2). The networks are written with PyTorch, the layers being those of the Keras "
       "examples of the course."),
    code("""from tweet_sentiment.advanced_models import RNNSentimentClassifier
NAMES = {"rnn": "simple RNN (word2vec init)", "lstm": "LSTM (word2vec init)",
         "cnn_lstm": "CNN + LSTM (word2vec init)"}
if RETRAIN:
    import subprocess
    subprocess.run([sys.executable, str(root / "scripts" / "run_task2_neural.py")], check=True, cwd=root)

rnn_rows, rnn_preds = [], {}
for arch, name in NAMES.items():
    clf = RNNSentimentClassifier.load(MODELS / f"task2_{arch}.pt")
    v = scores(val["polarity"], clf.predict(val["text"]))
    rnn_preds[name] = clf.predict(test["text"])
    t = scores(test["polarity"], rnn_preds[name])
    rnn_rows.append({"model": name, "val_macro_f1": v["macro_f1"], "test_macro_f1": t["macro_f1"],
                     "test_macro_recall": t["macro_recall"], "test_f1_pn": t["f1_pn"],
                     "test_accuracy": t["accuracy"], "test_recall_neg": t["recall_neg"],
                     "test_recall_neu": t["recall_neu"], "test_recall_pos": t["recall_pos"]})
neural = pd.DataFrame(rnn_rows).set_index("model")
display(neural.round(3))"""),
    code("""hist = pd.read_csv(TABLES / "task2_neural_training_history.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in hist.groupby("model", sort=False):
    axes[0].plot(h["epoch"], h["train_loss"], marker="o", label=name)
    axes[1].plot(h["epoch"], h["val_macro_f1"], marker="o", label=name)
axes[0].set(title="Training loss (weighted cross-entropy)", xlabel="epoch")
axes[1].set(title="Validation macro-F1", xlabel="epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIGURES / "task2_neural_training_curves.png", dpi=120)
plt.show()
plot_confusions(rnn_preds, test["polarity"], FIGURES / "task2_neural_confusions.png")
display(Image(FIGURES / "task2_neural_confusions.png"))"""),
    md("**Interpretation.**\n\n"
       "- The **simple RNN** is clearly the weakest network (test macro-F1 0.572) and its "
       "validation score jumps from one epoch to the next. This is the limit described in Ch4: the "
       "gradient flowing back through many tanh steps vanishes, so the network struggles to relate "
       "the end of a tweet to its beginning.\n"
       "- The **LSTM** gains almost 4 points (0.609): its gates let information and gradients go "
       "through the whole sequence. The **CNN + LSTM** adds a small gain (0.612) and has the best "
       "macro-recall of all models (0.622): the convolution detects local patterns (*not good*, "
       "*can't wait*) before the LSTM reads them in order.\n"
       "- The networks have the best **recall on negative tweets** (0.71-0.73 vs 0.57-0.64 for "
       "the linear models), which is what lifts their macro-recall and F1-PN on a test set where "
       "negatives are twice as frequent as in training.\n"
       "- They **overfit quickly**: the training loss keeps falling while the validation score "
       "peaks after 1-2 epochs, hence early stopping. With 44 000 training tweets and embeddings "
       "learned from them only, the networks do not clearly beat a well-tuned linear model on "
       "macro-F1."),

    md("## 6. Comparison of all models"),
    code("""allm = pd.concat([classical, neural]).sort_values("val_macro_f1", ascending=False)
display(allm[["val_macro_f1", "test_macro_f1", "test_macro_recall", "test_f1_pn", "test_accuracy"]].round(3))
plot_model_comparison(allm.filter(like="test_").rename(columns=lambda c: c[5:]), FIGURES / "task2_all_models_comparison.png")
display(Image(FIGURES / "task2_all_models_comparison.png"))
gap = (allm["val_macro_f1"] - allm["test_macro_f1"])
print(f"validation - test gap: mean {gap.mean():.3f}, min {gap.min():.3f}, max {gap.max():.3f}")"""),
    md("**Interpretation.** Every model loses about 4 points of macro-F1 between validation and "
       "test, whatever its type. This gap is not the overfitting of one model but the **shift "
       "between the two files** found in Task 1: the test tweets were posted in late 2016, about "
       "other topics, with twice as many negative tweets. The ranking is stable: Delta TF-IDF + SVM "
       "for macro-F1, CNN + LSTM for macro-recall. For reference, the best systems of SemEval-2017, "
       "with embeddings pre-trained on millions of tweets, reached a macro-recall of about 0.68 "
       "on this test set."),

    md("## 7. Error analysis\n\nMisclassified test tweets of the best model on validation "
       "(linear SVM on Delta TF-IDF), a few per (true, predicted) pair."),
    code("""best_name = classical["val_macro_f1"].idxmax()
d = test.assign(predicted=test_preds[best_name])
wrong = d[d["polarity"] != d["predicted"]]
print(f"{best_name}: {len(wrong):,} errors out of {len(d):,} test tweets")
display(wrong.sample(frac=1, random_state=0).groupby(["polarity", "predicted"]).head(3)
        .sort_values(["polarity", "predicted"])[["polarity", "predicted", "text"]])"""),
    md("**Interpretation.** About 37 % of the test tweets are misclassified. Typical errors "
       "seen in the sample:\n\n"
       "- **irony and sarcasm**: *\"Happy @elizabethforma oppression day\"* (labelled neutral, "
       "predicted positive because of *happy*), or *\"in good news tonight, Fidel Castro assumed "
       "room temperature\"*, where *good news* announces a death;\n"
       "- **hostility without opinion words**: political attacks written with insults that are not "
       "in the training vocabulary (*creature*, *inbred neo-nazi*) are predicted neutral;\n"
       "- **neutral information about a negative topic** (*#Syria: #Hezbollah and #Russia "
       "working together*) predicted negative because of the topic words learned in training;\n"
       "- **out-of-vocabulary signals**: emoji (💗, 😜) never appear in the training file, and some "
       "test tweets are not even in English (*\"Lunes de The Walking Dead y Westworld…\"*);\n"
       "- **debatable labels**: *\"Tim Duncan and Kareem both had better careers than Kobe\"* is "
       "labelled neutral, and some training tweets were annotated twice with different labels.\n\n"
       "Task 3 studies some of these cues directly (punctuation, emoticons and emoji, irony markers, "
       "insults)."),

    md("## 8. Summary\n\n"
       "- Preprocessing: lemmatisation helps slightly; removing stop words does not (keep "
       "negations in any case).\n"
       "- Best classical model: **linear SVM on improved Delta TF-IDF** (test macro-F1 0.617).\n"
       "- Best network: **CNN + LSTM** (test macro-F1 0.612, best macro-recall 0.622 and best "
       "recall on negative tweets).\n"
       "- LSTM ≫ simple RNN (+4 points), as expected from the vanishing-gradient problem.\n"
       "- Majority vote, MLP and weighted word2vec do not beat the simple linear models here.\n"
       "- All models lose ~4 points on the test file because of its different period, topics "
       "and class balance."),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("models_analysis.ipynb")
nbf.write(nb, out)
print("written", out)
