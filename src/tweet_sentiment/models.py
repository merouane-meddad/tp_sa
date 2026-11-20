"""Task 2 - sentiment classification with the models of the course.

Ch3 (Analyse des sentiments)
    * text representation: bag-of-words TF-IDF, improved Delta TF-IDF,
      word2vec tweet vectors weighted by Delta TF-IDF;
    * classifiers: Naive Bayes (NB), linear SVM, Maximum Entropy (ME =
      multinomial logistic regression), and their majority vote
      ("Architecture globale - SVM + NB + ME");
    * preprocessing: stop words, lemmatisation, stemming (compared).
Ch4 (Perceptron multicouches)
    * multilayer perceptron on TF-IDF features (here, scikit-learn);
    * recurrent networks (RNN, LSTM, CNN+LSTM) in ``advanced_models.py``.

Every model is a scikit-learn pipeline with ``fit(texts, labels)`` and
``predict(texts)`` (team interface).
"""
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE
from .features import DeltaTfidfVectorizer, WeightedWord2Vec
from .preprocessing import TextNormalizer

# Preprocessing variants compared in Task 2 (course: stop words, lemma, stem)
PREPROCESSING_VARIANTS = {
    "clean only": TextNormalizer(stop_words=False, reduce=None),
    "stop words": TextNormalizer(stop_words=True, reduce=None),
    "stop words + lemma": TextNormalizer(stop_words=True, reduce="lemma"),
    "stop words + stem": TextNormalizer(stop_words=True, reduce="stem"),
    "lemma (stop words kept)": TextNormalizer(stop_words=False, reduce="lemma"),
}


def tfidf(normalizer=None, ngram_range=(1, 2), min_df=2, max_features=None):
    return TfidfVectorizer(tokenizer=normalizer or TextNormalizer(), lowercase=False,
                           token_pattern=None, ngram_range=ngram_range, min_df=min_df,
                           sublinear_tf=True, max_features=max_features)


def naive_bayes(alpha=0.3):
    # fit_prior=False: uniform class priors, the NB equivalent of class
    # weighting (the training priors differ from the test priors).
    return MultinomialNB(alpha=alpha, fit_prior=False)


def maxent(C=2.0):
    return LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                              random_state=RANDOM_STATE)


def linear_svm(C=0.1):
    return LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE)


def build_models(normalizer=None, params=None):
    """Classical pipelines. ``params`` overrides hyper-parameters, e.g.
    ``{"svm_C": 0.2}`` (see ``PARAM_GRIDS`` for the tuned values)."""
    p = {"nb_alpha": 0.3, "me_C": 2.0, "svm_C": 0.1, "delta_svm_C": 0.3,
         "delta_scheme": "ovr", "w2v_C": 0.1}
    p.update(params or {})
    norm = normalizer or TextNormalizer()

    return {
        "NB (TF-IDF)": Pipeline([("vec", tfidf(norm)), ("clf", naive_bayes(p["nb_alpha"]))]),
        "ME / logistic regression (TF-IDF)": Pipeline([("vec", tfidf(norm)),
                                                       ("clf", maxent(p["me_C"]))]),
        "linear SVM (TF-IDF)": Pipeline([("vec", tfidf(norm)), ("clf", linear_svm(p["svm_C"]))]),
        "linear SVM (Delta TF-IDF)": Pipeline([
            ("vec", DeltaTfidfVectorizer(tokenizer=norm, scheme=p["delta_scheme"])),
            ("clf", linear_svm(p["delta_svm_C"]))]),
        "linear SVM (weighted word2vec)": Pipeline([
            ("vec", WeightedWord2Vec(tokenizer=norm, random_state=RANDOM_STATE)),
            ("scale", StandardScaler()),
            ("clf", linear_svm(p["w2v_C"]))]),
        "vote NB + SVM + ME (TF-IDF)": Pipeline([
            ("vec", tfidf(norm)),
            ("clf", VotingClassifier([("nb", naive_bayes(p["nb_alpha"])),
                                      ("svm", linear_svm(p["svm_C"])),
                                      ("me", maxent(p["me_C"]))], voting="hard"))]),
        "MLP (TF-IDF, 1 hidden layer)": build_mlp(norm),
    }


def build_mlp(normalizer=None, hidden=(256,), k_best=30000, alpha=1e-4):
    """Multilayer perceptron (course Ch4) on the 30 000 TF-IDF n-grams most
    associated with the label (chi2), ReLU, Adam, early stopping on 10 % of
    the training data. Class imbalance is handled with sample weights."""
    return Pipeline([
        ("vec", tfidf(normalizer)),
        ("select", SelectKBest(chi2, k=k_best)),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, batch_size=256,
                              learning_rate_init=1e-3, max_iter=30, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=3,
                              random_state=RANDOM_STATE)),
    ])


# Grids searched with 5-fold cross-validation on the training split.
PARAM_GRIDS = {
    "NB (TF-IDF)": {"clf__alpha": [0.1, 0.3, 1.0]},
    "ME / logistic regression (TF-IDF)": {"clf__C": [0.5, 1.0, 2.0, 5.0]},
    "linear SVM (TF-IDF)": {"clf__C": [0.03, 0.1, 0.3, 1.0]},
    "linear SVM (Delta TF-IDF)": {"vec__scheme": ["pos_neg", "ovr"], "clf__C": [0.1, 0.3, 1.0]},
}
# grid parameter -> build_models() parameter
GRID_TO_PARAM = {("NB (TF-IDF)", "clf__alpha"): "nb_alpha",
                 ("ME / logistic regression (TF-IDF)", "clf__C"): "me_C",
                 ("linear SVM (TF-IDF)", "clf__C"): "svm_C",
                 ("linear SVM (Delta TF-IDF)", "clf__C"): "delta_svm_C",
                 ("linear SVM (Delta TF-IDF)", "vec__scheme"): "delta_scheme"}

