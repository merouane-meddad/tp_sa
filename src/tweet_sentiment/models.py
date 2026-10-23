"""Task 2 - sentiment classification."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .config import RANDOM_STATE
from .preprocessing import clean_text


def build_models():
    def tfidf():
        return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    return {
        "naive_bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "logreg": Pipeline([("tfidf", tfidf()),
                            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                       random_state=RANDOM_STATE))]),
        "linear_svm": Pipeline([("tfidf", tfidf()),
                                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))]),
    }

# TODO: cross-validation, grid search, confusion matrix, error analysis
