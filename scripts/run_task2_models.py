import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from tweet_sentiment.config import MODELS
from tweet_sentiment.data import load_train, load_test
from tweet_sentiment.models import build_models

train, test = load_train(), load_test()
for name, pipe in build_models().items():
    cv = cross_val_score(pipe, train["text"], train["polarity"], cv=5, scoring="f1_macro")
    pipe.fit(train["text"], train["polarity"])
    print(f"=== {name} | CV macro-F1 = {cv.mean():.3f}")
    print(classification_report(test["polarity"], pipe.predict(test["text"])))
    joblib.dump(pipe, MODELS / f"{name}.joblib")
