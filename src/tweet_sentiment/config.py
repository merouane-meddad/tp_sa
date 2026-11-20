from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

TRAIN_FILE = RAW / "train_tweets.txt"
TEST_FILE = RAW / "test_tweets.txt"
LABELS = [-1, 0, 1]
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
RANDOM_STATE = 42

# Shared train/validation split: models are tuned on the validation part,
# the test file is only used once for the final comparison.
VAL_SIZE = 0.1
CV_FOLDS = 5


def ensure_output_dirs():
    for path in (PROCESSED, MODELS, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)
