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
RANDOM_STATE = 42
