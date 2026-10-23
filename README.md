# Tweet Sentiment Analysis

Sentiment analysis of tweets using descriptive statistics, classical machine learning models, and a study of tweet metadata (URLs, emoticons, punctuation, irony, censored insults).

## Dataset

Two files are expected in `data/raw/`:

| File | Purpose |
|---|---|
| `train_tweets.txt` | Training data |
| `test_tweets.txt` | Test data |

Each line contains three fields:

| Field | Type | Description |
|---|---|---|
| `idf` | integer | Tweet identifier |
| `polarity` | integer | Sentiment: `-1` negative, `0` neutral, `+1` positive |
| `text` | string | Raw tweet text |

> The loader in `src/tweet_sentiment/data.py` assumes tab-separated fields. If your files use another separator, change the `sep` argument of `load_tweets`.

## Tasks

1. **Descriptive analysis** of the tweets: class balance, text lengths, frequent words.
2. **Sentiment classification** with machine learning models (TF-IDF with Naive Bayes, logistic regression, linear SVM).
3. **Metadata analysis**: URLs, emoticons, punctuation, irony cues, censored insults.

## Project structure

```
tweet-sentiment/
├── data/
│   ├── raw/                 # train_tweets.txt, test_tweets.txt
│   └── processed/           # cleaned or intermediate data
├── src/tweet_sentiment/
│   ├── config.py            # paths, labels, random seed
│   ├── data.py              # data loading
│   ├── preprocessing.py     # text cleaning
│   ├── descriptive.py       # Task 1
│   ├── models.py            # Task 2
│   └── metadata.py          # Task 3
├── scripts/
│   ├── run_task1_descriptive.py
│   ├── run_task2_models.py
│   └── run_task3_metadata.py
├── notebooks/               # exploration
├── models/                  # saved models (.joblib)
├── reports/
    ├── figures/             # plots
    └── tables/              # CSV outputs

```

## Installation

Requires Python 3.9 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Place the data files in `data/raw/`, then run each task:

```bash
python scripts/run_task1_descriptive.py   # tables and figures in reports/
python scripts/run_task2_models.py        # cross-validation scores, test report, saved models
python scripts/run_task3_metadata.py      # metadata averages per polarity class
```

Run the tests with:

```bash
pytest
```

## Methods

### Task 1: Descriptive analysis
- Distribution of polarity classes
- Tweet length in characters and words, per class
- Planned: most frequent words and n-grams per class, word clouds, hashtags and mentions

### Task 2: Sentiment models
- Preprocessing: lowercasing, URLs replaced by `URL`, mentions by `USER`, hashtags stripped of `#`
- Features: TF-IDF on unigrams and bigrams
- Classifiers: Multinomial Naive Bayes, Logistic Regression, Linear SVM (class weights balanced)
- Evaluation: 5-fold cross-validated macro-F1 on the training set, then precision, recall and F1 per class on the test set
- Trained pipelines are saved to `models/`

### Task 3: Metadata
Features extracted per tweet and compared across polarity classes:

| Feature | Description |
|---|---|
| `n_urls` | Number of URLs |
| `n_emoticons` | Number of emoticons such as `:)`, `;D`, `<3` |
| `n_exclam`, `n_question`, `n_ellipsis` | Punctuation counts |
| `n_caps_words` | Fully capitalised words |
| `n_censored` | Censored insults such as `f**k` |
| `irony_cue` | Simple irony markers (`#sarcasm`, `/s`, "yeah right", ...) |

The irony and insult detectors are rule-based heuristics and should be validated on a sample of tweets.

## Results

_To be completed after running the scripts._

| Model | CV macro-F1 | Test macro-F1 |
|---|---|---|
| Naive Bayes | | |
| Logistic Regression | | |
| Linear SVM | | |

## Possible improvements

- Hyperparameter search with grid or random search
- French or English stopword removal and lemmatization
- Error analysis and confusion matrices
- Pretrained embeddings or a transformer model for comparison
- Using the metadata features as inputs to the classifier