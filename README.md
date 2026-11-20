# Tweet Sentiment Analysis

Sentiment analysis of tweets (negative / neutral / positive) with descriptive statistics, the
machine learning models of the course (Ch3 *Analyse des sentiments*, Ch4 *Perceptron multicouches*
and RNN) and a study of tweet metadata (URLs, emoticons, punctuation, irony, censored insults).

The main findings, in French, are in [`reports/RESULTS.md`](reports/RESULTS.md).

## Dataset

Two files are expected in `data/raw/`:

| File | Purpose |
|---|---|
| `train_tweets.txt` | Training data (50,306 tweets, 2011-2015) |
| `test_tweets.txt` | Test data (12,284 tweets, Nov.-Dec. 2016) |

Each line contains three tab-separated fields (plus an optional empty one):

| Field | Type | Description |
|---|---|---|
| `idf` | integer | Tweet identifier |
| `polarity` | integer | Sentiment: `-1` negative, `0` neutral, `+1` positive |
| `text` | string | Raw tweet text |

The loader (`src/tweet_sentiment/data.py`) repairs the encoding artefacts of the raw files: CSV
quotes with doubled inner quotes, literal JSON escapes (`\u2019`), HTML entities (`&amp;`), and
"mojibake" in the test file (`ðŸ˜‚` instead of 😂). `deduplicate()` removes tweets annotated with
contradictory labels and repeated tweets before modelling.

## Tasks

1. **Descriptive analysis**: class balance (train vs test), lengths, frequent and polar words
   (improved Delta TF-IDF), hashtags and mentions, visible markers.
2. **Sentiment classification**: preprocessing comparison (stop words, lemmatisation, stemming);
   TF-IDF, Delta TF-IDF and weighted word2vec representations; Naive Bayes, linear SVM, Maximum
   Entropy and their majority vote; multilayer perceptron; simple RNN, LSTM and CNN+LSTM.
3. **Metadata analysis**: URLs, emoticons and emoji, punctuation, emphasis, irony cues, censored
   insults; chi-square / Kruskal-Wallis tests; contribution to the classifier (ablation).

## Project structure

```
tweet-sentiment/
├── data/
│   ├── raw/                 # train_tweets.txt, test_tweets.txt
│   └── processed/           # cleaned or intermediate data
├── src/tweet_sentiment/
│   ├── config.py            # paths, labels, random seed, validation split
│   ├── data.py              # loading, encoding repair, deduplication, splits
│   ├── patterns.py          # shared regexes (URLs, emoticons, insults, irony)
│   ├── preprocessing.py     # cleaning, tokenisation, stop words, lemma, stem
│   ├── features.py          # Delta TF-IDF (improved) and weighted word2vec
│   ├── descriptive.py       # Task 1
│   ├── models.py            # Task 2: NB, SVM, ME, vote, MLP
│   ├── advanced_models.py   # Task 2: RNN, LSTM, CNN+LSTM (PyTorch)
│   ├── evaluation.py        # metrics, confusion matrices, error analysis
│   └── metadata.py          # Task 3
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_task1_descriptive.py
│   ├── run_task2_models.py
│   ├── run_task2_neural.py
│   └── run_task3_metadata.py
├── notebooks/
│   ├── descriptive_analysis.ipynb   # Task 1
│   ├── models_analysis.ipynb        # Task 2
│   ├── metadata_analysis.ipynb      # Task 3
│   └── build_*_notebook.py          # regenerate the notebooks
├── tests/                   # pytest
├── models/                  # saved models (.joblib, .pt)
└── reports/
    ├── RESULTS.md           # summary of the results (French)
    ├── figures/             # task1_*, task2_*, task3_* plots
    └── tables/              # task1_*, task2_*, task3_* CSV outputs
```

## Installation

Requires Python 3.9 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader stopwords wordnet omw-1.4
```

PyTorch is only needed for `run_task2_neural.py`. The RNNs use PyTorch rather than Keras because
TensorFlow has no stable release for Python 3.14 yet.

## Usage

```bash
python scripts/run_task1_descriptive.py   # ~20 s
python scripts/run_task2_models.py        # ~15 min (CV + grid search); --quick reuses saved params
python scripts/run_task2_neural.py        # ~10 min on CPU
python scripts/run_task3_metadata.py      # ~6 min
pytest
```

The notebooks of Tasks 2 and 3 reuse the tables and models saved by the scripts, so run the
scripts first (or set `RETRAIN = True` / `RECOMPUTE_ABLATION = True` in their first cell).

## Evaluation protocol

- The deduplicated training file is split into 90 % train / 10 % validation (stratified, seed 42).
- Preprocessing and hyper-parameters are chosen by 5-fold cross-validation on the train part;
  the RNNs use early stopping on the validation part.
- The test file is used once for the final comparison. Its class balance differs from the training
  file (32 % negative vs 16 %), so we report **macro-F1**, **macro-recall** (official SemEval-2017
  metric) and **F1-PN** (mean F1 of positive and negative), and use balanced class weights.

## Results

See `reports/tables/task2_all_models.csv` for all metrics.

| Model | Val macro-F1 | Test macro-F1 | Test macro-recall |
|---|---|---|---|
| linear SVM (Delta TF-IDF) | 0.658 | 0.617 | 0.614 |
| linear SVM (TF-IDF) | 0.656 | 0.606 | 0.606 |
| vote NB + SVM + ME (TF-IDF) | 0.656 | 0.610 | 0.611 |
| ME / logistic regression (TF-IDF) | 0.654 | 0.610 | 0.613 |
| CNN + LSTM (word2vec init) | 0.651 | 0.612 | 0.622 |
| LSTM (word2vec init) | 0.650 | 0.609 | 0.620 |
| MLP (TF-IDF, 1 hidden layer) | 0.621 | 0.583 | 0.592 |
| simple RNN (word2vec init) | 0.617 | 0.572 | 0.585 |
| linear SVM (weighted word2vec) | 0.615 | 0.566 | 0.593 |
| NB (TF-IDF) | 0.610 | 0.585 | 0.597 |

Main observations: Delta TF-IDF gives the best linear model; LSTM-based networks have the best
recall on the negative class and the best macro-recall; the simple RNN is clearly below the LSTM
(vanishing gradients); the majority vote does not beat its best member. Metadata add about one
point of validation macro-F1, almost entirely through punctuation.

## Limitations

- The test set comes from a later period with different topics: every model loses about 4 points
  between validation and test.
- Irony and censored-insult detectors are rule-based. They were checked by hand on samples
  (`task3_validation_samples.csv`): explicit irony markers are rare and often literal.
- word2vec is trained on the 44,000 training tweets only. Embeddings pre-trained on large tweet
  corpora would likely improve the neural models.
