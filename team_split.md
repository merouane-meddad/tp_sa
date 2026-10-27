# Tweet Sentiment Project: 5-Person Split

## Roles and ownership

Each person owns one module of the repo, so people rarely edit the same file and merge conflicts stay low.

| # | Role | Owns | Main deliverables |
| --- | --- | --- | --- |
| 1 | Data and infrastructure lead | `data.py`, `preprocessing.py`, `config.py`, `tests/`, git setup | Reliable loaders (separator, encoding, bad lines), text cleaning, train/validation split, repo and branch rules |
| 2 | Task 1: descriptive analysis | `descriptive.py`, `run_task1_descriptive.py` | Class balance, length stats, top words and n-grams per class, hashtags and mentions, word clouds. Also edits the final report |
| 3 | Task 2a: baseline models | `models.py`, `run_task2_models.py` | TF-IDF pipelines (Naive Bayes, logistic regression, SVM), cross-validation, hyperparameter search |
| 4 | Task 2b: evaluation and advanced models | `evaluation.py` (new), `advanced_models.py` (new) | Confusion matrices, per-class F1, error analysis, one stronger model (embeddings or small transformer) for comparison |
| 5 | Task 3: metadata | `metadata.py`, `run_task3_metadata.py` | URLs, emoticons, punctuation, irony cues, censored insults, with a statistical comparison across polarity classes |

## Timeline

Work runs in three phases, and person 1's loader comes first because everyone else depends on it.

1. **Phase 0, first day.** Person 1 pushes the working loader and `clean_text` to `main` as quickly as possible, even if rough. The others read the data, and the whole team agrees on the shared interfaces below.
2. **Phase 1, parallel work.** Each person works on their own branch and module.
3. **Phase 2, integration.** Person 5 hands the metadata features to persons 3 and 4, who test them as extra classifier inputs. Person 4 compares all models on the same split. Person 2 assembles the report and figures.

## Interfaces to agree on first

Five contracts keep the modules interchangeable, so people can build in parallel without waiting on each other.

- `load_train()` and `load_test()` return a DataFrame with columns `idf`, `polarity`, `text`.
- `clean_text(str) -> str` stays stable. Changes go through person 1.
- `extract_metadata(df)` returns the same DataFrame with added numeric columns.
- Every model is a scikit-learn style pipeline with `fit(texts, labels)` and `predict(texts)`, so persons 3 and 4 can swap models freely.
- Everyone uses the same random seed and the same train/validation split from `config.py`.

## Git workflow

Everyone works on a feature branch and merges through a reviewed pull request, so `main` always runs.

- `main` is protected. Branches are named `feature/<name>`, and each PR has one reviewer (for example 3 reviews 4, and 2 reviews 5).
- Only person 1 edits `config.py`, `data.py` and `preprocessing.py`. Others request changes rather than editing them.
- Commit small and merge often. Don't wait until the end.
- Outputs (figures, tables, models) go to their own folders. Prefix file names by task, such as `task1_*.png`, so nobody overwrites another person's files.

## Risks and a smaller team

The main risk is that person 1 becomes a bottleneck on day one, because the other four need working loaders before they can start.

- **Early bottleneck.** The first push to `main` should happen within the first few hours, even if rough.
- **Uneven workload.** Task 2 is split across two people, while person 2 writes less code, so person 2 also takes the report and slides.
- **Metadata decision.** Whether to feed metadata features to the classifier is a team decision, best made once the Phase 1 baselines exist.
- **Fewer people.** With a team of four, merge roles 2 and 5. The structure stays the same.