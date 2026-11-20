import numpy as np
import scipy.sparse as sp

from tweet_sentiment.data import deduplicate, fix_encoding
from tweet_sentiment.features import delta_idf
from tweet_sentiment.metadata import extract_metadata
from tweet_sentiment.patterns import EMO_NEG_RE, EMO_POS_RE, URL_RE, is_censored_swear
from tweet_sentiment.preprocessing import TextNormalizer, clean_text

import pandas as pd


def test_fix_encoding():
    assert fix_encoding('"He said ""hi"" today"') == 'He said "hi" today'
    assert fix_encoding("I don\\u2019t care\\u002c ok") == "I don’t care, ok"
    assert fix_encoding("salt &amp; pepper") == "salt & pepper"


def test_emoticons_ignore_urls():
    text = "look https://t.co/8pDxA :) but :("
    no_url = URL_RE.sub(" ", text)
    assert EMO_POS_RE.findall(no_url) == [":)"]
    assert EMO_NEG_RE.findall(no_url) == [":("]
    assert EMO_NEG_RE.findall(URL_RE.sub(" ", "https://example.com")) == []


def test_clean_text_tokens():
    assert clean_text("Soooo GOOD :) @bob #Win http://x.co/a") == "soo good EMOPOS USER win URL"


def test_negations_are_not_stop_words():
    tokens = TextNormalizer(stop_words=True)("I do not like the film, I don't")
    assert "not" in tokens and "don't" in tokens and "the" not in tokens


def test_censored_swear():
    assert is_censored_swear("f**k") and is_censored_swear("sh*t") and is_censored_swear("a$$")
    assert not is_censored_swear("A$AP") and not is_censored_swear("#rap")


def test_delta_idf_reproduces_course_table():
    # slide "DELTA TF-IDF améliorée": |N|/|P| = 0.59, alpha = 0.5
    n_pos, n_neg = 1000, 590
    y = np.array([1] * n_pos + [-1] * n_neg)
    cols = []
    for p_w, n_w in ((100, 0), (2, 0), (100, 1)):
        col = np.zeros(len(y))
        col[:p_w] = 1
        col[n_pos:n_pos + n_w] = 1
        cols.append(col)
    w = delta_idf(sp.csr_matrix(np.column_stack(cols)), y)
    assert np.allclose(w, [6.90, 1.57, 5.32], atol=0.05)


def test_deduplicate_drops_conflicts():
    df = pd.DataFrame({"idf": [1, 1, 2, 3], "polarity": [0, -1, 1, 1],
                       "text": ["same", "same", "hello there", "hello there"]})
    out = deduplicate(df)
    assert list(out["text"]) == ["hello there"]


def test_extract_metadata_counts():
    df = pd.DataFrame({"polarity": [1], "text": ["WOW!!! so f**k #sarcasm :) https://t.co/x"]})
    m = extract_metadata(df).iloc[0]
    assert m["n_urls"] == 1 and m["n_emo_pos"] == 1 and m["n_emo_neg"] == 0
    assert m["multi_exclam"] == 1 and m["n_censored"] == 1 and m["irony_marker"] == 1


def test_fix_mojibake():
    assert fix_encoding("so funny ðŸ˜‚ and moreâ€¦") == "so funny 😂 and more…"
    assert fix_encoding("café") == "café"
