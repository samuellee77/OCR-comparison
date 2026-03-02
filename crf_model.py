from __future__ import annotations

import timeit
from typing import Any, Dict

import numpy as np
import sklearn_crfsuite

from ocr_utils import i2l, l2i, sequence_accuracy


def sequences_to_crf_features(X_seqs):
    X_out = []
    for X in X_seqs:
        X = np.asarray(X, dtype=float)
        T, d = X.shape
        seq_feats = []
        for t in range(T):
            f = {f"f{j}": float(X[t, j]) for j in range(d)}
            f["bias"] = 1.0
            seq_feats.append(f)
        X_out.append(seq_feats)
    return X_out


def sequences_to_crf_labels(y_seqs):
    return [[i2l(int(y)) for y in seq] for seq in y_seqs]


def run_crf(X_tr, y_tr, X_te, y_te) -> Dict[str, Any]:
    X_tr_crf = sequences_to_crf_features(X_tr)
    X_te_crf = sequences_to_crf_features(X_te)
    y_tr_str = sequences_to_crf_labels(y_tr)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )

    start_train = timeit.default_timer()
    crf.fit(X_tr_crf, y_tr_str)
    train_time = timeit.default_timer() - start_train

    start_test = timeit.default_timer()
    y_tr_pred_str = crf.predict(X_tr_crf)
    y_te_pred_str = crf.predict(X_te_crf)
    test_time = timeit.default_timer() - start_test

    y_tr_pred = [[l2i(ch) for ch in seq] for seq in y_tr_pred_str]
    y_te_pred = [[l2i(ch) for ch in seq] for seq in y_te_pred_str]

    tr_word_acc, tr_char_acc = sequence_accuracy(y_tr, y_tr_pred)
    te_word_acc, te_char_acc = sequence_accuracy(y_te, y_te_pred)

    return {
        "model": crf,
        "train_time": train_time,
        "test_time": test_time,
        "train_word_acc": tr_word_acc,
        "train_char_acc": tr_char_acc,
        "test_word_acc": te_word_acc,
        "test_char_acc": te_char_acc,
    }

