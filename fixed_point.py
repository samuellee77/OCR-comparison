from __future__ import annotations

import timeit
from collections import defaultdict
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

from ocr_utils import flatten_sequences_for_classification, sequence_accuracy


def estimate_transition_log_probs(y_seqs, K: int, smoothing: float = 1.0) -> np.ndarray:
    counts = np.zeros((K, K), dtype=float) + smoothing
    for seq in y_seqs:
        seq = np.asarray(seq, dtype=int)
        for t in range(len(seq) - 1):
            counts[int(seq[t]), int(seq[t + 1])] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs)


def fixed_point_inference(log_proba_flat, seq_ids, positions, trans_logp, K: int, alpha: float = 0.5, max_iter: int = 5):
    log_proba_flat = np.asarray(log_proba_flat, dtype=float)
    seq_ids = np.asarray(seq_ids, dtype=int)
    positions = np.asarray(positions, dtype=int)

    groups = defaultdict(list)
    for idx, (s, p) in enumerate(zip(seq_ids, positions)):
        groups[int(s)].append((int(p), idx))

    max_seq = max(groups.keys())
    y_pred_seqs = []

    for s in range(max_seq + 1):
        items = sorted(groups[s])
        idxs = [idx for _, idx in items]
        T = len(idxs)

        y_seq = np.argmax(log_proba_flat[idxs], axis=1).astype(int)

        for _ in range(max_iter):
            changed = False
            for pos in range(T):
                idx_global = idxs[pos]
                scores = log_proba_flat[idx_global].copy()

                if pos > 0:
                    y_left = int(y_seq[pos - 1])
                    scores += alpha * trans_logp[y_left]
                if pos < T - 1:
                    y_right = int(y_seq[pos + 1])
                    scores += alpha * trans_logp[:, y_right]

                y_new = int(np.argmax(scores))
                if y_new != y_seq[pos]:
                    y_seq[pos] = y_new
                    changed = True
            if not changed:
                break

        y_pred_seqs.append(list(y_seq))

    return y_pred_seqs


def run_fixed_point(X_tr, y_tr, X_te, y_te, K: int, C: float = 1.0, alpha: float = 0.5, max_iter: int = 5) -> Dict[str, Any]:
    X_tr_flat, y_tr_flat, tr_seq_ids, tr_positions = flatten_sequences_for_classification(X_tr, y_tr)
    X_te_flat, _, te_seq_ids, te_positions = flatten_sequences_for_classification(X_te, y_te)

    clf = LogisticRegression(C=C, max_iter=200, multi_class="multinomial", solver="lbfgs", n_jobs=-1)
    start_train = timeit.default_timer()
    clf.fit(X_tr_flat, y_tr_flat)
    trans_logp = estimate_transition_log_probs(y_tr, K)
    train_time = timeit.default_timer() - start_train

    start_test = timeit.default_timer()
    log_proba_tr = np.log(clf.predict_proba(X_tr_flat) + 1e-12)
    log_proba_te = np.log(clf.predict_proba(X_te_flat) + 1e-12)

    y_tr_pred = fixed_point_inference(log_proba_tr, tr_seq_ids, tr_positions, trans_logp, K, alpha=alpha, max_iter=max_iter)
    y_te_pred = fixed_point_inference(log_proba_te, te_seq_ids, te_positions, trans_logp, K, alpha=alpha, max_iter=max_iter)
    test_time = timeit.default_timer() - start_test

    tr_word_acc, tr_char_acc = sequence_accuracy(y_tr, y_tr_pred)
    te_word_acc, te_char_acc = sequence_accuracy(y_te, y_te_pred)

    return {
        "clf_local": clf,
        "train_time": train_time,
        "test_time": test_time,
        "train_word_acc": tr_word_acc,
        "train_char_acc": tr_char_acc,
        "test_word_acc": te_word_acc,
        "test_char_acc": te_char_acc,
    }

