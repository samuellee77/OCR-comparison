from __future__ import annotations

import timeit
from collections import defaultdict
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

from ocr_utils import flatten_sequences_for_classification, sequence_accuracy


def build_auto_context_features(X_base, seq_ids, positions, y_pred_stage, K: int):
    X_base = np.asarray(X_base, dtype=float)
    y_pred_stage = np.asarray(y_pred_stage, dtype=int)
    N, d = X_base.shape
    num_states = K + 1  # boundary state
    feat_size = d + 2 * num_states

    X_ac = np.zeros((N, feat_size), dtype=float)
    X_ac[:, :d] = X_base

    boundary_state = K

    groups = defaultdict(list)
    for idx, (s, p) in enumerate(zip(seq_ids, positions)):
        groups[int(s)].append((int(p), idx))
    for s in groups:
        groups[s].sort()

    for _, plist in groups.items():
        T = len(plist)
        for local_pos, (_, idx) in enumerate(plist):
            if local_pos > 0:
                idx_left = plist[local_pos - 1][1]
                state_left = int(y_pred_stage[idx_left])
            else:
                state_left = boundary_state

            if local_pos < T - 1:
                idx_right = plist[local_pos + 1][1]
                state_right = int(y_pred_stage[idx_right])
            else:
                state_right = boundary_state

            X_ac[idx, d + state_left] = 1.0
            X_ac[idx, d + num_states + state_right] = 1.0

    return X_ac


def reconstruct_sequences_from_flat(seq_ids, positions, y_flat):
    seq_ids = np.asarray(seq_ids, dtype=int)
    positions = np.asarray(positions, dtype=int)
    y_flat = np.asarray(y_flat, dtype=int)

    groups = defaultdict(list)
    for idx, (s, p) in enumerate(zip(seq_ids, positions)):
        groups[int(s)].append((int(p), int(y_flat[idx])))

    max_seq = max(groups.keys())
    sequences = []
    for s in range(max_seq + 1):
        items = sorted(groups[s])
        sequences.append([lab for _, lab in items])
    return sequences


def run_auto_context(X_tr, y_tr, X_te, y_te, K: int, C: float = 1.0) -> Dict[str, Any]:
    X_tr_flat, y_tr_flat, tr_seq_ids, tr_positions = flatten_sequences_for_classification(X_tr, y_tr)
    X_te_flat, _, te_seq_ids, te_positions = flatten_sequences_for_classification(X_te, y_te)

    clf1 = LogisticRegression(C=C, max_iter=200, multi_class="multinomial", solver="lbfgs", n_jobs=-1)
    start_train = timeit.default_timer()
    clf1.fit(X_tr_flat, y_tr_flat)

    y_tr_pred_stage1 = clf1.predict(X_tr_flat)
    y_te_pred_stage1 = clf1.predict(X_te_flat)

    X_tr_ac = build_auto_context_features(X_tr_flat, tr_seq_ids, tr_positions, y_tr_pred_stage1, K)
    X_te_ac = build_auto_context_features(X_te_flat, te_seq_ids, te_positions, y_te_pred_stage1, K)

    clf2 = LogisticRegression(C=C, max_iter=200, multi_class="multinomial", solver="lbfgs", n_jobs=-1)
    clf2.fit(X_tr_ac, y_tr_flat)
    train_time = timeit.default_timer() - start_train

    start_test = timeit.default_timer()
    y_tr_pred_flat = clf2.predict(X_tr_ac)
    y_te_pred_flat = clf2.predict(X_te_ac)
    test_time = timeit.default_timer() - start_test

    y_tr_pred = reconstruct_sequences_from_flat(tr_seq_ids, tr_positions, y_tr_pred_flat)
    y_te_pred = reconstruct_sequences_from_flat(te_seq_ids, te_positions, y_te_pred_flat)

    tr_word_acc, tr_char_acc = sequence_accuracy(y_tr, y_tr_pred)
    te_word_acc, te_char_acc = sequence_accuracy(y_te, y_te_pred)

    return {
        "clf_local": clf1,
        "clf_context": clf2,
        "train_time": train_time,
        "test_time": test_time,
        "train_word_acc": tr_word_acc,
        "train_char_acc": tr_char_acc,
        "test_word_acc": te_word_acc,
        "test_char_acc": te_char_acc,
    }

