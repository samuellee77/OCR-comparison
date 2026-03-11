from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


def l2i(a: str) -> int:
    return int(ord(a) - ord("a"))


def i2l(i: int) -> str:
    if i >= 0:
        return chr(i + ord("a"))
    return "_"


def iors(s):
    try:
        return int(s)
    except ValueError:
        return s


def read_OCR(filename: str, n_features: int) -> Dict:
    with open(filename) as f:
        dataset = {
            "ids": [],
            "labels": [],
            "labelDic": {},
            "next_ids": [],
            "word_ids": [],
            "positions": [],
            "folds": [],
            "features": [],
        }

        for str_line in f.readlines():
            line0 = list(map(iors, filter(None, re.split("\t", str_line.strip()))))

            dataset["ids"].append(int(line0.pop(0)))
            lab = l2i(line0.pop(0))
            dataset["labels"].append(lab)
            dataset["labelDic"][lab] = dataset["labelDic"].get(lab, 0) + 1

            dataset["next_ids"].append(int(line0.pop(0)))
            dataset["word_ids"].append(int(line0.pop(0)))
            dataset["positions"].append(int(line0.pop(0)))
            dataset["folds"].append(int(line0.pop(0)))

            if len(line0) != n_features:
                raise ValueError(f"Unexpected feature length: {len(line0)} != {n_features}")
            dataset["features"].append(line0)

    return dataset


def build_word_sequences(dataset: Dict, max_words: int = 5000, shuffle: bool = True, random_state: Optional[int] = None):
    """Group flat OCR letters into word-level sequences.

    Returns
    -------
    word_features : list of np.ndarray, each of shape (T_i, D)
    word_labels   : list of np.ndarray, each of shape (T_i,)
    """
    word_to_indices = defaultdict(list)
    for idx, wid in enumerate(dataset["word_ids"]):
        word_to_indices[wid].append(idx)

    word_ids = sorted(word_to_indices.keys())
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(word_ids)

    if max_words is not None:
        word_ids = word_ids[:max_words]

    word_features = []
    word_labels = []
    for wid in word_ids:
        idxs = sorted(word_to_indices[wid], key=lambda i: dataset["positions"][i])
        seq_feats = np.asarray([dataset["features"][i] for i in idxs], dtype=float)
        seq_labels = np.asarray([dataset["labels"][i] for i in idxs], dtype=int)
        word_features.append(seq_feats)
        word_labels.append(seq_labels)

    return word_features, word_labels


def make_sliding_window_features(word_features, window_radius: int):
    """Create per-character features using a sliding window over each word.

    window_radius = 0  -> window size 1 (current character only)
    window_radius = 1  -> window size 3 (previous, current, next), etc.
    """
    if window_radius == 0:
        return [np.asarray(X, dtype=float) for X in word_features]

    out = []
    for X in word_features:
        X = np.asarray(X, dtype=float)
        T, d = X.shape
        win_size = 2 * window_radius + 1
        d_win = d * win_size
        Xw = np.zeros((T, d_win), dtype=float)
        for t in range(T):
            feats = []
            for offset in range(-window_radius, window_radius + 1):
                pos = t + offset
                if 0 <= pos < T:
                    feats.append(X[pos])
                else:
                    feats.append(np.zeros(d, dtype=float))
            Xw[t] = np.concatenate(feats, axis=0)
        out.append(Xw)
    return out


def train_test_split_sequences(X_seqs, y_seqs, n_train: int, n_test: int, random_state: int = 0):
    """Random train/test split on word-level sequences."""
    assert len(X_seqs) == len(y_seqs)
    total = len(X_seqs)
    if n_train + n_test > total:
        raise ValueError(f"Requested n_train+n_test={n_train+n_test} > total={total}")

    rng = np.random.RandomState(random_state)
    indices = np.arange(total)
    rng.shuffle(indices)
    selected = indices[: n_train + n_test]
    tr_idx = selected[:n_train]
    te_idx = selected[n_train : n_train + n_test]

    X_tr = [X_seqs[i] for i in tr_idx]
    y_tr = [y_seqs[i] for i in tr_idx]
    X_te = [X_seqs[i] for i in te_idx]
    y_te = [y_seqs[i] for i in te_idx]
    return X_tr, y_tr, X_te, y_te


def sequence_accuracy(y_true, y_pred) -> Tuple[float, float]:
    """Compute word-level and character-level accuracy."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of sequences")

    word_correct = 0
    for yt, yp in zip(y_true, y_pred):
        if np.array_equal(np.asarray(yt, dtype=int), np.asarray(yp, dtype=int)):
            word_correct += 1
    word_acc = float(word_correct) / float(len(y_true))

    correct_chars = 0
    total_chars = 0
    for yt, yp in zip(y_true, y_pred):
        yt = np.asarray(yt, dtype=int)
        yp = np.asarray(yp, dtype=int)
        L = min(len(yt), len(yp))
        correct_chars += int(np.sum(yt[:L] == yp[:L]))
        total_chars += int(len(yt))
    char_acc = float(correct_chars) / float(total_chars)
    return word_acc, char_acc


def flatten_sequences_for_classification(X_seqs, y_seqs):
    """Flatten word-level sequences into per-character examples."""
    X_list = []
    y_list = []
    seq_ids = []
    positions = []
    for s_idx, (X, y) in enumerate(zip(X_seqs, y_seqs)):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        T = len(y)
        for t in range(T):
            X_list.append(X[t])
            y_list.append(y[t])
            seq_ids.append(s_idx)
            positions.append(t)
    return (
        np.asarray(X_list, dtype=float),
        np.asarray(y_list, dtype=int),
        np.asarray(seq_ids, dtype=int),
        np.asarray(positions, dtype=int),
    )

