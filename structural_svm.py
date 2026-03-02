from __future__ import annotations

import timeit
from typing import Any, Dict, List

import dlib
import numpy as np

from ocr_utils import sequence_accuracy


def _dlib_to_numpy_1d(v) -> np.ndarray:
    """Robustly convert dlib vectors (and similar) to 1D numpy arrays."""
    try:
        arr = np.asarray(v, dtype=float)
        if arr.size > 0:
            return arr.ravel()
    except Exception:
        pass

    if hasattr(v, "size"):
        n = int(v.size())
        return np.array([v[i] for i in range(n)], dtype=float)

    if hasattr(v, "__len__"):
        return np.array(list(v), dtype=float)

    raise TypeError(f"Unsupported weight type: {type(v)}")


class LinearChainSSVMProblem:
    C = 1.0

    def __init__(self, samples, labels, K: int, d_window: int):
        self.samples = [np.asarray(x, dtype=float) for x in samples]
        self.labels = [np.asarray(y, dtype=int) for y in labels]
        self.K = int(K)
        self.d_window = int(d_window)
        self.num_samples = len(self.samples)
        self.num_dimensions = self.K * self.d_window + self.K * self.K
        self.loss_for_loop = True  # Hamming loss

    def make_psi(self, X_seq, y_seq):
        X_seq = np.asarray(X_seq, dtype=float)
        y_seq = np.asarray(y_seq, dtype=int)
        T, d_w = X_seq.shape
        assert d_w == self.d_window

        psi = dlib.vector()
        psi.resize(self.num_dimensions)

        # Unary: sum_t 1[y_t=k] * x_t
        for t in range(T):
            y = int(y_seq[t])
            base = y * self.d_window
            xt = X_seq[t]
            for j in range(self.d_window):
                psi[base + j] += float(xt[j])

        # Pairwise: sum_t 1[y_t=i, y_{t+1}=j]
        offset_trans = self.K * self.d_window
        for t in range(T - 1):
            y_prev = int(y_seq[t])
            y_curr = int(y_seq[t + 1])
            psi[offset_trans + y_prev * self.K + y_curr] += 1.0

        return psi

    def get_truth_joint_feature_vector(self, idx: int):
        return self.make_psi(self.samples[idx], self.labels[idx])

    def separation_oracle(self, idx: int, current_solution):
        """Loss-augmented inference via Viterbi."""
        X = self.samples[idx]
        y_true = self.labels[idx]
        T, d_w = X.shape
        K = self.K

        w = _dlib_to_numpy_1d(current_solution)
        w_unary = w[: K * d_w].reshape(K, d_w)
        w_trans = w[K * d_w :].reshape(K, K)

        local = X.dot(w_unary.T)  # (T,K)
        for t in range(T):
            yt = int(y_true[t])
            for y in range(K):
                if y != yt:
                    local[t, y] += 1.0

        backp = np.zeros((T, K), dtype=int)
        dp = np.zeros((T, K), dtype=float)
        dp[0] = local[0]
        for t in range(1, T):
            for y in range(K):
                scores = dp[t - 1] + w_trans[:, y]
                best_prev = int(np.argmax(scores))
                dp[t, y] = scores[best_prev] + local[t, y]
                backp[t, y] = best_prev

        y_hat = [0] * T
        y_hat[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 2, -1, -1):
            y_hat[t] = int(backp[t + 1, y_hat[t + 1]])

        y_hat = np.asarray(y_hat, dtype=int)
        loss = float(np.sum(y_hat != y_true))
        psi_hat = self.make_psi(X, y_hat)
        return loss, psi_hat


def viterbi_decode_sequence(X_seq, weights, K: int, d_window: int):
    X = np.asarray(X_seq, dtype=float)
    T, d_w = X.shape
    assert d_w == d_window

    w = _dlib_to_numpy_1d(weights)
    w_unary = w[: K * d_w].reshape(K, d_w)
    w_trans = w[K * d_w :].reshape(K, K)

    local = X.dot(w_unary.T)
    backp = np.zeros((T, K), dtype=int)
    dp = np.zeros((T, K), dtype=float)
    dp[0] = local[0]
    for t in range(1, T):
        for y in range(K):
            scores = dp[t - 1] + w_trans[:, y]
            best_prev = int(np.argmax(scores))
            dp[t, y] = scores[best_prev] + local[t, y]
            backp[t, y] = best_prev

    y_hat = [0] * T
    y_hat[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 2, -1, -1):
        y_hat[t] = int(backp[t + 1, y_hat[t + 1]])
    return y_hat


def run_structured_svm(X_tr, y_tr, X_te, y_te, K: int, d_window: int, C: float = 1.0) -> Dict[str, Any]:
    problem = LinearChainSSVMProblem(X_tr, y_tr, K=K, d_window=d_window)
    problem.C = C

    start_train = timeit.default_timer()
    weights = dlib.solve_structural_svm_problem(problem)
    train_time = timeit.default_timer() - start_train

    start_test = timeit.default_timer()
    y_tr_pred = [viterbi_decode_sequence(x, weights, K, d_window) for x in X_tr]
    y_te_pred = [viterbi_decode_sequence(x, weights, K, d_window) for x in X_te]
    test_time = timeit.default_timer() - start_test

    tr_word_acc, tr_char_acc = sequence_accuracy(y_tr, y_tr_pred)
    te_word_acc, te_char_acc = sequence_accuracy(y_te, y_te_pred)

    return {
        "weights": weights,
        "train_time": train_time,
        "test_time": test_time,
        "train_word_acc": tr_word_acc,
        "train_char_acc": tr_char_acc,
        "test_word_acc": te_word_acc,
        "test_char_acc": te_char_acc,
    }

