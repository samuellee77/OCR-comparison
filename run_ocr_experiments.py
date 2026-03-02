import argparse
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

from ocr_utils import (
    read_OCR,
    build_word_sequences,
    make_sliding_window_features,
    train_test_split_sequences,
)
from structural_svm import run_structured_svm
from crf_model import run_crf
from auto_context import run_auto_context
from fixed_point import run_fixed_point


# Default problem parameters
N_WORDS = 5000  # total words to consider
D = 128  # per-character feature length
K = 26  # number of labels (a-z)


def setup_wandb(enabled: bool, project: str, run_name: str | None):
    """Initialize Weights & Biases if requested and available."""
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception:
        print("wandb not available; skipping logging. Install with: pip install wandb")
        return None

    wandb.init(
        project=project,
        name=run_name,
        config={"N_WORDS": N_WORDS, "D": D, "K": K},
    )
    return wandb


def run_smoke_tests(word_features, word_labels, models, wandb=None):
    """Run small sanity tests for each classifier on a tiny subset."""
    SMOKE_WINDOW_RADIUS = 1  # window size = 3
    SMOKE_TRAIN = 50
    SMOKE_TEST = 50

    print("=== Smoke tests ===")
    X_smoke = make_sliding_window_features(word_features, SMOKE_WINDOW_RADIUS)
    d_window_smoke = X_smoke[0].shape[1]
    print("Feature size:", d_window_smoke)

    X_tr_s, y_tr_s, X_te_s, y_te_s = train_test_split_sequences(
        X_smoke, word_labels, SMOKE_TRAIN, SMOKE_TEST, random_state=0
    )

    smoke_rows: list[dict] = []

    def _record(name: str, res: dict):
        smoke_rows.append(
            {
                "method": name,
                "train_word_error": 1.0 - res["train_word_acc"],
                "test_word_error": 1.0 - res["test_word_acc"],
                "train_char_error": 1.0 - res["train_char_acc"],
                "test_char_error": 1.0 - res["test_char_acc"],
                "train_time": res["train_time"],
                "test_time": res["test_time"],
            }
        )

    if "struct_svm" in models:
        res_ssvm_smoke = run_structured_svm(
            X_tr_s, y_tr_s, X_te_s, y_te_s, K=K, d_window=d_window_smoke
        )
        _record("struct_svm", res_ssvm_smoke)

    if "crf" in models:
        res_crf_smoke = run_crf(X_tr_s, y_tr_s, X_te_s, y_te_s)
        _record("crf", res_crf_smoke)

    if "auto_context" in models:
        res_auto_smoke = run_auto_context(X_tr_s, y_tr_s, X_te_s, y_te_s, K=K)
        _record("auto_context", res_auto_smoke)

    if "fixed_point" in models:
        res_fp_smoke = run_fixed_point(X_tr_s, y_tr_s, X_te_s, y_te_s, K=K)
        _record("fixed_point", res_fp_smoke)

    smoke_df = pd.DataFrame(smoke_rows).sort_values("method")
    print("\nSmoke test summary:")
    print(smoke_df.to_string(index=False))

    if wandb is not None:
        wandb.log({"smoke/results": wandb.Table(dataframe=smoke_df)})

    return smoke_df


def run_full_experiments(word_features, word_labels, models, wandb=None):
    """Run the full grid of experiments and return a DataFrame."""
    window_radii = [0, 1, 2]  # window sizes 1, 3, 5
    splits = [(1000, 4000), (2500, 2500), (4000, 1000)]

    results: list[dict] = []
    wandb_step = 0

    def maybe_wandb_log(row: dict):
        nonlocal wandb_step
        if wandb is not None:
            wandb.log(
                {
                    "method": row["method"],
                    "window_radius": row["window_radius"],
                    "window_size": row["window_size"],
                    "n_train": row["n_train"],
                    "n_test": row["n_test"],
                    "train_word_error": row["train_word_error"],
                    "test_word_error": row["test_word_error"],
                    "train_char_error": row["train_char_error"],
                    "test_char_error": row["test_char_error"],
                    "train_time": row["train_time"],
                    "test_time": row["test_time"],
                },
                step=wandb_step,
            )
            wandb_step += 1

    print("=== Full experiments ===")
    for W in window_radii:
        print(f"==== Window radius {W} (window size {2 * W + 1}) ====")
        X_win = make_sliding_window_features(word_features, W)
        d_window = X_win[0].shape[1]

        for n_train, n_test in splits:
            print(f"-- Split train/test = {n_train}/{n_test}")
            X_tr, y_tr, X_te, y_te = train_test_split_sequences(
                X_win, word_labels, n_train, n_test, random_state=0
            )

            if "struct_svm" in models:
                res_ssvm = run_structured_svm(
                    X_tr, y_tr, X_te, y_te, K=K, d_window=d_window
                )
                row = {
                    "method": "struct_svm",
                    "window_radius": W,
                    "window_size": 2 * W + 1,
                    "n_train": n_train,
                    "n_test": n_test,
                    "train_word_error": 1.0 - res_ssvm["train_word_acc"],
                    "test_word_error": 1.0 - res_ssvm["test_word_acc"],
                    "train_char_error": 1.0 - res_ssvm["train_char_acc"],
                    "test_char_error": 1.0 - res_ssvm["test_char_acc"],
                    "train_time": res_ssvm["train_time"],
                    "test_time": res_ssvm["test_time"],
                }
                results.append(row)
                maybe_wandb_log(row)

            if "crf" in models:
                res_crf = run_crf(X_tr, y_tr, X_te, y_te)
                row = {
                    "method": "crf",
                    "window_radius": W,
                    "window_size": 2 * W + 1,
                    "n_train": n_train,
                    "n_test": n_test,
                    "train_word_error": 1.0 - res_crf["train_word_acc"],
                    "test_word_error": 1.0 - res_crf["test_word_acc"],
                    "train_char_error": 1.0 - res_crf["train_char_acc"],
                    "test_char_error": 1.0 - res_crf["test_char_acc"],
                    "train_time": res_crf["train_time"],
                    "test_time": res_crf["test_time"],
                }
                results.append(row)
                maybe_wandb_log(row)

            if "auto_context" in models:
                res_auto = run_auto_context(X_tr, y_tr, X_te, y_te, K=K)
                row = {
                    "method": "auto_context",
                    "window_radius": W,
                    "window_size": 2 * W + 1,
                    "n_train": n_train,
                    "n_test": n_test,
                    "train_word_error": 1.0 - res_auto["train_word_acc"],
                    "test_word_error": 1.0 - res_auto["test_word_acc"],
                    "train_char_error": 1.0 - res_auto["train_char_acc"],
                    "test_char_error": 1.0 - res_auto["test_char_acc"],
                    "train_time": res_auto["train_time"],
                    "test_time": res_auto["test_time"],
                }
                results.append(row)
                maybe_wandb_log(row)

            if "fixed_point" in models:
                res_fp = run_fixed_point(X_tr, y_tr, X_te, y_te, K=K)
                row = {
                    "method": "fixed_point",
                    "window_radius": W,
                    "window_size": 2 * W + 1,
                    "n_train": n_train,
                    "n_test": n_test,
                    "train_word_error": 1.0 - res_fp["train_word_acc"],
                    "test_word_error": 1.0 - res_fp["test_word_acc"],
                    "train_char_error": 1.0 - res_fp["train_char_acc"],
                    "test_char_error": 1.0 - res_fp["test_char_acc"],
                    "train_time": res_fp["train_time"],
                    "test_time": res_fp["test_time"],
                }
                results.append(row)
                maybe_wandb_log(row)

    results_df = pd.DataFrame(results).sort_values(
        ["method", "window_radius", "n_train"]
    )

    print("\nFinal results (head):")
    print(results_df.head().to_string(index=False))

    if wandb is not None:
        wandb.log({"results/table": wandb.Table(dataframe=results_df)})

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR structured prediction experiments (SSVM, CRF, auto-context, fixed-point)."
    )
    parser.add_argument(
        "--data-path",
        default="OCRdataset/letter.data",
        help="Path to OCR letter.data file.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="ocr-comparison",
        help="W&B project name (if enabled).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the small smoke tests (skip full experiments).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["struct_svm", "crf", "auto_context", "fixed_point", "all"],
        default=["struct_svm", "crf", "auto_context", "fixed_point"],
        help=(
            "Which models to run. Choose one or more of "
            "'struct_svm', 'crf', 'auto_context', 'fixed_point', or 'all'. "
            "Default: all models."
        ),
    )

    args = parser.parse_args()

    # Normalize model list
    if "all" in args.models:
        models = ["struct_svm", "crf", "auto_context", "fixed_point"]
    else:
        models = args.models

    # Initialize wandb (optional)
    wandb = setup_wandb(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
    )

    # Load dataset
    print("Loading dataset from", args.data_path)
    dataset1 = read_OCR(args.data_path, D)
    print(
        "max label=",
        max(dataset1["labels"]),
        "min label=",
        min(dataset1["labels"]),
        "#labels=",
        len(dataset1["labelDic"]),
    )
    print("Total letters=", len(dataset1["ids"]))
    print("Features shape=", np.array(dataset1["features"]).shape)

    # Build word-level sequences
    word_features, word_labels = build_word_sequences(
        dataset1, max_words=N_WORDS, shuffle=True, random_state=0
    )
    print("Number of words:", len(word_features))

    if args.smoke_only:
        # Smoke tests
        smoke_df = run_smoke_tests(word_features, word_labels, models=models, wandb=wandb)
        print("Smoke-only mode; skipping full experiments.")
    else:
        # Full experiments
        results_df = run_full_experiments(word_features, word_labels, models=models, wandb=wandb)


if __name__ == "__main__":
    main()

