import argparse
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

from ocr_utils import (
    read_OCR,
    build_word_sequences,
    make_sliding_window_features,
    train_test_split_sequences,
)
from auto_context import run_auto_context
from fixed_point import run_fixed_point


# Default problem parameters
N_WORDS = 5000  # total words to consider
D = 128  # per-character feature length
K = 26  # number of labels (a-z)


def parse_window_radii(raw: str) -> list[int]:
    window_radii = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not window_radii:
        raise ValueError("At least one window radius must be provided.")
    return window_radii


def parse_splits(raw: str) -> list[tuple[int, int]]:
    splits: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            n_train_str, n_test_str = item.split(":")
            splits.append((int(n_train_str), int(n_test_str)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid split '{item}'. Expected format n_train:n_test."
            ) from exc
    if not splits:
        raise ValueError("At least one train/test split must be provided.")
    return splits


def normalize_models(model_args: list[str]) -> list[str]:
    if "all" in model_args:
        return ["struct_svm", "crf", "auto_context", "fixed_point"]
    return model_args


def build_wandb_config(
    max_words: int,
    model: str,
    random_state,
    window_radius: int,
    n_train: int,
    n_test: int,
    repeat_index: int,
):
    return {
        "n_words": max_words,
        "d": D,
        "k": K,
        "model": model,
        "random_state": random_state,
        "window_radius": window_radius,
        "window_size": 2 * window_radius + 1,
        "n_train": n_train,
        "n_test": n_test,
        "repeat_index": repeat_index,
    }


def get_wandb_module(enabled: bool):
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception:
        print("wandb not available; skipping logging. Install with: pip install wandb")
        return None

    return wandb


def start_wandb_run(wandb, project: str, run_name: str | None, config: dict):
    if wandb is None:
        return None

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        reinit="finish_previous",
    )
    return run


def add_metric_columns(row: dict) -> dict:
    row = dict(row)
    row["train_word_acc"] = 1.0 - row["train_word_error"]
    row["test_word_acc"] = 1.0 - row["test_word_error"]
    row["train_char_acc"] = 1.0 - row["train_char_error"]
    row["test_char_acc"] = 1.0 - row["test_char_error"]
    return row


def run_model(method: str, X_tr, y_tr, X_te, y_te, d_window: int) -> dict:
    if method == "auto_context":
        res = run_auto_context(X_tr, y_tr, X_te, y_te, K=K)
    elif method == "fixed_point":
        res = run_fixed_point(X_tr, y_tr, X_te, y_te, K=K)
    else:
        raise ValueError(f"Unsupported model: {method}")

    return {
        "method": method,
        "train_word_error": 1.0 - res["train_word_acc"],
        "test_word_error": 1.0 - res["test_word_acc"],
        "train_char_error": 1.0 - res["train_char_acc"],
        "test_char_error": 1.0 - res["test_char_acc"],
        "train_time": res["train_time"],
        "test_time": res["test_time"],
    }


def log_single_run_to_wandb(wandb, row: dict):
    if wandb is None:
        return

    row_with_acc = add_metric_columns(row)
    wandb.log(
        {
            "window_radius": row_with_acc["window_radius"],
            "window_size": row_with_acc["window_size"],
            "n_train": row_with_acc["n_train"],
            "n_test": row_with_acc["n_test"],
            "repeat_index": row_with_acc["repeat_index"],
            "train_word_acc": row_with_acc["train_word_acc"],
            "test_word_acc": row_with_acc["test_word_acc"],
            "train_char_acc": row_with_acc["train_char_acc"],
            "test_char_acc": row_with_acc["test_char_acc"],
            "train_word_error": row_with_acc["train_word_error"],
            "test_word_error": row_with_acc["test_word_error"],
            "train_char_error": row_with_acc["train_char_error"],
            "test_char_error": row_with_acc["test_char_error"],
            "train_time": row_with_acc["train_time"],
            "test_time": row_with_acc["test_time"],
        }
    )
    for key, value in row_with_acc.items():
        if key != "method":
            wandb.summary[key] = value


def run_smoke_tests(word_features, word_labels, models, random_state=None, wandb=None):
    """Run small sanity tests for each classifier on a tiny subset."""
    SMOKE_WINDOW_RADIUS = 1  # window size = 3
    SMOKE_TRAIN = 50
    SMOKE_TEST = 50

    print("=== Smoke tests ===")
    X_smoke = make_sliding_window_features(word_features, SMOKE_WINDOW_RADIUS)
    d_window_smoke = X_smoke[0].shape[1]
    print("Feature size:", d_window_smoke)

    X_tr_s, y_tr_s, X_te_s, y_te_s = train_test_split_sequences(
        X_smoke, word_labels, SMOKE_TRAIN, SMOKE_TEST, random_state=random_state
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

    if "auto_context" in models:
        res_auto_smoke = run_auto_context(X_tr_s, y_tr_s, X_te_s, y_te_s, K=K)
        _record("auto_context", res_auto_smoke)

    if "fixed_point" in models:
        res_fp_smoke = run_fixed_point(X_tr_s, y_tr_s, X_te_s, y_te_s, K=K)
        _record("fixed_point", res_fp_smoke)

    smoke_df = pd.DataFrame(smoke_rows).sort_values("method")
    smoke_df = smoke_df.apply(add_metric_columns, axis=1, result_type="expand")
    print("\nSmoke test summary:")
    print(smoke_df.to_string(index=False))

    return smoke_df


def run_full_experiments(
    word_features,
    word_labels,
    models,
    random_state=None,
    num_repeats: int = 3,
    window_radii: list[int] | None = None,
    splits: list[tuple[int, int]] | None = None,
    wandb=None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    max_words: int = N_WORDS,
):
    """Run the full grid of experiments and return a DataFrame."""
    if window_radii is None:
        window_radii = [0, 1, 2]
    if splits is None:
        splits = [(1000, 4000), (2500, 2500), (4000, 1000)]

    results: list[dict] = []

    print("=== Full experiments ===")
    for W in window_radii:
        print(f"==== Window radius {W} (window size {2 * W + 1}) ====")
        X_win = make_sliding_window_features(word_features, W)
        d_window = X_win[0].shape[1]

        for n_train, n_test in splits:
            print(f"-- Split train/test = {n_train}/{n_test}")
            for repeat_index in range(num_repeats):
                X_tr, y_tr, X_te, y_te = train_test_split_sequences(
                    X_win,
                    word_labels,
                    n_train,
                    n_test,
                    random_state=random_state,
                )

                for method in models:
                    row = run_model(method, X_tr, y_tr, X_te, y_te, d_window=d_window)
                    row.update(
                        {
                            "repeat_index": repeat_index,
                            "window_radius": W,
                            "window_size": 2 * W + 1,
                            "n_train": n_train,
                            "n_test": n_test,
                        }
                    )
                    if wandb is not None:
                        suffix = (
                            f"{method}-w{W}-tr{n_train}-te{n_test}-r{repeat_index + 1}"
                        )
                        run_name = (
                            f"{wandb_run_name}-{suffix}"
                            if wandb_run_name
                            else suffix
                        )
                        run = start_wandb_run(
                            wandb,
                            project=wandb_project,
                            run_name=run_name,
                            config=build_wandb_config(
                                max_words=max_words,
                                model=method,
                                random_state=random_state,
                                window_radius=W,
                                n_train=n_train,
                                n_test=n_test,
                                repeat_index=repeat_index,
                            ),
                        )
                        log_single_run_to_wandb(wandb, row)
                        if run is not None:
                            run.finish()
                    results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        ["method", "window_radius", "n_train", "repeat_index"]
    )
    results_df = results_df.apply(add_metric_columns, axis=1, result_type="expand")

    by_window_df = (
        results_df.groupby(["method", "window_radius", "window_size"], as_index=False)
        .agg(
            runs=("repeat_index", "size"),
            mean_train_word_error=("train_word_error", "mean"),
            mean_test_word_error=("test_word_error", "mean"),
            mean_train_char_error=("train_char_error", "mean"),
            mean_test_char_error=("test_char_error", "mean"),
            mean_train_time=("train_time", "mean"),
            mean_test_time=("test_time", "mean"),
        )
        .sort_values(["method", "window_radius"])
    )
    by_split_df = (
        results_df.groupby(["method", "n_train", "n_test"], as_index=False)
        .agg(
            runs=("repeat_index", "size"),
            mean_train_word_error=("train_word_error", "mean"),
            mean_test_word_error=("test_word_error", "mean"),
            mean_train_char_error=("train_char_error", "mean"),
            mean_test_char_error=("test_char_error", "mean"),
            mean_train_time=("train_time", "mean"),
            mean_test_time=("test_time", "mean"),
        )
        .sort_values(["method", "n_train", "n_test"])
    )

    print("\nFinal results (head):")
    print(results_df.head().to_string(index=False))
    print("\nMean results by window size:")
    print(by_window_df.to_string(index=False))
    print("\nMean results by train/test split:")
    print(by_split_df.to_string(index=False))

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
        "--wandb_project",
        default="ocr-comparison",
        dest="wandb_project",
        help="W&B project name (if enabled).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--random-state",
        "--random_state",
        type=int,
        default=None,
        dest="random_state",
        help="Optional RNG seed for dataset shuffling and train/test splits. Default: None.",
    )
    parser.add_argument(
        "--max-words",
        "--max_words",
        type=int,
        default=N_WORDS,
        dest="max_words",
        help=f"Maximum number of words to load. Default: {N_WORDS}.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for CSV summaries. Default: outputs.",
    )
    parser.add_argument(
        "--num-repeats",
        "--num_repeats",
        type=int,
        default=3,
        dest="num_repeats",
        help="Number of repeated runs per window/split setting. Default: 3.",
    )
    parser.add_argument(
        "--window-radii",
        "--window_radii",
        default="0,1,2",
        dest="window_radii",
        help="Comma-separated window radii for full experiments. Default: 0,1,2.",
    )
    parser.add_argument(
        "--splits",
        default="1000:4000,2500:2500,4000:1000",
        help="Comma-separated train:test splits for full experiments. Default: 1000:4000,2500:2500,4000:1000.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the small smoke tests (skip full experiments).",
    )
    parser.add_argument(
        "--model",
        choices=["auto_context", "fixed_point", "all"],
        help="Single model alias for sweep agents.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["auto_context", "fixed_point", "all"],
        default=["auto_context", "fixed_point"],
        help=(
            "Which models to run. Choose one or more of "
            "'auto_context', 'fixed_point', or 'all'. "
            "Default: all models."
        ),
    )

    args = parser.parse_args()

    if args.model is not None:
        models = normalize_models([args.model])
    else:
        models = normalize_models(args.models)
    window_radii = parse_window_radii(args.window_radii)
    splits = parse_splits(args.splits)

    wandb = get_wandb_module(enabled=not args.no_wandb)

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
        dataset1,
        max_words=args.max_words,
        shuffle=True,
        random_state=args.random_state,
    )
    print("Number of words:", len(word_features))

    if args.smoke_only:
        # Smoke tests
        smoke_df = run_smoke_tests(
            word_features,
            word_labels,
            models=models,
            random_state=args.random_state,
            wandb=wandb,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        smoke_df.to_csv(output_dir / "smoke_results.csv", index=False)
        print("Smoke-only mode; skipping full experiments.")
    else:
        # Full experiments
        results_df = run_full_experiments(
            word_features,
            word_labels,
            models=models,
            random_state=args.random_state,
            num_repeats=args.num_repeats,
            window_radii=window_radii,
            splits=splits,
            wandb=wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            max_words=args.max_words,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / "full_results.csv", index=False)
        (
            results_df.groupby(["method", "window_radius", "window_size"], as_index=False)
            .agg(
                mean_train_word_error=("train_word_error", "mean"),
                mean_test_word_error=("test_word_error", "mean"),
                mean_train_char_error=("train_char_error", "mean"),
                mean_test_char_error=("test_char_error", "mean"),
                mean_train_time=("train_time", "mean"),
                mean_test_time=("test_time", "mean"),
            )
            .sort_values(["method", "window_radius"])
            .to_csv(output_dir / "full_results_by_window.csv", index=False)
        )
        (
            results_df.groupby(["method", "n_train", "n_test"], as_index=False)
            .agg(
                mean_train_word_error=("train_word_error", "mean"),
                mean_test_word_error=("test_word_error", "mean"),
                mean_train_char_error=("train_char_error", "mean"),
                mean_test_char_error=("test_char_error", "mean"),
                mean_train_time=("train_time", "mean"),
                mean_test_time=("test_time", "mean"),
            )
            .sort_values(["method", "n_train", "n_test"])
            .to_csv(output_dir / "full_results_by_split.csv", index=False)
        )

if __name__ == "__main__":
    main()
