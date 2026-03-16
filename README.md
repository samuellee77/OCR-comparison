# OCR Comparison

This repository compares two sequence-labeling approaches for OCR word recognition on the classic `letter.data` dataset:

- `auto_context`: a two-stage classifier that feeds neighboring predicted labels back in as contextual features
- `fixed_point`: a local classifier followed by iterative inference using learned transition statistics

The project runs controlled experiments over different train/test splits and sliding-window feature sizes, then saves CSV summaries and plots for comparison.

## Repository Layout

- [run_ocr_experiments.py](/Users/camellee/OCR-comparison/run_ocr_experiments.py): main experiment runner
- [ocr_utils.py](/Users/camellee/OCR-comparison/ocr_utils.py): dataset loading, sequence building, feature generation, and evaluation helpers
- [auto_context.py](/Users/camellee/OCR-comparison/auto_context.py): auto-context model
- [fixed_point.py](/Users/camellee/OCR-comparison/fixed_point.py): fixed-point inference model
- [OCRdataset/letter.data](/Users/camellee/OCR-comparison/OCRdataset/letter.data): OCR character dataset used by the experiments
- [outputs/full_results.csv](/Users/camellee/OCR-comparison/outputs/full_results.csv): raw experiment results across all runs
- [output/jupyter-notebook/ocr-results-comparison-plots.ipynb](/Users/camellee/OCR-comparison/output/jupyter-notebook/ocr-results-comparison-plots.ipynb): notebook for visualizing split, window-size, and model effects

## What The Repo Does

The workflow is:

1. Load OCR character examples from `letter.data`.
2. Group characters into word-level sequences.
3. Build sliding-window features with configurable window radii.
4. Run repeated experiments across:
   - train/test splits such as `1000/4000`, `2500/2500`, `4000/1000`
   - window sizes `1`, `3`, and `5`
   - both `auto_context` and `fixed_point`
5. Save per-run and aggregated CSV results.
6. Visualize the comparisons in a notebook or from the generated plot files in `outputs/`.

The main evaluation metrics are:

- word accuracy / word error
- character accuracy / character error
- train time
- test time

## Setup

This project uses Python and a small set of scientific Python dependencies.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Run the default experiment grid and disable Weights & Biases logging:

```bash
python run_ocr_experiments.py --no-wandb
```

This runs:

- models: `auto_context`, `fixed_point`
- window radii: `0,1,2` which correspond to window sizes `1,3,5`
- splits: `1000:4000`, `2500:2500`, `4000:1000`
- repeats: `3`

Results are written to `outputs/`, including:

- `full_results.csv`
- `full_results_by_window.csv`
- `full_results_by_split.csv`

### Useful Variants

Run smoke tests only:

```bash
python run_ocr_experiments.py --smoke-only --no-wandb
```

Run a single model:

```bash
python run_ocr_experiments.py --model auto_context --no-wandb
python run_ocr_experiments.py --model fixed_point --no-wandb
```

Customize window sizes, splits, or repeats:

```bash
python run_ocr_experiments.py \
  --window-radii 0,1,2 \
  --splits 1000:4000,2500:2500,4000:1000 \
  --num-repeats 5 \
  --no-wandb
```

Set a seed for reproducibility:

```bash
python run_ocr_experiments.py --random-state 42 --no-wandb
```
