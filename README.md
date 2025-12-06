Team: Newton
-- Yaw Danso: Primary coder and implementation
-- Sean Bruno: Testing and validation, bug fixes
# Team Newton — UNM ML Test

## Team Members
- **Yaw Danso** — Primary coder and implementation
- **Sean Bruno** — Testing and validation, bug fixes

## Kaggle Score
- **60.272%**

## Requirements
- Python 3.x
- Python packages: `pandas`, `scikit-learn`, `numpy`, `seaborn`, `matplotlib`

## Quickstart
1. Download the training data and place the files in the `data/` directory.
2. Run `scripts/project_setup.py` to initialize data and create trees:

```
python3 scripts/project_setup.py
```

3. Run individual scripts in the `scripts/` directory to generate specific results.

## Parquet cache and dev tips
- To rebuild the parquet cache from the original CSV files (force CSV read and rewrite parquet):

```
python3 scripts/project_setup.py --rebuild-cache
```

- To load only a subset of rows (useful for fast development/testing):

```
python3 scripts/project_setup.py --nrows 10000
```

- Recommended venv setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## File Manifest
- `EXAMPLE` — Example output from running on Linux
- `README.md` — This file
- `scripts/project_setup.py` — Run first (setup)
- `scripts/chi_square_test.py` — Chi-Squared implementation
- `scripts/fraud_detection.py` — Fraud detection pipeline
- `scripts/visualizations.py` — Generates graphs used in the paper
- `scripts/decision_tree.py` — Decision tree implementation
- `scripts/random_forest.py` — Run and generate random forests
- `scripts/information_gain.py` — Calculate information gain, GINI, etc.
- `data/` — Place data CSV files here

## Expected Data Files
The repository expects the following files in `data/`:

```
sample_sub.csv
test.csv
train.csv
```

## Notes
- If you see references to `README.txt`, this has been replaced by `README.md`.
- For reproducibility, run `scripts/project_setup.py` before running other scripts.

