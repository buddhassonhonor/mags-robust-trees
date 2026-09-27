# Local Margin Preferences in Decision Trees: A Controlled Empirical Study of Robustness

This repository contains the implementation, experiment configurations, per-run results, statistical summaries, figures, and integrity checks for **“Local Margin Preferences in Decision Trees: A Controlled Empirical Study of Robustness.”**

The adjacent-gap preference has a small, heterogeneous mean effect relative to an implementation-matched Gini tree. Pruned CART and adversarially trained single trees have lower flip rates under the evaluated attack.

## Repository contents

```text
README.md                 Reproduction guide and expected results
LICENSE                   MIT license for code
CITATION.cff              Citation metadata for the fixed release
requirements.txt          Pinned Python dependencies
configs/                  Dataset, seed, method, noise, and attack settings
src/                      Margin-aware tree implementation
exp/                      Experiment and figure-generation implementation
scripts/                  Data preparation, execution, statistics, and checks
results/raw/               Final per-run metrics, tree statistics, and attacks
results/summary/           Master result file and aggregate statistical tables
figures/                   Final Figures 1--5 in PNG and PDF
metadata/                  Hardware, software, dataset, and run metadata
tests/                     Unit and artifact-integrity tests
```

## Data

Raw datasets are **not** redistributed in this repository. They remain available from scikit-learn, the UCI Machine Learning Repository, OpenML, and ProPublica under their respective terms. The preparation script downloads or prepares the required public data and records the resolved sources. Set `MAGS_DATA_ROOT` to store data outside the repository if preferred.

```powershell
python scripts/prepare_data.py --data-root data
```

The experiment reads only the selected data root; it does not require author-specific paths or pre-existing OpenML caches.

## Reproduce from a clean Python environment

Python 3.13 is used for the archived run. From a clean checkout on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/prepare_data.py --data-root data
$env:MAGS_DATA_ROOT = (Resolve-Path data)
python scripts/run_core.py --workers 4
python scripts/run_baselines.py --workers 4
python scripts/run_attacks.py --workers 4
python scripts/build_classaware_statistics.py
python scripts/audit_candidate_space.py
python scripts/compare_robust_trees.py
python scripts/summarize_robust_trees.py
python scripts/generate_figures.py
python scripts/validate_results.py
python scripts/audit_consistency.py
python -m pytest -q tests
```

For a single end-to-end command, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1
```

The full 27-dataset, 30-seed run is CPU based. On the reference Intel Core Ultra 9 275HX system with four workers, allow approximately 15--30 minutes for experiments and attacks; data download time depends on the network. Peak memory is kept below 28 GB by limiting each worker and BLAS process to one thread.

## Outputs

- Per-run predictive metrics: `results/raw/results.csv`
- Tree structure and timing: `results/raw/tree_stats.csv`
- Constrained exact opposite-leaf attacks: `results/raw/exact_attacks.csv`
- Binary-class robust-tree comparison: `results/raw/robust_tree_comparison.csv`
- Robust-tree dataset means and paired comparisons: `results/summary/robust_tree_methods.csv` and `results/summary/robust_tree_comparisons.csv`
- Unique numerical source: `results/summary/results_master.json`
- Main comparison statistics: `results/summary/table2_comparisons.csv`
- Attack comparison statistics: `results/summary/table3_attack_comparisons.csv`
- Completeness report: `results/summary/integrity_report.json`
- Figures 1--5: `figures/`
- Reproduction log: `metadata/clean_reproduction.log`

## Expected checks and key results

The main artifact contains 27 benchmark configurations, seeds 0--29, 14 methods at each of five noise levels, and constrained exact attacks for ten single-tree methods: 56,700 predictive records, 11,340 tree-statistic records, and 8,100 attack records. The configurations include one synthetic task and two sources of German Credit. `scripts/validate_results.py` checks completeness and uniqueness of these records.

As cross-checks against `results_master.json`, fixed MAGS minus matched Gini at $\sigma=0.2$ is +0.21 percentage points on average (95% CI -0.08 to 0.50; Wilcoxon $p=0.434$). Under the constrained attack at $\epsilon=0.05$, the flip-rate delta is -0.73 percentage points (Holm-adjusted $p=0.013$). Excluding the synthetic task and duplicate German Credit source gives a noisy-accuracy delta of +0.25 points ($p=0.274$) and flip-rate delta of -0.69 points (unadjusted $p=0.011$).

The robust-tree comparison uses 14 distinct public binary-class datasets and 30 paired splits per dataset. It evaluates GROOT and the GROOT implementation of the Chen-style robust Gini heuristic at the same continuous-feature attack budget, with encoded binary features fixed. Run `python scripts/compare_robust_trees.py` and `python scripts/summarize_robust_trees.py` to reproduce its per-run and summary files. The raw comparison also contains an alternate German Credit source, which the summary excludes to avoid double-counting.

## Prior fixed release

The original benchmark artifact was published as GitHub release
[`v1.0.1`](https://github.com/buddhassonhonor/mags-robust-trees/releases/tag/v1.0.1),
archived on Zenodo with the version-specific DOI
[`10.5281/zenodo.21798553`](https://doi.org/10.5281/zenodo.21798553).
The concept DOI for all versions is
[`10.5281/zenodo.21798552`](https://doi.org/10.5281/zenodo.21798552).

## License

Code is released under the MIT License. Dataset licenses and terms remain those of the original data providers.
