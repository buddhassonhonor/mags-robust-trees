# A Lightweight Margin-Aware Split Preference for Interpretable Decision Trees

This repository is the reproducibility artifact for the paper **“A Lightweight Margin-Aware Split Preference for Interpretable Decision Trees.”** It contains the implementation, experiment configurations, final per-run results, statistical summaries, manuscript figures, environment metadata, and integrity tests used for the submitted revision.

The evidence supports a deliberately narrow conclusion: the adjacent-gap preference has a small, heterogeneous mean effect relative to an implementation-matched Gini tree and does not establish a general predictive or adversarial-robustness advantage.

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
python scripts/build_statistics.py
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
- Unique numerical source: `results/summary/results_master.json`
- Main comparison statistics: `results/summary/table2_comparisons.csv`
- Attack comparison statistics: `results/summary/table3_attack_comparisons.csv`
- Completeness report: `results/summary/integrity_report.json`
- Figures 1--5: `figures/`
- Reproduction log: `metadata/clean_reproduction.log`

## Expected checks and key results

The final artifact contains 27 datasets, seeds 0--29, 12 methods at each of five noise levels, and constrained exact attacks for all eight single-tree methods: 48,600 predictive records, 9,720 tree-statistic records, and 6,480 attack records. `scripts/validate_results.py` exits nonzero if a dataset--seed--method task is missing, duplicated, or marked as failed.

As cross-checks against `results_master.json`, fixed MAGS minus matched Gini at $\sigma=0.2$ is 0.17 percentage points on average (95% CI -0.07 to 0.41), with median 0.00, 12/5/10 wins/ties/losses, and Wilcoxon $p=0.527$. Under the constrained attack at $\epsilon=0.05$, the fixed-MAGS minus matched-Gini flip-rate delta is -0.52 percentage points (95% CI -0.95 to -0.09; Holm-adjusted $p=0.072$). These values are checksums for the archived run, not broad performance claims.

## Fixed release

The submission cites the fixed GitHub release
[`v1.0.1`](https://github.com/buddhassonhonor/mags-robust-trees/releases/tag/v1.0.1),
archived on Zenodo with the version-specific DOI
[`10.5281/zenodo.21798553`](https://doi.org/10.5281/zenodo.21798553).
The concept DOI for all versions is
[`10.5281/zenodo.21798552`](https://doi.org/10.5281/zenodo.21798552).

## License

Code is released under the MIT License. Dataset licenses and terms remain those of the original data providers.
