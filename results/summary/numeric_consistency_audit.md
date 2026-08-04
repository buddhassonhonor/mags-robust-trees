# Numerical Consistency Audit

Unique source: `results/summary/results_master.json`, regenerated from the three final raw CSV files.

| Check | Status | Evidence |
|---|---:|---|
| Raw-result integrity | PASS | 48,600 metrics; 9,720 tree statistics; 6,480 attacks; zero missing tasks |
| Common seeds | PASS | Every method uses seeds 0--29 |
| Method count | PASS | Observed 12 methods |
| Primary mean delta | PASS | 0.17 pp; 95% CI [-0.07, 0.41] |
| Primary median and rank test | PASS | median 0.00 pp; p=0.527; W/T/L 12/5/10 |
| Tuned MAGS vs tuned CART | PASS | mean 0.19 pp; median 0.03 pp |
| Tuned MAGS vs pruned CART | PASS | mean -0.23 pp; median -0.26 pp |
| Matched constrained-attack delta | PASS | -0.52 pp; Holm p=0.072 |
| Table 2 reporting fields | PASS | Family, raw/Holm p, effect, CI, and W/T/L are present |
| Table 3 paired fields | PASS | Paired delta, raw/Holm p, effect, CI, and W/T/L are present |
| Method summaries | PASS | 12 accuracy methods and 8 attacked tree methods |
| Archived runtime summary | PASS | Tuned MAGS mean fit time 1.331 s |
| Figures 1--5 | PASS | PNG and PDF present for all five figures |
| Main-manuscript numerical strings | PASS | Primary, tuned, pruned, attack, and seed statements match the master |
| Reply numerical strings | PASS | Primary and corrected medians, 30-seed scope, and R1/R2 labels match |
| No stale numerical statements | PASS | All primary headline values match the master; no reduced-seed description remains |

Overall status: **PASS**.
