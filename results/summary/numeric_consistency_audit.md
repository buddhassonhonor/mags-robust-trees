# Numerical Consistency Audit

Unique source: `results/summary/results_master.json`, regenerated from the final raw CSV files.

| Check | Status | Evidence |
|---|---:|---|
| Common seeds | PASS | Every method uses seeds 0--29 |
| Method count | PASS | Observed 14 methods |
| Primary mean delta | PASS | 0.21 pp; 95% CI [-0.08, 0.50] |
| Primary median and rank test | PASS | median 0.00 pp; p=0.434; W/T/L 11/6/10 |
| Tuned MAGS vs tuned CART | PASS | mean 0.27 pp; median 0.00 pp |
| Tuned MAGS vs pruned CART | PASS | mean -0.15 pp; median -0.26 pp |
| Matched constrained-attack delta | PASS | -0.73 pp; W/T/L 16/5/6; Holm p=0.013 |
| Table 3 W/T/L direction | PASS | Wins follow lower flip rate and higher attacked accuracy |
| Table 2 reporting fields | PASS | Family, raw/Holm p, effect, CI, and W/T/L are present |
| Table 3 paired fields | PASS | Paired delta, raw/Holm p, effect, CI, direction, and W/T/L are present |
| Figures 1--5 | PASS | PNG and PDF present for all five figures |
| Main-manuscript numerical strings | PASS | Primary, tuned, pruned, attack, and seed statements match the master |
| No stale numerical statements | PASS | All primary headline values match the master; no reduced-seed description remains |

Overall status: **PASS**.
