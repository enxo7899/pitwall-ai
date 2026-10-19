# PITWALL AI · test-set results (sample 1, n = 120 general questions)

| Arm | Joint success | Style ≥ 2 | Correct | Helpful ≥ 2 | Median tokens | Cut off |
|---|---|---|---|---|---|---|
| A | 0/120 (0.0%, 95% CI 0.0–3.1) | 0/120 (0.0%, 95% CI 0.0–3.1) | 67/120 (55.8%, 95% CI 46.9–64.4) | 72/120 (60.0%, 95% CI 51.1–68.3) | 331.0 | 0 |
| B | 20/120 (16.7%, 95% CI 11.1–24.3) | 70/120 (58.3%, 95% CI 49.4–66.8) | 48/120 (40.0%, 95% CI 31.7–48.9) | 45/120 (37.5%, 95% CI 29.4–46.4) | 279.5 | 3 |
| C | 30/120 (25.0%, 95% CI 18.1–33.4) | 105/120 (87.5%, 95% CI 80.4–92.3) | 45/120 (37.5%, 95% CI 29.4–46.4) | 37/120 (30.8%, 95% CI 23.3–39.6) | 284.5 | 0 |
| C_3epoch | 24/120 (20.0%, 95% CI 13.8–28.0) | 73/120 (60.8%, 95% CI 51.9–69.1) | 55/120 (45.8%, 95% CI 37.2–54.7) | 50/120 (41.7%, 95% CI 33.2–50.6) | 296.5 | 0 |

| Comparison (paired, 10,000 bootstrap resamples) | Joint success Δ (pp) | 95% CI | McNemar p | Verdict |
|---|---|---|---|---|
| C_vs_B | +8.3 | -0.8 to +17.5 | 0.099 | no reliable difference |
| C_vs_A | +25.0 | +17.5 to +33.3 | 0.000 | first arm better |
| B_vs_A | +16.7 | +10.0 to +24.2 | 0.000 | first arm better |
| C_vs_C_3epoch | +5.0 | -4.2 to +14.2 | 0.362 | no reliable difference |
