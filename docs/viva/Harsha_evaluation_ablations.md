# Viva guide — Harsha (Evaluation protocol, metrics & ablations)

**Your files:** `notebooks/05_ablation_studies.ipynb`, `notebooks/06_final_evaluation.ipynb`, Sections A and F of `notebooks/00_common.ipynb`
(`CFG`, `MODEL_CFG`, `set_seed`, `pinball`, `score`, `attach_actuals`, `plot_fan`). Report Sections V–VI. Slides 11–13.

## 1. What you built, in 60 seconds
- The **shared configuration** (origins, horizon, quantiles, default hyper-parameters, seeds) so every model runs under one protocol.
- The **metrics** (MAE, RMSE, WAPE, pinball, coverage, width) used by every notebook.
- The **ablation study**: 6 new variants × 3 origins × 3 seeds = 54 trained models (+ A0/A1 reused) → `results/tables/05_ablations.csv`.
- The **final report notebook**: main table, per origin, per segment/department, in-stock view, by-horizon plot, coverage-vs-width plot,
  paired bootstrap significance, LaTeX tables.

## 2. The protocol (draw the timeline)
```
 ... train windows ... | validation 28 d | test 28 d |
                      o-28              o           o+28
```
Three origins: o = 2016-02-28, 2016-03-27, 2016-04-24. Re-fit every model per origin. Deep models: 3 seeds each.
All models are scored on the **same 25,200 (SKU, day) cells** and the **same horizons**.

## 3. Metrics — formulas and why
- **MAE** = mean |y − ŷ|; **RMSE** = √mean (y − ŷ)² — RMSE punishes big misses (spikes) more.
- **WAPE** = Σ|y − ŷ| / Σy. MAPE divides by y → undefined when y = 0, and most of our days are 0. WAPE is volume-weighted.
- **Pinball** L_τ(y, q) = max(τ(y − q), (τ − 1)(y − q)). For τ = 0.9, under-forecasting costs 0.9 per unit, over-forecasting 0.1 —
  so the optimum is the 90% quantile. Averaged over τ = 0.1, 0.5, 0.9. **Strictly proper** (Gneiting & Raftery 2007): the expected loss is
  minimised only by the true quantile → can't be gamed by making intervals wider or narrower.
- **Coverage** of [q10, q90] (target 0.80) and **width** (sharpness). "Maximise sharpness subject to calibration."
- Point forecast = predictive **mean** for all models.

**Why the DL models over-cover (0.91) with the narrowest intervals:** for intermittent SKUs q10 = 0, so every 0 is "inside". Discrete data
inflates coverage. A randomised PIT histogram would be the proper calibration check (future work).

## 4. Final results (memorise the shape, not every digit)
| | WAPE | pinball | cov80 |
|---|---|---|---|
| seasonal naive | 0.757 | 0.514 | 0.84 |
| LightGBM | 0.655 | 0.402 | 0.80 |
| Kalman | 0.636 | 0.416 | 0.87 |
| DeepAR | 0.641 ± 0.005 | 0.395 ± 0.004 | 0.91 |
| 50/50 avg | **0.633** | 0.400 | 0.86 |
| **KF-DeepAR** | 0.641 ± 0.004 | **0.394 ± 0.001** | 0.91 |

- Ranking stable over all 3 origins (DL best at each).
- Largest DL gain on **lumpy** SKUs: 0.461 vs 0.510 (Kalman), −9.6%.
- In-stock days only (5.7% of test cells are flagged stockouts): ranking unchanged.
- **Paired bootstrap** (per-SKU pinball, 5000 resamples): hybrid significantly better than seasonal naive, LightGBM, Kalman, 50/50;
  vs DeepAR CI = [−0.0024, +0.0016] → not significant.

## 5. Ablations (3 origins × 3 seeds, paired)
| variant | Δ pinball | Δ in-stock | runs worse (of 9) |
|---|---|---|---|
| A1 − Kalman channels | +0.1% | +0.2% | 5 |
| A2 − item embedding | +0.4% | +0.3% | 4 |
| A3 − calendar & price | 0.0% | 0.0% | 4 |
| **A4 Gaussian likelihood** | **+4.8%** | **+4.0%** | **9** |
| A5 GRU instead of LSTM | 0.0% | −0.1% | 3 |
| A6 − regularisation | −0.1% | −0.1% | 5 |
| A7 − stockout masking | −0.9% | −0.1% | 2 |

**How to read it:** a component matters only if Δ is bigger than the seed std (~0.3–0.9%) and the sign is consistent across the 9 paired runs.
Only A4 passes. Explanations:
- **A4:** Gaussian puts mass on negative sales, can't model the spike at 0 → worse in every run.
- **A6:** the model is small (27.6k params) and early stopping + random windows already prevent over-fitting (val NLL < train NLL in both).
- **A3:** the 56-day context already shows the weekly cycle; the LSTM infers it from y_{t−1} history.
- **A7:** looks better on all days only because it learns to predict stockout zeros (censored sales); on in-stock days it is equal. For inventory we
  want *demand*, so masking is the right choice even though it doesn't improve the score.
- **Lesson:** a single seed would have misled us — in the first single-seed run A6 and A7 seemed to "help" by ~1–1.5%. With 3 seeds most of that
  disappeared. That is why we report seed std and paired comparisons.

## 6. Likely viva questions
1. *Why 3 seeds, not more?* CPU budget (each model ~45 s; ablations alone are 54 trainings). Enough to show the noise level.
2. *Why bootstrap over SKUs and not over days?* Days of the same SKU are correlated; SKUs are closer to independent units.
3. *What is a paired comparison?* Same SKUs, same origins, same seeds — only the model differs, so shared noise cancels.
4. *Why not MAPE / sMAPE?* Division by zero / instability with intermittent demand.
5. *Diebold–Mariano test?* A classical alternative for comparing forecast errors; we used the bootstrap because it needs fewer assumptions.
6. *Why is the 50/50 average best on WAPE but worse on pinball?* Averaging means reduces variance (Bates & Granger); averaging the quantiles of two
   different distributions does not give a good distribution.
