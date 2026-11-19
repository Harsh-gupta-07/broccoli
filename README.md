# Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models — Phase 2 (Deep Learning)

Course: **Deep Learning (CSA332)**, B.Tech CS&AI / CS&DS, Semester 5.
Team: **Saumya · Harshvardhan · Shivansh · Krishiv · Harsha** (roles in [CONTRIBUTIONS.md](CONTRIBUTIONS.md)).

We forecast 28 days of daily unit sales for 300 Walmart SKUs (M5) and compare four kinds of model under one
rolling-origin protocol:

* a **local Kalman filter** per SKU (Phase 1),
* a **tabular ML baseline** (LightGBM with lag features, Phase 1),
* a **global deep model**, DeepAR (LSTM + Negative Binomial likelihood), written from scratch in PyTorch (Phase 2),
* a **hybrid**, *KF-DeepAR*, where the Kalman filter's causal predictions and uncertainty are fed to DeepAR as inputs (Phase 2).

Every model produces **quantiles** (q10, q50, q90), not just point forecasts, and is scored with MAE, RMSE, WAPE,
pinball loss and 80%-interval coverage on the same horizons.

## Results

Rolling-origin backtest: 300 SKUs, 3 origins, 28-day horizon, 25,200 test cells. Deep models: mean ± std over 3 seeds.

| Model | MAE | RMSE | WAPE | Pinball ↓ | Cov. 80% | Width 80% |
|---|---|---|---|---|---|---|
| Seasonal naive | 1.577 | 3.342 | 0.757 | 0.514 | 0.84 | 3.86 |
| LightGBM (Phase 1) | 1.365 | 2.666 | 0.655 | 0.402 | 0.80 | 3.99 |
| Kalman filter (Phase 1) | 1.325 | 2.571 | 0.636 | 0.416 | 0.87 | 4.03 |
| DeepAR (global DL) | 1.336 ± 0.010 | 2.579 ± 0.027 | 0.641 ± 0.005 | 0.395 ± 0.004 | 0.91 | 3.68 |
| 50/50 avg Kalman + DeepAR | **1.320** | **2.543** | **0.633** | 0.400 ± 0.002 | 0.86 | 3.86 |
| **KF-DeepAR hybrid (ours)** | 1.336 ± 0.007 | 2.573 ± 0.015 | 0.641 ± 0.004 | **0.394 ± 0.001** | 0.91 | **3.67** |

Key findings:
* The global DL model improves the pinball loss by 1.8% over LightGBM and 5.1% over the Kalman filter. The largest gain is on lumpy SKUs (−9.6% vs Kalman).
* **KF-DeepAR** has the lowest pinball loss. It is significantly better (paired bootstrap over SKUs) than every Phase-1 model and than a fixed 50/50 average.
  It ties plain DeepAR overall (95% CI contains 0) and is the **best model for cold-start SKUs** (30 unseen SKUs with 28 days of history:
  pinball 0.552 vs 0.556 DeepAR and 0.571 Kalman).
* Ablations (3 origins × 3 seeds): the Negative Binomial likelihood is the one component that clearly matters (Gaussian: +4.8%, worse in 9/9 runs).
  The other choices are within seed noise.
* The Kalman filter is still the strongest *point* forecaster among single models. The 50/50 average has the best WAPE but worse quantiles.

Full numbers: `results/tables/`. Report: [`report/main.pdf`](report/main.pdf). Slides: [`presentation/slides.pdf`](presentation/slides.pdf).

## Repository layout

```
notebooks/
  00_common.ipynb               shared library (loaded with %run) — sections owned by different members
  01_data_pipeline.ipynb        Saumya       data cleaning, stockout flags, features, DL windows
  02_baselines_kalman.ipynb     Harshvardhan seasonal naive, LightGBM, Kalman filter, "the gap"
  03_deepar_global.ipynb        Shivansh     DeepAR: theory, training, 3 origins x 3 seeds
  04_hybrid_kalman_deepar.ipynb Krishiv      KF-DeepAR hybrid, 50/50 control, cold-start test
  05_ablation_studies.ipynb     Harsha       7 ablations of the final model
  06_final_evaluation.ipynb     Harsha       rolling-origin report, significance tests, LaTeX tables
data/raw/                       M5 csv files (not committed, see data/raw/README.md)
data/processed/                 panel.npz, series_meta.csv, kalman_states.npz (regenerated)
results/forecasts/              every model's test forecasts (parquet)
results/tables/                 csv + LaTeX tables used in the report
results/models/                 trained PyTorch weights (state_dict)
figures/                        all plots (PNG)
report/                         IEEE LaTeX report (main.tex, references.bib, main.pdf)
presentation/                   Beamer slides (slides.tex, slides.pdf)
docs/viva/                      one viva preparation guide per member
```

## How to run

```bash
pip3 install -r requirements.txt
```

Put the three M5 csv files in `data/raw/` (see [data/raw/README.md](data/raw/README.md)), then run the notebooks **in order**
(01 → 06). Each notebook loads the shared code with `%run ./00_common.ipynb`.

```bash
cd notebooks
for nb in 01_data_pipeline 02_baselines_kalman 03_deepar_global 04_hybrid_kalman_deepar 05_ablation_studies 06_final_evaluation; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
```

Runtime on a 4-core CPU (no GPU needed): about 1 min for 01–02, ~10 min each for 03 and 04, ~20 min for 05, 1 min for 06.
On Google Colab, upload the repo, put the data in `data/raw/`, and run the notebooks from the `notebooks/` folder.

Build the report and slides (needs a LaTeX installation with IEEEtran and beamer):

```bash
cd report && latexmk -pdf main.tex
```

```bash
cd presentation && latexmk -pdf slides.tex
```

## Troubleshooting

**"Kernel died" in `02_baselines_kalman` (macOS).** PyTorch ships its own OpenMP runtime and LightGBM uses Homebrew's `libomp`,
so both get loaded in one process. `00_common.ipynb` sets `KMP_DUPLICATE_LIB_OK=TRUE`, which stops the `OMP: Error #15` abort,
but multi-threaded LightGBM still segfaults once torch is imported (in `train` and in `predict`). Notebook 02 therefore runs
LightGBM with `num_threads=1` on macOS (`LGB_THREADS`); the full run of notebook 02 takes about a minute. To check your setup outside Jupyter:

```bash
python -c "import torch, lightgbm as lgb, numpy as np; b = lgb.train({'verbose': -1, 'num_threads': 1}, lgb.Dataset(np.random.rand(5000, 5), np.random.rand(5000)), 5); b.predict(np.random.rand(100, 5), num_threads=1); print('ok')"
```

LightGBM on macOS also needs `brew install libomp`. A Python version that is 1–2 years old (3.11 or 3.12) is the safest choice for PyTorch and LightGBM wheels.

## Protocol in one paragraph
300 SKUs (store CA_1; FOODS_3, HOUSEHOLD_1, HOBBIES_1; 100 random each), last 1000 days. Three rolling origins
(2016-02-28, 2016-03-27, 2016-04-24), horizon 28 days. For each origin: training data ends 28 days before the origin,
the 28 days before the origin are the validation range (early stopping), the next 28 days are the test range. Every model is re-fitted
per origin. Deep models: 3 seeds. Stockouts are flagged causally and **marked, not imputed**: excluded from training losses and the
Kalman update, and reported separately in the evaluation.

## Git workflow (so each member's contribution is visible)
Each member commits their own notebook, their section of `00_common.ipynb`, their report section and viva guide from their own
GitHub account, e.g.

```bash
git add notebooks/02_baselines_kalman.ipynb docs/viva/Harshvardhan_kalman_baselines.md
git commit -m "Kalman filter baseline and gap analysis"
```
