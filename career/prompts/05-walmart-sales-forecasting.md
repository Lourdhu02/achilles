# Prompt: advanced Walmart sales-forecasting project

Paste everything below the line into a new Claude Code session with a new empty repo attached (e.g. `Lourdhu02/walmart-forecasting`).

---

You are building a **portfolio-grade, research-quality** sales-forecasting project on public Walmart data. It should show senior data-science judgment: correct validation, strong baselines, honest comparisons, uncertainty and business framing. Flashy models alone don't count.

## Data

- **Primary: M5 Forecasting (Walmart):** 30,490 item-store series across 3 states, 10 stores and 3 categories, with daily sales, prices and calendar/SNAP events. Metrics: **WRMSSE** (accuracy) and **WSPL** (uncertainty), exactly as defined in the M5 competition guidelines. Get it from the Kaggle "M5 Forecasting - Accuracy" and "M5 Forecasting - Uncertainty" competitions; document the download steps, and don't commit the data.
- **Secondary, optional:** "Walmart Recruiting - Store Sales Forecasting" (45 stores, weekly, holiday-weighted WMAE), as a second benchmark.

## What to do

1. **Setup:** a `uv` project, `make data`, `make train`, `make eval`, pinned dependencies, seeds and a config per experiment. Use Polars or DuckDB for feature engineering at this scale.
2. **Validation:** time-based backtesting with at least 3 rolling origins of 28 days each, mirroring M5's horizon. Never leak future data (prices, events or rolling features). Write a leakage test.
3. **Baselines first:** seasonal naive, ETS and Croston/TSB for intermittent series. Every later model must beat these on WRMSSE, or the README says it didn't.
4. **Main models:**
   - a global LightGBM with Tweedie loss on lag, rolling, price, calendar and SNAP features (recursive vs direct strategies compared);
   - a deep model (N-BEATS, N-HiTS or TFT via `neuralforecast`);
   - one foundation-model zero-shot comparison (e.g. Chronos or TimesFM);
   - an ensemble.
5. **Hierarchy:** forecast all 12 aggregation levels, and reconcile with MinT vs bottom-up (`hierarchicalforecast`). Report WRMSSE per level.
6. **Uncertainty:** quantile forecasts (9 quantiles, as in M5), scored with WSPL, plus conformal calibration. Report coverage.
7. **Analysis:** error by category, store and intermittency class; SHAP for LightGBM; the effect of prices and SNAP events.
8. **Business layer:** turn forecasts into a newsvendor inventory decision for one category. Show the cost of over- vs under-stocking under each model; a better WRMSSE is only worth what it saves.
9. **Deliverables:**
   - a README with a results table (WRMSSE and WSPL per model, with backtest mean ± std, compute time and hardware);
   - a 2-page technical report;
   - a small Streamlit or Evidence dashboard;
   - CI running unit tests and a tiny-sample end-to-end pipeline.

## Rules

- No invented or projected numbers. Every metric comes from the backtest scripts with a committed config. Compare with the published M5 leaderboard scores for context, and never claim a rank you didn't get.
- Compute: RTX 5060 laptop GPU (8 GB) and CPU. Keep training within that and state the run times.
- Commit as `Lourdhu Raju <b.lourdhuraju1234@gmail.com>`, with no Co-Authored-By or session trailers. Small PRs, CI green.

## Done when

The backtested results table covers the baselines through the ensemble, with hierarchy, uncertainty and the business layer. The pipeline reproduces from `make all` on a clean clone. Then give me two resume bullets with measured WRMSSE/WSPL and the inventory-cost result.
