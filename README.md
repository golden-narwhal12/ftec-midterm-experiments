# FTEC midterm direction experiment

Download and extract `FTEC_midterm_experiments.zip`, then open `ftec_midterm/FTEC_midterm_experiments.ipynb`. The ZIP contains the full folder structure, including the raw and processed datasets. The notebook and script are also available separately at the repository root for reading. It includes executed outputs, source citations, data preparation, chronological validation, classification and Prophet/regression comparisons, and a figure.

Install dependencies with `python -m pip install -r requirements.txt`. Run all notebook cells in Jupyter, or run `python experiments.py` from this folder. The two downloaded source files are included. If they are absent, the notebook/script attempts to download them using the source URLs in its first cell. To use Colab later, upload the notebook and extract the complete folder, or let its download fallback obtain the sources. No paid API key or GPU is needed. Re-downloaded sources may change; bundled snapshots and hashes establish this run.

## Question and methods

Predict whether FTEC's next trading-day close is higher than its current close. Compare regularized logistic regression and random forests with price history, sentiment history, and both. Keep all topics and ticker associations in the selected news source. Older news dates are assigned to the next trading session, never to the same-date close. Training runs through 2020, validation is 2021, and testing is 2022–2023. Purge labels that reach the next split's first prediction date. Settings are selected by validation log loss only.

## Data and citations

- Nasdaq FTEC history: https://www.nasdaq.com/market-activity/etf/ftec/historical
- Price download: https://api.nasdaq.com/api/quote/FTEC/historical?assetclass=etf&limit=5000&fromdate=2013-10-24&todate=2023-12-29
- The price request asked for 2013 onward, but actually returned 1,793 sessions from November 14, 2016 through December 29, 2023. No earlier history is claimed.
- HYL scored news: https://huggingface.co/datasets/HYL/NASDAQ-News-Multi-LLM-Scores
- Scored download: https://huggingface.co/datasets/HYL/NASDAQ-News-Multi-LLM-Scores/resolve/main/scores.parquet
- Underlying FNSPID: Zihan Dong, Xinyu Fan, and Zhiyuan Peng (2024), FNSPID: A Comprehensive Financial News Dataset in Time Series, https://arxiv.org/abs/2402.06698 and https://huggingface.co/datasets/Zihan1004/FNSPID
- HYL's original README is included. It states the scored release is CC BY-NC 4.0. Attribute both HYL and the underlying FNSPID/FinRL_DeepSeek lineage. Source-provider terms remain applicable to their respective data.

The actual news file has 127,176 article/ticker rows for 89 symbols, covering 2009–early 2024. It is not FTEC-specific or tech-only. Use the fixed sentiment_nano_title column (1–5), scaled to -1–1; aggregate repeated dated titles once and average their scores across ticker tags. In our feature window this yields 80,129 dated-title groups over 1,787 prediction dates. The package contains source snapshots, processed daily features, individual test predictions, validation settings, and summary results.

## Results

Training 1,034 examples, validation 251, test 500; two additional boundary examples are excluded. Classifier input sets are price history, sentiment history, and both. Logistic regression accuracy was 52.6%, 50.8%, 52.0%; random forest 51.4%, 51.4%, 52.8%. The training-frequency baseline always predicted up and scored 52.0%.

To match the team's draft, we also implemented ordinary linear regression, Ridge, Lasso, and Prophet for next-day returns, on the same split. Prophet calendar-only uses historical FTEC returns to learn trend and seasonality, but no extra predictors. Its direction accuracy was 54.4% and return RMSE 1.7322 percentage points. Prophet + price was 51.2% / 1.7915; Prophet + price + sentiment was 49.6% / 1.7925. No change had RMSE 1.7254, so Prophet did not improve the return-error baseline. Ordinary/Ridge/Lasso RMSE was 1.7862 / 1.7844 / 1.7551. Lasso kept 3 of 17 standardized coefficients. Direction is the sign of each predicted return.

Prophet regressors at a target date contain only histories from the previous prediction date. This is a sequence of conditional next-day forecasts computed in a batch, not an unconditional forecast of two years with future sentiment supplied. See the notebook for exact fixed Prophet settings and additional-regressor timing.

## Limits and next steps

Sentiment is a retrospective language-model estimate, not human ground truth or verified point-in-time data. Most article times are date-only, and publication-versus-update dates have not been audited. A conservative next-session lag mitigates same-day leakage without proving historical availability. Raw FTEC price returns exclude separately added dividends. One historical split and small score differences do not establish causality, significance, live predictive usefulness, or profitability.

Further Prophet tuning, topic filtering, a second archive, and expanding-window tests are later extensions. The shared report has not been edited. Team review remains pending. The test period was inspected repeatedly during development, so these results are exploratory rather than a pristine final evaluation. A future final evaluation should reserve a new period.

Prophet documentation: https://meta.github.io/prophet/docs/quick_start.html and https://meta.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
