from pathlib import Path
import json
import hashlib
import urllib.request
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

ROOT = Path.cwd()
if (ROOT / 'ftec_midterm' / 'data').exists():
    ROOT = ROOT / 'ftec_midterm'
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'results' / 'direction'
PROCESSED = ROOT / 'data' / 'processed'
for folder in [RAW, OUT, PROCESSED]:
    folder.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'ftec_2016_2023_nasdaq.json': 'https://api.nasdaq.com/api/quote/FTEC/historical?assetclass=etf&limit=5000&fromdate=2013-10-24&todate=2023-12-29',
    'nasdaq_news_scores.parquet': 'https://huggingface.co/datasets/HYL/NASDAQ-News-Multi-LLM-Scores/resolve/main/scores.parquet'
}
for filename, url in SOURCES.items():
    path = RAW / filename
    if not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            path.write_bytes(response.read())
print('Python', platform.python_version(), '| pandas', pd.__version__, '| scikit-learn', sklearn.__version__)


raw_prices = json.loads((RAW / 'ftec_2016_2023_nasdaq.json').read_text())
prices = pd.DataFrame(raw_prices['data']['tradesTable']['rows'])
prices['date'] = pd.to_datetime(prices['date'], format='%m/%d/%Y')
for column in ['open', 'high', 'low', 'close', 'volume']:
    prices[column] = pd.to_numeric(prices[column].str.replace(',', '', regex=False).str.replace('$', '', regex=False), errors='coerce')
prices = prices.sort_values('date').reset_index(drop=True)
assert prices.date.is_unique
assert len(prices) == int(raw_prices['data']['totalRecords'])
assert prices[['open', 'high', 'low', 'close', 'volume']].notna().all().all()
assert (prices.close > 0).all()

news_raw = pd.read_parquet(RAW / 'nasdaq_news_scores.parquet')
assert news_raw['sentiment_nano_title'].notna().all()
assert news_raw['sentiment_nano_title'].between(1, 5).all()
news = news_raw[['Date', 'Article_title', 'Stock_symbol', 'sentiment_nano_title']].copy()
news['news_date'] = pd.to_datetime(news['Date'].str.replace(' UTC', '', regex=False)).dt.normalize()
news['sentiment'] = (news['sentiment_nano_title'].astype(float) - 3) / 2
news['title_key'] = news['Article_title'].str.lower().str.replace(r'[^a-z0-9]+', '', regex=True)
articles = news.groupby(['news_date', 'title_key'], as_index=False).agg(sentiment=('sentiment', 'mean'), ticker_tags=('Stock_symbol', 'nunique'))

print('FTEC:', len(prices), 'sessions,', prices.date.min().date(), 'through', prices.date.max().date())
print('News:', len(news_raw), 'article/ticker rows;', news_raw.Stock_symbol.nunique(), 'symbols')
print('News dates:', news.news_date.min().date(), 'through', news.news_date.max().date())
print('Unique date + normalized-title groups:', len(articles))
prices.to_csv(PROCESSED / 'ftec_long_prices.csv', index=False)


session_dates = prices['date'].to_numpy(dtype='datetime64[ns]')
positions = np.searchsorted(session_dates, articles.news_date.to_numpy(dtype='datetime64[ns]'), side='right')
usable = positions < len(prices)
articles = articles.loc[usable].copy()
positions = positions[usable]
articles['available_date'] = session_dates[positions]
assert (articles.available_date > articles.news_date).all()
daily = articles.groupby('available_date')['sentiment'].agg(['mean', 'count'])
daily.columns = ['sentiment', 'article_count']
frame = prices.merge(daily, left_on='date', right_index=True, how='left')
frame[['sentiment', 'article_count']] = frame[['sentiment', 'article_count']].fillna(0)
frame['return'] = frame.close.pct_change()
frame['target_return'] = frame['return'].shift(-1)
frame['target_date'] = frame.date.shift(-1)
frame['target_up'] = (frame.target_return > 0).astype(int)
price_features = []
sentiment_features = []
for lag in range(5):
    price_name = 'return_lag_' + str(lag)
    sentiment_name = 'sentiment_lag_' + str(lag)
    frame[price_name] = frame['return'].shift(lag)
    frame[sentiment_name] = frame['sentiment'].shift(lag)
    price_features.append(price_name)
    sentiment_features.append(sentiment_name)
frame['momentum_5'] = frame.close.pct_change(5)
frame['volatility_5'] = frame['return'].rolling(5).std()
frame['price_to_mean_5'] = frame.close / frame.close.rolling(5).mean() - 1
frame['sentiment_mean_5'] = frame.sentiment.rolling(5).mean()
frame['sentiment_change'] = frame.sentiment.diff()
frame['sentiment_acceleration'] = frame.sentiment_change.diff()
frame['log_article_count'] = np.log1p(frame.article_count)
price_features += ['momentum_5', 'volatility_5', 'price_to_mean_5']
sentiment_features += ['sentiment_mean_5', 'sentiment_change', 'sentiment_acceleration', 'log_article_count']
all_features = price_features + sentiment_features
frame = frame.dropna(subset=all_features + ['target_return', 'target_date']).copy()
assert (frame.target_date > frame.date).all()
frame.to_csv(PROCESSED / 'direction_features.csv', index=False)
daily.to_csv(PROCESSED / 'daily_broad_sentiment.csv')

print('Usable prediction examples:', len(frame))
print('Prediction dates:', frame.date.min().date(), 'through', frame.date.max().date())
print('Sessions with news:', int((frame.article_count > 0).sum()), '/', len(frame))
print('Dated-title groups used in these examples:', int(frame.article_count.sum()))
print('Features:', len(price_features), 'price;', len(sentiment_features), 'sentiment')


validation = frame[frame.date.dt.year == 2021].copy()
test = frame[frame.date.dt.year >= 2022].copy()
train = frame[(frame.date.dt.year <= 2020) & (frame.target_date < validation.date.min())].copy()
validation = validation[validation.target_date < test.date.min()].copy()
train_valid = pd.concat([train, validation]).sort_values('date')
assert train.target_date.max() < validation.date.min()
assert train_valid.target_date.max() < test.date.min()
splits = []
for name, subset in [('train', train), ('validation', validation), ('test', test)]:
    assert subset.target_up.nunique() == 2
    splits.append(dict(split=name, examples=len(subset), first_date=str(subset.date.min().date()), last_date=str(subset.date.max().date()), up_fraction=float(subset.target_up.mean())))
split_table = pd.DataFrame(splits)
print(split_table.round(4).to_string(index=False))
split_table.to_csv(OUT / 'splits.csv', index=False)

correlation_rows = []
for feature in ['sentiment_lag_0', 'sentiment_mean_5', 'sentiment_change', 'sentiment_acceleration']:
    correlation_rows.append(dict(feature=feature, train_pearson_with_next_return=float(train[feature].corr(train.target_return))))
correlations = pd.DataFrame(correlation_rows)
correlations.to_csv(OUT / 'training_correlations.csv', index=False)
print()
print('Descriptive training-set correlations, not evidence of forecasting or causality:')
print(correlations.round(5).to_string(index=False))


feature_sets = {'Price history': price_features, 'Sentiment history': sentiment_features, 'Both': all_features}

def make_model(family, setting):
    if family == 'Logistic regression':
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(setting), max_iter=3000, random_state=42))
    return RandomForestClassifier(n_estimators=100, max_depth=int(setting), min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=2)

grids = {'Logistic regression': [0.01, 0.1, 1.0, 10.0], 'Random forest': [2, 4]}
choices = []
validation_rows = []
for family, settings in grids.items():
    for input_name, columns in feature_sets.items():
        candidates = []
        for setting in settings:
            model = make_model(family, setting)
            model.fit(train[columns], train.target_up)
            probability = model.predict_proba(validation[columns])[:, 1]
            loss = log_loss(validation.target_up, probability, labels=[0, 1])
            candidates.append((loss, setting))
            validation_rows.append(dict(model=family, inputs=input_name, setting=setting, validation_log_loss=float(loss)))
        loss, setting = min(candidates, key=lambda item: item[0])
        choices.append((family, input_name, setting))

def evaluate(name, inputs, probabilities, setting=''):
    predicted = (np.asarray(probabilities) >= 0.5).astype(int)
    return dict(model=name, inputs=inputs, setting=str(setting), test_examples=len(test),
                accuracy=float(accuracy_score(test.target_up, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(test.target_up, predicted)),
                log_loss=float(log_loss(test.target_up, probabilities, labels=[0, 1])))

baseline_probability = float(train_valid.target_up.mean())
test_rows = [evaluate('Training frequency baseline', 'No features', np.full(len(test), baseline_probability))]
predictions = test[['date', 'target_date', 'target_return', 'target_up']].reset_index(drop=True).copy()
predictions['baseline_up_probability'] = baseline_probability
fitted_models = {}
for family, input_name, setting in choices:
    columns = feature_sets[input_name]
    model = make_model(family, setting)
    model.fit(train_valid[columns], train_valid.target_up)
    probability = model.predict_proba(test[columns])[:, 1]
    test_rows.append(evaluate(family, input_name, probability, setting))
    predictions[family + ' | ' + input_name] = probability
    fitted_models[(family, input_name)] = model
results = pd.DataFrame(test_rows)
pd.DataFrame(validation_rows).to_csv(OUT / 'validation_grid.csv', index=False)
results.to_csv(OUT / 'test_results.csv', index=False)
predictions.to_csv(OUT / 'test_predictions.csv', index=False)
print(results.round(5).to_string(index=False))
print()
print('Test up-days:', int(test.target_up.sum()), '/', len(test))


labels = results['model'] + ' — ' + results['inputs']
fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
ax.barh(labels, results.accuracy * 100, color=['#858585'] + ['#286c8e'] * 3 + ['#578a64'] * 3)
ax.axvline(results.iloc[0].accuracy * 100, color='#333333', linestyle='--', label='Training-frequency baseline')
ax.set_xlim(0, 100)
ax.set_xlabel('Test accuracy (%)')
ax.set_title('FTEC next-day direction: 2022–2023 test period')
ax.invert_yaxis()
ax.legend(loc='lower right')
fig.savefig(OUT / 'direction_accuracy.png', dpi=160)
plt.show()

summary = {'raw_price_rows': len(prices), 'raw_news_rows': len(news_raw), 'unique_dated_titles': int(news[['news_date', 'title_key']].drop_duplicates().shape[0]),
           'usable_examples': len(frame), 'covered_sessions': int((frame.article_count > 0).sum()),
           'splits': split_table.to_dict(orient='records'), 'test_results': results.to_dict(orient='records'),
           'training_correlations': correlations.to_dict(orient='records')}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
manifest = []
for filename, url in SOURCES.items():
    path = RAW / filename
    manifest.append(dict(file='data/raw/' + filename, source_url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size))
(ROOT / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_score(name, prediction_pp):
    actual_pp = test.target_return.to_numpy() * 100
    return dict(model=name, test_examples=len(test),
                rmse_pp=float(np.sqrt(mean_squared_error(actual_pp, prediction_pp))),
                mae_pp=float(mean_absolute_error(actual_pp, prediction_pp)),
                r2=float(r2_score(actual_pp, prediction_pp)),
                direction_accuracy=float(accuracy_score(test.target_up, np.asarray(prediction_pp) > 0)))

regression_predictions = test[['date', 'target_date', 'target_return', 'target_up']].reset_index(drop=True).copy()
regression_rows = []
for name, values in [('No change', np.zeros(len(test))), ('Training mean return', np.full(len(test), train_valid.target_return.mean() * 100))]:
    regression_rows.append(regression_score(name, values))
    regression_predictions[name] = values

regression_grids = {'Ridge': [0.1, 1.0, 10.0, 100.0], 'Lasso': [0.001, 0.01, 0.1, 1.0]}
regression_constructors = {'Ridge': Ridge, 'Lasso': Lasso}
regression_settings = {}
regression_validation = []
for name, grid in regression_grids.items():
    candidates = []
    for alpha in grid:
        model = make_pipeline(StandardScaler(), regression_constructors[name](alpha=alpha, max_iter=100000))
        model.fit(train[all_features], train.target_return * 100)
        prediction = model.predict(validation[all_features])
        rmse = float(np.sqrt(mean_squared_error(validation.target_return * 100, prediction)))
        candidates.append((rmse, alpha))
        regression_validation.append(dict(model=name, alpha=alpha, validation_rmse_pp=rmse))
    regression_settings[name] = min(candidates, key=lambda value: value[0])[1]

regression_models = {'Ordinary linear regression': make_pipeline(StandardScaler(), LinearRegression())}
for name, alpha in regression_settings.items():
    regression_models[name] = make_pipeline(StandardScaler(), regression_constructors[name](alpha=alpha, max_iter=100000))
for name, model in regression_models.items():
    model.fit(train_valid[all_features], train_valid.target_return * 100)
    prediction = model.predict(test[all_features])
    regression_predictions[name] = prediction
    regression_rows.append(regression_score(name, prediction))
pd.DataFrame(regression_validation).to_csv(OUT / 'regression_validation.csv', index=False)
lasso_coefficients = regression_models['Lasso'][-1].coef_
pd.DataFrame({'feature': all_features, 'scaled_coefficient_pp': lasso_coefficients}).to_csv(OUT / 'lasso_coefficients.csv', index=False)
print('Selected return-regression penalties:', regression_settings)
print('Lasso nonzero feature coefficients:', int(np.count_nonzero(lasso_coefficients)), '/', len(all_features))
print(pd.DataFrame(regression_rows).round(5).to_string(index=False))


from prophet import Prophet
import prophet
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
print('Prophet version:', prophet.__version__)

prophet_inputs = {'Prophet calendar only': [], 'Prophet + price': price_features, 'Prophet + price + sentiment': all_features}
prophet_models = {}
for name, columns in prophet_inputs.items():
    history = train_valid[columns].reset_index(drop=True).copy()
    history['ds'] = train_valid.target_date.to_numpy()
    history['y'] = train_valid.target_return.to_numpy() * 100
    future = test[columns].reset_index(drop=True).copy()
    future['ds'] = test.target_date.to_numpy()
    assert history.ds.max() < test.date.min()
    assert (test.date.to_numpy() < future.ds.to_numpy()).all()
    model = Prophet(growth='linear', n_changepoints=10, changepoint_prior_scale=0.05,
                    yearly_seasonality=5, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_prior_scale=1.0, uncertainty_samples=0)
    for column in columns:
        model.add_regressor(column, prior_scale=0.1, standardize=True)
    model.fit(history, seed=42)
    forecast = model.predict(future)
    prediction = forecast['yhat'].to_numpy()
    assert np.isfinite(prediction).all()
    regression_predictions[name] = prediction
    regression_rows.append(regression_score(name, prediction))
    prophet_models[name] = model

regression_results = pd.DataFrame(regression_rows)
regression_results.to_csv(OUT / 'return_forecast_results.csv', index=False)
regression_predictions.to_csv(OUT / 'return_forecast_predictions.csv', index=False)
summary['regression_settings'] = regression_settings
summary['lasso_nonzero_coefficients'] = int(np.count_nonzero(lasso_coefficients))
summary['return_forecast_results'] = regression_results.to_dict(orient='records')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(regression_results.round(5).to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
axes[0].barh(regression_results.model, regression_results.rmse_pp, color='#286c8e')
axes[0].set_xlabel('Return RMSE (percentage points)')
axes[0].set_title('Same 500 test sessions: return error')
axes[1].barh(regression_results.model, regression_results.direction_accuracy * 100, color='#578a64')
axes[1].set_xlabel('Direction accuracy (%)')
axes[1].set_xlim(0, 100)
axes[1].set_title('Sign of the forecast return')
for ax in axes:
    ax.invert_yaxis()
fig.savefig(OUT / 'prophet_regression_comparison.png', dpi=160)
plt.show()
