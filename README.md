# Stock Market Prediction (S&P 500)

Predicts whether the S&P 500 will close higher the next trading day, then tests whether that prediction is actually useful. The focus is honest evaluation: every result is compared against the market's base rate and against buy-and-hold.

## Approach

- **Data:** daily S&P 500 (`^GSPC`) prices from `yfinance`, from 1990 onward. Saved as `sp500_snapshot.csv` (data through 16 Oct 2026) so results are reproducible.
- **Model:** Random Forest. First with basic price and volume features, then with rolling-window features (price relative to its moving averages, recent up-day counts) and a 0.6 confidence threshold.
- **Validation:** walk-forward backtest (train on the past, test on the following year, repeat) instead of a random split, which would leak future information.
- **Comparisons:** base rate, Gradient Boosting and Logistic Regression, a simple trading strategy vs. buy-and-hold, per-year precision, feature importance and a confusion matrix.

## Results

| Approach | Precision | Base rate |
|---|---|---|
| Single split, last 100 days | 0.511 | 0.470 |
| Walk-forward, basic features | 0.538 | 0.545 |
| Walk-forward, rolling features + 0.6 threshold | 0.566 | 0.544 |

- **Model comparison:** Random Forest 0.566, Gradient Boosting 0.560, Logistic Regression 0.551.
- **Trading simulation:** the model strategy grew $1 to about $2.33, against $5.78 for buy-and-hold (no transaction costs included).

## Key takeaways

- A single 100-day split is unreliable. Its score swung between about 0.45 and 0.89 during development, so the walk-forward backtest is the number to trust.
- The final model is only about 2 points above the base rate, which is within the range of chance for 716 "up" calls.
- A small precision edge does not mean profit: the confident-only strategy sits out most of the market and underperforms buy-and-hold.
- Price and volume history alone is a weak predictor. Three very different models land within about 1.5 points of each other.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebook and run all cells. The data file is included, so it loads from `sp500_snapshot.csv` and only downloads from `yfinance` if the file is missing.

**Stack:** Python, Pandas, Scikit-learn, Plotly
