# Predictive Analytics of Indian Stock Prices Using Linear Regression
An end-to-end econometric and data engineering case study modeling daily adjusted stock prices of **Reliance Industries (RELIANCE.NS)** using Ordinary Least Squares (OLS) estimation.

##  Project Overview
This repository contains a fully reproducible, rigorous data pipeline designed to evaluate whether a classical linear regression architecture can outperform a naive no-change benchmark in forecasting the next trading session's adjusted closing price. Rather than treating regression as a black box, this workflow explicitly maps structural data auditing, feature engineering, chronological train/validation/test splits, and extensive statistical hypothesis diagnostics to uncover structural limitations.

*   **Development Timeline:** October 2024 – January 2025
*   **Primary Technologies:** Python 3.13, Statsmodels, Scikit-Learn, Pandas, NumPy, Seaborn
*   **Target Asset:** Reliance Industries (RELIANCE.NS) traded on the National Stock Exchange (NSE) of India

---

## 📊 Empirical Results & Statistical Summary
When executed against the historical daily dataset spanning January 1, 2015, to October 12, 2026, the locked holdout test yielded the following final metrics:

| Metric | Selected OLS Model | Naive No-Change Benchmark |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 13.418 INR | **13.358 INR** |
| **Root Mean Squared Error (RMSE)**| 17.885 INR | **17.775 INR** |
| **Mean Absolute Percentage Error (MAPE)** | 0.988% | **0.983%** |
| **Out-of-Sample R² Score** | 0.9693 | **0.9697** |
| **Directional Accuracy (%)** | 49.13% | 0.52% |
| **RMSE Out-of-Sample Predictive Skill**| **-0.62%** | 0.00% (Baseline) |

###  Crucial Analytical Insight
While the model exhibits an exceptionally high price-level R² (96.93%), it fails to outperform the naive no-change baseline out-of-sample. In fact, its predictive skill is **negative (-0.62%)**. This strongly demonstrates the structural danger of using simple regression on non-stationary, persistent financial time-series. **High price-level R² scores are artifacts of price persistence and must not be misinterpreted as predictive advantage.**

---

##  Project Structure & Data Pipeline
The repository features an auditable, end-to-end framework:
```text
├── predictive_stock_analytics.ipynb  # Comprehensive Jupyter Notebook with all outputs
├── requirement_packages.txt          # Verified dependency version constraints
├── data_snapshot.csv                 # Embedded offline reproducible dataset (INR values)
└── README.md                         # This architecture and methodology manual
```

### 1. Data Ingestion & Integrity Audit
*   **Source:** Yahoo Finance daily OHLCV via the `yfinance` API.
*   **Data Footprint:** 2,908 initial rows ingested.
*   **Audit Exceptions:** Checked for missing variables, duplicate timestamps, and logical violations (e.g., Low > High or Volume < 0).
*   **Cleaning Actions:** Identified and removed 1 invalid row containing null entries (`2026-10-12`) without forward-filling, preventing data manufacturing.
*   **Final Baseline Dataset:** 2,907 clean, logically validated rows (January 1, 2015, to October 9, 2026).
*   **Price Levels:** Cleaned prices range from a minimum of 172.69 INR to a maximum of 1,584.97 INR, with a mean of 823.32 INR.

### 2. Chronological Validation Architecture
To protect against time-series data leakage, data rows are partitioned strictly chronologically (avoiding randomized shuffles):
*   **Training Set (60%):** 1,730 observations (January 30, 2015, to February 2, 2022).
*   **Validation Set (20%):** 576 observations (February 4, 2022, to June 7, 2024).
*   **Locked Holdout Test Set (20%):** 578 observations (June 11, 2024, to October 9, 2026).

---

##  Feature Engineering Matrix
At the close of day $t$, all inputs use information dated $t$ or earlier to model day $t+1$. Features are cross-standardized on training parameters to ensure stable conditioning.

### Predictor Set
1.  `return_1`: Daily simple return, calculated as $\frac{\text{Close}_t}{\text{Close}_{t-1}} - 1$.
2.  `return_lag1`: 1-session lagged simple return.
3.  `return_lag2`: 2-session lagged simple return.
4.  `volatility_20`: Rolling 20-day standard deviation of daily simple returns.
5.  `range_pct`: Intraday high-low tracking span, computed as $\frac{\text{High}_t - \text{Low}_t}{\text{Close}_t}$.
6.  `volume_relative`: Logarithmic tracking of volume spikes relative to its 20-day mean: $\ln(1 + \text{Volume}_t) - \text{rolling\_mean}(\ln(1 + \text{Volume}))_{20}$.

### Candidate Models specified in advance
*   **OLS Price (Full Feature Set):** Regresses $\text{Close}_{t+1}$ on $\text{Close}_t$ plus all engineered returns, volatilities, ranges, and volume parameters.
*   **OLS Price (Reduced Feature Set):** Regresses $\text{Close}_{t+1}$ on $\text{Close}_t$, excluding momentum/moving average parameters to mitigate collinearity.
*   **OLS Return (Reduced Feature Set):** Regresses daily simple return $R_{t+1}$ on the reduced features. Predictions are mapped back to price levels via:
$$\widehat{\text{Close}}_{t+1} = \text{Close}_t \times (1 + \hat{R}_{t+1})$$

---

##  Econometric Diagnostics (Training Model)
All diagnostics are executed strictly on the training partition to isolate statistical anomalies prior to holdout evaluation.

### 1. Multicollinearity Assessment (VIF)
Variance Inflation Factors were computed to monitor inflation of coefficient variances:
*   `return_1`: 1.024 | `return_lag1`: 1.029 | `return_lag2`: 1.018
*   `volatility_20`: 1.527 | `range_pct`: 1.932 | `volume_relative`: 1.413
*   **Verdict:** All VIF metrics are safely under 2.0, confirming the absence of severe linear collocations among predictors.

### 2. Core Hypothesis Diagnostics ($\alpha = 0.05$)
*   **Homoskedasticity (Breusch–Pagan Test):** Yielded a $p$-value of $7.358 \times 10^{-47}$. Null of constant variance rejected; severe heteroskedasticity detected, requiring robust error corrections.
*   **Residual Normality (Jarque–Bera Test):** Yielded a $p$-value of $0.000$. Null of normal error structure rejected, confirming fat-tailed (leptokurtic) time-series distributions.
*   **Serial Independence (Ljung–Box lag 10 & Breusch–Godfrey):** Ljung–Box $p$-value was $0.00544$; Breusch–Godfrey $p$-value was $0.00729$. Null of zero serial dependence rejected; residual autocorrelation verified.
*   **Functional Specification (Ramsey RESET Test):** Yielded a $p$-value of $0.649$. Failed to reject the null hypothesis, indicating no severe structural omission of higher-order fitted terms.
*   **Descriptive Statistics:** Durbin–Watson statistic sits close to the baseline at $1.999$, and the scaled design condition number is exceptionally well-conditioned at $2.398$.

---

##  Replicating the Workflow

### 1. Installation of Packages
Install the exact verified dependency version ranges utilized in this environment:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels yfinance
```

### 2. Execution
Run the complete pipeline, which parses the embedded CSV data layer, fits the cross-validation folds, and outputs the diagnostic visual graphs:
```bash
jupyter notebook predictive_stock_analytics.ipynb
```

##  Analytical Caveats & Constraints
This case study is strictly an educational research demonstration and does not constitute investment advice, financial planning, or trading recommendations. Real-world trading applications require historical point-in-time exchange feeds to avoid the lookahead biases inherent in retrospectively adjusted corporate-action data layers. Because classical OLS assumptions (homoskedasticity and serial independence) failed, the inference engine applies Newey–West Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors (max lag = 5) to generate valid $p$-values.
