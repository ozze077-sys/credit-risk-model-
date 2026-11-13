# Credit Risk Analytics & Default Prediction Dashboard

An interactive dashboard served by a **plain Python web server** (one HTML page + JSON API; Dash and
Streamlit versions are also included) that analyses historical credit-card data, trains a
logistic-regression model, estimates each customer's **probability of defaulting next month**,
groups customers into risk segments, and explains the results.

Everything runs locally. No Power BI, cloud services or API keys.

**Live online version:** https://ozze077-sys.github.io/credit-risk-model-/

---

## Online version (GitHub Pages)

GitHub Pages can only host static files, so it cannot run `server.py`. Instead,
`build_static_site.py` writes one self-contained `index.html` at the project root:

- the same page as `web/index.html` (same look, pages and charts);
- the customer table, the model's coefficients and every fixed chart, computed by `server.py`'s own functions and embedded as JSON;
- `web/static_api.js`, which answers the page's `/api/...` requests **in the browser** with the same formulas,
  so the threshold sliders, filters, customer lookup, new-customer estimates, warning-level slider and CSV downloads all still work.

Retraining and "Save all predictions to outputs/" need Python, so they are hidden online.

**To rebuild and publish** after the data or model changes:

```powershell
python build_static_site.py     # writes index.html
git add index.html
git commit -m "Rebuild online dashboard"
git push                        # .github/workflows/pages.yml deploys it automatically
```

One-time setup: in the GitHub repo, **Settings → Pages → Source** must be set to **GitHub Actions**.

---

## 1. Quick start (Windows PowerShell, inside VS Code)

Open the project folder in VS Code, then open a terminal (**Terminal → New Terminal**, PowerShell):

```powershell
# 1. Create a virtual environment (an isolated set of Python packages for this project)
python -m venv .venv

# 2. Activate it. If PowerShell blocks scripts, run this once first:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1

# 3. Install the packages
python -m pip install -r requirements.txt

# 4. Start the dashboard, then open http://127.0.0.1:8000 in your browser
python server.py
```

Stop the dashboard with **Ctrl + C** in the terminal.

The first start reads the Excel file and, if no valid model exists yet, trains one (about 30 seconds).
After that the saved model is reused.

### Three dashboard versions

All versions use the same data, model and prediction code and show the same six pages.

| | Plain Python server (main) | Plotly Dash | Streamlit |
|---|---|---|---|
| Start | `python server.py` | `python dash_app.py` | `python -m streamlit run app.py` |
| Address | http://127.0.0.1:8000 | http://127.0.0.1:8050 | http://localhost:8501 |
| Files | `server.py`, `web/index.html`, `charts.py` | `dash_app.py`, `charts.py`, `assets/style.css` | `app.py`, `.streamlit/config.toml` |

### How the plain-server version works

```
Browser (web/index.html)                         server.py (Python standard library http.server)
 ├─ CSS: navy header, sidebar, KPI cards          ├─ loads dataset + model once at start-up
 ├─ JavaScript: fetch('/api/...') ─── JSON ──────>├─ /api/... endpoints build KPIs, tables and
 └─ Plotly.js draws the chart JSON <── JSON ───────┘   Plotly chart JSON (charts.py)
```

- `GET /` returns the single HTML page. Plotly.js is served from the installed `plotly` package at
  `/static/plotly.min.js`, so no internet connection is needed.
- Pages are switched with the URL hash (`#/overview`, `#/exploration`, ...), without reloading the page.
- Errors come back as JSON (`{"error": "..."}`) with status 400/404/500, and the page shows them as messages.
- The thresholds you choose are remembered in your browser.

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Dataset and model info, or the start-up error |
| `GET /api/overview?medium=30&high=50` | KPI cards and the five Overview charts |
| `GET /api/exploration` | KPIs, default-rate charts, limit charts, correlation heatmap |
| `GET /api/exploration/bill-payment?stat=Mean` | Bill and payment charts (Mean or Median) |
| `GET /api/exploration/distribution?var=LIMIT_BAL` | Distribution chart and caption |
| `GET /api/segmentation?medium=30&high=50` | Segment cards, histogram, summary table, filter options |
| `GET /api/segmentation/customers?segment=&age=&limit=&split=&top=25` | Filtered count and highest-risk table |
| `GET /api/segmentation/export.csv?...` | The filtered predictions as CSV |
| `POST /api/segmentation/save` `{"medium":30,"high":50}` | Writes `outputs/customer_predictions.csv` and `test_predictions.csv` |
| `GET /api/customer?id=1` | One customer: details, probability, segment, actual outcome, explanation |
| `GET /api/form-defaults` | Fields, medians and code options for the new-customer form |
| `POST /api/predict` `{"values":{...}}` | Probability for a new customer (or a list of validation errors) |
| `GET /api/performance` | Test-set metrics, explanations, report, all performance charts |
| `GET /api/performance/threshold?t=0.3` | Metrics at another decision threshold |
| `GET /api/quality` | Data-quality report |
| `GET /api/download/metrics.json`, `test_predictions.csv`, `data_quality.csv` | File downloads |
| `POST /api/retrain` `{"balanced":false}` | Retrains the model |

How the Dash version works:
- `app.layout` describes the page as a tree of components.
- Each `@callback` function runs when one of its *Inputs* changes (a slider, dropdown, button or the URL) and
  returns new content for its *Outputs*.
- The data and model are loaded once when `dash_app.py` starts.

---

## 2. Project files

| File / folder | What it does |
|---|---|
| `default of credit card clients.xls` | The original UCI dataset. It is **only read, never modified**. |
| `data_loader.py` | Finds the dataset automatically, detects the real header row, standardises column names, validates the target, builds the data-quality report. |
| `data_analysis.py` | Historical default rates by group, outcome comparisons, correlations (used by the Data Exploration page). |
| `model_training.py` | Train/test split, preprocessing + logistic-regression pipeline, evaluation, feature interpretation, saving the model. |
| `prediction.py` | Loads the model, predicts probabilities, assigns risk segments, explains individual predictions, validates form input, exports CSVs. |
| `server.py` | The **plain Python web server**: serves the page and the JSON API endpoints. |
| `web/index.html` | The single HTML page (layout, styles and JavaScript) for `server.py`. |
| `charts.py` | All Plotly chart builders (used by `server.py` and `dash_app.py`). |
| `dash_app.py` | The Plotly Dash dashboard (alternative). |
| `assets/style.css` | Navy-and-white theme for the Dash dashboard (loaded automatically). |
| `app.py` | The alternative Streamlit dashboard. |
| `models/credit_risk_model.joblib` | The saved, fitted pipeline plus metadata (features, test IDs, metrics, dataset fingerprint). |
| `outputs/` | Exported results (see section 8). |
| `tests/` | Automated checks (`python -m pytest -v`). |
| `.streamlit/config.toml` | Theme for the Streamlit version. |

---

## 3. The dataset

**UCI "Default of Credit Card Clients"**: 30,000 credit-card customers of a Taiwanese bank, April to September 2005.

The Excel file has an extra first row (`X1 … X23, Y`). The real column names are on **row 2**, and
`data_loader.py` detects this automatically. Checks on the actual file found 30,000 rows, 25 columns,
no missing values and no duplicate IDs. There are 35 rows identical apart from their ID; they are kept,
because different customers can have the same values.

| Column(s) | Meaning |
|---|---|
| `ID` | Customer identifier. **Never used as a model input.** |
| `LIMIT_BAL` | Credit limit (NT dollars), for the individual and their family. |
| `SEX` | 1 = male, 2 = female (categorical) |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = others; **0, 5, 6 are undocumented** and kept as their own categories |
| `MARRIAGE` | 1 = married, 2 = single, 3 = others; **0 is undocumented** |
| `AGE` | Age in years |
| `PAY_0, PAY_2 … PAY_6` | Repayment status for Sep, Aug, Jul, Jun, May, Apr 2005 (there is no `PAY_1` in the original file) |
| `BILL_AMT1 … BILL_AMT6` | Bill statement amount, Sep (1) back to Apr (6). Negative values are credit balances (overpayments). |
| `PAY_AMT1 … PAY_AMT6` | Amount paid, Sep (1) back to Apr (6) |
| `default payment next month` | **Target**: 1 = defaulted in October 2005, 0 = did not. Verified to contain only 0/1, and 22.1% are 1. |

### Adding more data files

Every `.xls`, `.xlsx` or `.csv` file in the project folder whose header has the expected column names is
**combined** with the others for training. Files that don't match, and anything inside `outputs/`, are ignored.

- Each row gets a `Source_File` column. It appears in exports and filters but is **never a model input**.
- Customer IDs must be unique across files; otherwise the app stops with a clear message.
- The train/test split is stratified by outcome **and** source file, so every file has customers in the test set.
  The Model Performance page shows the results for each file separately.
- Adding, removing or editing a file changes the data fingerprint, so the model retrains automatically on the next start.

Current files: `default of credit card clients.xls` (30,000 customers, 22.1% defaulted) and
`credit_risk_test_data traning 2.xlsx` (120 customers, 30.8% defaulted).

### Repayment-status codes

These are **category codes, not amounts**:

| Code | Meaning |
|---|---|
| -2 | No balance / no credit used that month* |
| -1 | Paid in full ("pay duly") |
| 0 | Revolving credit: the minimum was paid and a balance was carried over* |
| 1 – 9 | Payment delayed by 1 – 9 months |

\* -2 and 0 are not described in the original UCI documentation. The meanings above are the widely
used interpretation. Because the codes are not on a numeric scale, the model **one-hot encodes** them
(one yes/no column per code) instead of treating them as numbers. No rows are deleted because of them.

### What "default risk" means

Default means the customer **failed to make the required payment** the following month. Default risk
is the likelihood that this happens. Lenders use it to price credit, set limits and decide whom to contact early.

---

## 4. Dashboard pages

| Page | What it shows |
|---|---|
| **Overview** | KPI cards: total customers, *actual* default rate and count, average limit, number of *predicted* high-risk customers, average *predicted* probability. Charts: historical outcomes, historical default rate by repayment status, predicted segments, predicted probability distribution, most important model inputs. |
| **Data Exploration** | Historical default rates by age, credit-limit range, repayment status, education and marital status. Defaulters vs non-defaulters (limits, bills, payments), correlation heatmap, distributions. |
| **Risk Segmentation** | Customers per segment, share, average predicted probability, probability histogram with thresholds, filters (segment, age group, limit range, train/test), highest-risk table, CSV download. |
| **Individual Risk Assessment** | Pick a customer by ID, or type in a new customer. Shows the estimated probability, segment, the main inputs pushing it up or down, and, for existing customers, the *recorded* outcome kept separate from the prediction. |
| **Model Performance** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report, ROC and precision-recall curves, a decision-threshold explorer, calibration, coefficients, permutation importance, fairness notes. |
| **Data Quality** | Rows/columns, types, missing values, duplicates, target distribution, unusual values, undocumented codes, descriptive statistics. |

**Colour convention:** blue/orange = *historical* (what happened), violet = *model predictions*,
green/amber/red = predicted risk segments (always with a text label).

The risk-segment thresholds (default **Low < 30% ≤ Medium < 50% ≤ High**) can be changed in the
sidebar. They are **demonstration thresholds, not banking standards** or validated lending rules.

---

## 5. How the model works (beginner level)

### Logistic regression

Logistic regression adds up the inputs, each multiplied by a learned weight (a *coefficient*), to get a
score called the **log-odds**:

```
log-odds = intercept + w1 × LIMIT_BAL + w2 × AGE + … + (weight for "PAY_0 = 2 months late") + …
```

It then squashes that score into a probability between 0 and 1 with the *sigmoid* function:

```
probability = 1 / (1 + e^(−log-odds))
```

During training, the weights are chosen so the predicted probabilities match the real outcomes in the
training data as closely as possible.

### Steps in `model_training.py`

1. **Features**: credit limit, demographics, the six repayment statuses, six bill amounts and six payment amounts.
   The `ID` and the target are excluded. All features describe Apr–Sep 2005, i.e. they are known *before* the
   October outcome being predicted.
2. **Split**: 80% training, 20% test, **stratified** (both keep the 22% default share), `random_state=42`
   so results are reproducible.
3. **Pipeline** (fitted on the training data only):
   - numeric columns → fill missing with the training median → standardise (mean 0, std 1);
   - categorical columns (`SEX`, `EDUCATION`, `MARRIAGE`, `PAY_x`) → fill missing with the most frequent
     code → one-hot encode, with `handle_unknown="ignore"` so unseen codes don't crash the model;
   - `LogisticRegression`.
4. **Evaluate** on the untouched test set and save everything to `models/credit_risk_model.joblib`.

### Class imbalance

Only 22% of customers defaulted. Two options were compared on the test set:

| | Standard (default) | `class_weight="balanced"` |
|---|---|---|
| Recall at 50% cut-off | 0.35 | 0.55 |
| Precision at 50% cut-off | 0.66 | 0.49 |
| ROC-AUC | 0.758 | 0.759 |
| Average predicted probability (actual rate 22.1%) | 22.1% | 43.5% |

`balanced` gives defaulters more weight, so it catches more of them at the 50% cut-off. The cost is
probabilities that are **systematically too high** (poorly calibrated). This dashboard shows
probabilities and probability-based segments, so the **standard model is the default**. To catch more
defaulters, lower the decision threshold on the Model Performance page, or retrain with `balanced`
from the sidebar.

### Individual probabilities

For a customer, the app passes their inputs (never the ID or actual outcome) through the *same* saved
pipeline and calls `predict_proba()`. The second column is the probability of default. A result of 0.72
is shown as 72%.

This is a **multivariable estimate** for that one customer. It is *not* the historical default rate of
a group, such as "all customers with PAY_0 = 2".

The "main drivers" chart compares the customer with an average training customer: contribution =
coefficient × (customer's value − average value), summed per original input. Red bars raise the
estimate and blue bars lower it.

---

## 6. Evaluation metrics in plain English

The model calls someone a *defaulter* if their probability is ≥ 50% (adjustable on the Model Performance page).

- **Accuracy**: share of all customers classified correctly. *Misleading when classes are imbalanced*:
  always predicting "no default" already scores about 78%.
- **Precision**: of the customers flagged as defaulters, how many really defaulted. Low precision means many false alarms.
- **Recall**: of the customers who really defaulted, how many the model caught. In credit risk, a missed
  defaulter (false negative) usually costs more than a false alarm, so **recall matters a lot**.
- **F1 score**: one number balancing precision and recall (their harmonic mean).
- **ROC-AUC**: the chance that a random defaulter receives a higher score than a random non-defaulter.
  0.5 = guessing, 1.0 = perfect. It does not depend on the cut-off.
- **Confusion matrix**: counts of true/false positives and negatives.
- **Calibration**: whether "40%" really means about 40 in 100 such customers defaulted. Measured with a
  calibration curve and the Brier score.

All values shown in the app are calculated from the actual test set. None are hard-coded.

Results of the current model on the 6,000 test customers: ROC-AUC 0.758, accuracy 0.817 (baseline 0.779),
precision 0.661, recall 0.352, F1 0.459. Average predicted probability is 22.1% vs an actual rate of 22.1%.

---

## 7. Retraining the model

The app trains automatically **only** if the model file is missing, can't be loaded, or was trained on
a different version of the dataset (checked with a file fingerprint). Otherwise it reuses the saved model.

To retrain on purpose:

- In either dashboard: sidebar → **Retrain model** → *Retrain now* (optionally tick `class_weight="balanced"`).
- From the terminal:

  ```powershell
  python model_training.py              # standard model
  python model_training.py --balanced   # class_weight="balanced"
  ```

  This also refreshes the files in `outputs/`.

---

## 8. Downloading results

| File (in `outputs/`) | Contents | Where to download in the app |
|---|---|---|
| `customer_predictions.csv` | Every customer: original inputs, `Data_Split` (Train/Test), `Actual_Default`, `Predicted_Default_Probability`, `Predicted_Default`, `Risk_Segment` | Risk Segmentation (filtered download, or *Save all predictions to outputs/*) |
| `test_predictions.csv` | Held-out test customers only: actual vs predicted, plus `Correct_Prediction` | Model Performance |
| `model_metrics.json` | All test-set metrics, curves, calibration, permutation importance | Model Performance |
| `data_quality_summary.csv` | Overall checks, per-column details, undocumented codes | Data Quality |

Predictions for **training** customers are slightly optimistic, because the model learned from their outcomes.
Use the `Test` rows to judge real performance.

---

## 9. Running the tests

```powershell
python -m pytest -v
```

The tests check:
- the loader against the real file (30,000 × 25, header on row 2, target 0/1);
- header and alias detection, and a missing-target error;
- that the ID and target are excluded from the features;
- the stratified 80/20 split;
- probabilities between 0 and 1;
- metrics matching scikit-learn;
- that the explanation adds up to the model's log-odds;
- segment boundaries, input validation and CSV export;
- missing or corrupted model files;
- that every Dash page renders and the main Dash callbacks work (`tests/test_dash_app.py`);
- every JSON endpoint of `server.py` over real HTTP, including error responses, downloads and
  new-customer validation (`tests/test_server.py`).

---

## 10. Limitations and fairness

- **Historical snapshot**: one Taiwanese bank, 2005. Customer behaviour, economic conditions and lending
  rules have changed, so the model should not be applied to other populations without re-validation.
- **Not a time series**: there is no monthly trend to chart; the six months are features of one snapshot.
- **Correlation ≠ causation**: coefficients, importances and group default rates show associations. A
  feature with a large weight does not *cause* default.
- **Correlated inputs**: the six bill amounts move together, so their individual coefficients can be unstable.
- **Moderate accuracy**: ROC-AUC ≈ 0.76. Many defaulters still receive low probabilities.
- **Undocumented codes**: the meanings of `PAY_x = -2 / 0` and `EDUCATION 0/5/6`, `MARRIAGE 0` are not
  officially documented.
- **Fairness**: `SEX`, `AGE`, `EDUCATION` and `MARRIAGE` are demographic attributes, and other inputs can act
  as proxies for them. Using such attributes in real credit decisions may be unlawful or discriminatory.
  Any real-world use would require legal review, fairness testing across groups, and probably removing
  these inputs.
- **Not a lending decision tool**: probabilities are estimates, not guarantees, and the risk thresholds are
  for demonstration only.
