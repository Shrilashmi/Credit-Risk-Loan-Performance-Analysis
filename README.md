# Credit Risk & Loan Performance Analysis

An end-to-end, reproducible credit portfolio project using synthetic loan data. It covers portfolio and vintage analysis, an interpretable logistic regression model, chronological validation, calibration and stability diagnostics, SQL examples, and a standalone dashboard.

**All loan records and outcomes are synthetic.**

## Run it

Requirements: Python 3.10 or later, NumPy, and pandas. Open a terminal in the project folder and run these commands separately:

```powershell
python -m pip install -r requirements.txt
python src/credit_risk.py
```

Then open `outputs/dashboard.html` in a browser. See `outputs/findings.md` for the written results. The fixed random seed makes the generated data reproducible. The dashboard works without an internet connection.

## Project structure

- `Project_Overview.docx` — Word overview of the project, results, setup, and limitations.
- `src/credit_risk.py` — generates the synthetic data, prepares features, splits the data by date, and fits the logistic regression model.
- `src/portfolio_insights.py` — creates portfolio summaries, model diagnostics, findings, and the dashboard.
- `data/synthetic_loan_book.csv` — generated loan data with 12,000 fictional records.
- `outputs/dashboard.html` — standalone dashboard for portfolio, loss, vintage, calibration, and model results.
- `outputs/findings.md` — executive summary and limitations.
- `outputs/model_metrics.csv` — model validation metrics.
- `outputs/threshold_sensitivity.csv` — illustrative score-cutoff diagnostics; not lending policy.
- `outputs/calibration_deciles.csv` — predicted and observed default rates by score decile.
- `outputs/validation_risk_bands.csv` — observed outcomes by illustrative risk band.
- `outputs/model_coefficients.csv` — standardized model coefficients and odds ratios.
- `outputs/feature_stability_psi.csv` — feature stability screening between training and holdout data.
- `outputs/holdout_loan_scores.csv` — predicted probabilities and actual outcomes for holdout loans.
- `outputs/risk_grade_performance.csv`, `outputs/term_performance.csv`, and `outputs/monthly_performance.csv` — portfolio segment, term, and vintage summaries.
- `outputs/resume_project_brief.md` — resume bullets and interview talking points.
- `sql/portfolio_analysis.sql` — SQL examples for portfolio, risk-grade, vintage, and term analysis.

## Data and target

The generator creates 12,000 fictional consumer installment loans with origination dates from 2021-01-01 through 2024-06-30. The target, `default_90dpd_12m`, is a simulated indicator for reaching 90 or more days past due within the first 12 months. Each generated loan is assumed to have a complete 12-month observation window.

The generated features include risk grade, loan term, income, debt-to-income ratio, revolving utilization, recent inquiries, prior delinquencies, loan amount, and interest rate. Charge-offs, recoveries, and ending balances are also simulated.

`loan_id` is a synthetic identifier. Amounts are illustrative and are not tied to a real country, currency, lender, or portfolio. The dataset contains no personal or protected attributes.

## Methodology

The model is a regularized logistic regression implemented with NumPy. Features are standardized using the training data; income and loan amount are log-transformed. The latest 20% of origination dates are held out for chronological validation. The split is made by date so loans from the same origination day do not appear in both training and holdout data.

The project reports ROC-AUC, Gini, KS, average precision, Brier score, threshold precision and recall, calibration by score decile, and outcomes by risk band. Portfolio views include default incidence, pricing, average loan amount, charge-offs, recoveries, and loss rates relative to original principal.

The population stability index (PSI) is a screening measure, not a universal pass/fail test. The 50% score threshold is illustrative and is not a recommended lending cutoff.

## Resume summary

See the [resume project brief](outputs/resume_project_brief.md) for suggested resume bullets and interview talking points. Describe the results as findings from a synthetic case study, not as model performance on real borrowers.

## Limitations and responsible use

The synthetic data deliberately includes relationships between loan features and default outcomes. The model therefore measures how well it recovers patterns designed into the data; it does not demonstrate real-world predictive performance.

The project omits macroeconomic changes, incomplete observation windows, prepayment, fraud, approval selection, servicing behavior, and real decision costs. Excluding protected attributes does not establish fairness. This is a learning and portfolio project, not an underwriting tool, credit policy, expected-credit-loss estimate, or regulatory-compliance claim.

Real-world use would require representative data, independent validation, calibration, subgroup and legal review, decision-cost analysis, governance, explainability, and ongoing monitoring.