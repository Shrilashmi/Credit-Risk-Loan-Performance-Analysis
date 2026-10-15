# Credit Risk & Loan Performance Analysis

An end-to-end, reproducible credit portfolio project: deterministic loan generation, portfolio/vintage analysis, an interpretable logistic regression score, chronological validation, calibration and drift diagnostics, and a standalone dashboard. **All records and outcomes are synthetic.**

## Run it

Requirements: Python 3.10+, NumPy, and pandas. From this project folder:

```powershell
python -m pip install -r requirements.txt
python src/credit_risk.py
```

Open `outputs/dashboard.html` in a browser. Read `outputs/findings.md` for the written summary. The fixed random seed makes the generated files reproducible. The dashboard uses embedded charts and does not need an internet connection.

## Project structure

- `src/credit_risk.py` — seeded synthetic-data generator, feature preparation, chronological split, and baseline logistic regression.
- `src/portfolio_insights.py` — portfolio loss cuts, calibration and stability diagnostics, and dashboard/report generation.
- `data/synthetic_loan_book.csv` — generated loan book (12,000 fictional records).
- `outputs/dashboard.html` — responsive dashboard covering portfolio, loss, vintages, calibration, and model performance.
- `outputs/findings.md` — executive findings and limitations.
- `outputs/model_metrics.csv` — holdout ranking, calibration, and threshold metrics.
- `outputs/threshold_sensitivity.csv` — score-cutoff sensitivity diagnostics, not lending policy.
- `outputs/calibration_deciles.csv` — mean predicted and observed default by score decile.
- `outputs/validation_risk_bands.csv` — observed rates by illustrative predicted-risk bands.
- `outputs/model_coefficients.csv` — standardized coefficients and odds ratios.
- `outputs/feature_stability_psi.csv` — train/holdout feature population stability screening.
- `outputs/holdout_loan_scores.csv` — loan ID, actual label, and predicted probability for the temporal holdout.
- `outputs/risk_grade_performance.csv`, `outputs/term_performance.csv`, `outputs/monthly_performance.csv` — portfolio segments, vintage trends, and loss ratios.
- `sql/portfolio_analysis.sql` — portable SQL examples for independent portfolio cuts.
- `outputs/resume_project_brief.md` — concise project bullets and interview talking points.

## Data and target

The generator creates 12,000 fictional consumer installment loans with origination dates from 2021-01-01 through 2024-06-30. The target, `default_90dpd_12m`, is a simulated binary 90+ days-past-due event within the first 12 months. Each generated record is stipulated to have a full observation window. Risk grade, term, income, debt-to-income, utilization, inquiries, prior delinquency, principal, and APR are generated features. Charge-offs, recoveries, and ending balances are also simulated.

`loan_id` is a synthetic identifier. Amounts are illustrative currency units; they are not tied to a real country, currency, portfolio, or institution. No personal or protected attributes are created.

## Methodology

The model is a regularized logistic regression implemented with NumPy. Features are standardized on the training sample; continuous income and principal are log-transformed. The latest 20% of calendar dates are held out chronologically, with the cutoff set by date so a single origination day cannot leak across the split. The dashboard and CSVs report ROC-AUC, Gini, KS, average precision, Brier score, threshold precision/recall, score-decile calibration, and risk-band outcomes. Portfolio views include default incidence, APR, average principal, charge-offs, recoveries, and principal-normalized loss ratios.

PSI is included as a screening diagnostic only. A PSI value has no universal pass/fail interpretation without context, stable binning, and a monitoring policy. The 50% score threshold is illustrative and is not a recommended lending cutoff.

## Resume-ready summary

See [`outputs/resume_project_brief.md`](outputs/resume_project_brief.md) for suggested bullets and a clear description of the synthetic scope. Report metrics as results on a designed synthetic case study, never as performance on real borrowers.

## Limitations and responsible use

The data-generating process intentionally builds relationships between inputs and default. Strong-looking results therefore show recovery of a designed signal, not real-world generalization. The project omits macroeconomic changes, censoring, prepayment, fraud, approval selection, servicing, and actual decision economics. Excluding protected attributes does not establish fairness. This is a learning and portfolio-demonstration project, not an underwriting tool, credit policy, expected-credit-loss estimate, or regulatory-compliance claim. Real use needs representative data, independent validation, calibration, subgroup and legal review, decision-cost analysis, governance, explainability, and ongoing monitoring.
