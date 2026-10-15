# Credit Risk & Loan Performance — Findings

> **Synthetic case study:** all 12,000 loans and outcomes are generated data. No lender or customer records are used.

## Portfolio snapshot

- Originations: **$202,033,181** across 12,000 loans.
- Simulated 12-month 90+ DPD incidence: **17.38%**.
- Simulated net charge-offs: **$11,531,168** after recoveries.
- Grade A to E default incidence: **3.4% to 47.9%**.

## Out-of-time validation

Regularized logistic regression on nine application-time features. Split by origination date, with training through **2023-10-18** and holdout **2023-10-19–2024-06-30** (2,413 loans). Holdout ROC-AUC **0.786**, Gini **0.572**, KS **0.455**, average precision **0.462** versus 16.9% prevalence, and Brier score **0.116**.

At an illustrative 50% score threshold, precision is 69.9% and recall is 14.3%; see `model_metrics.csv` for additional thresholds. These are diagnostics, not lending cutoffs. Review `calibration_deciles.csv` and `validation_risk_bands.csv` with ranking metrics.

## Monitoring and interpretation

- Compare grade-level default incidence, pricing, gross charge-offs, recoveries, and net loss rates together.
- Monthly vintage performance is noisy for smaller cohorts. Every synthetic loan is stipulated to have a full 12-month window; real analyses need seasoning and censoring controls.
- `feature_stability_psi.csv` is a screening aid, not a universal pass/fail rule. Interpret shifts using domain and data-lineage context.

## Reproduction and limitations

Run `python src/credit_risk.py` from the project root. The deterministic seed regenerates the dataset and all `outputs/` artifacts. Outcome relationships are deliberately designed, so model performance measures recovery of that signal, not predictive power on real borrowers. The project omits macroeconomic scenarios, censoring, prepayment, fraud, selection effects, servicing behavior, and decision economics. No protected attributes are generated; that does not establish fairness. This is not an underwriting tool, policy recommendation, accounting estimate, or regulatory-compliance claim. Real use requires representative data, independent validation, legal and subgroup review, decision-cost analysis, governance, explainability, and monitoring.
