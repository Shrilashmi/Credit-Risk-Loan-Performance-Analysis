# Resume project brief

## Project title

**Credit Risk & Loan Performance Analysis | Python, NumPy, pandas, SQL, HTML/SVG**

## Resume bullets

- Built a reproducible end-to-end credit portfolio analytics pipeline using **12,000 deterministically generated synthetic installment loans**, including feature preparation, default/loss analysis, monthly vintages, and a responsive standalone dashboard.
- Developed an interpretable regularized logistic regression and **chronological out-of-time validation**; evaluated ranking and probability quality with ROC-AUC, Gini, KS, average precision, Brier score, calibration deciles, and score-cutoff sensitivity.
- Added portfolio monitoring outputs for risk grade and term, principal-normalized charge-off rates, recoveries, feature stability screening (PSI), and loan-level holdout scores; documented data limitations and responsible-use boundaries.

## Demonstration results (synthetic only)

The seeded example run reports **0.786 ROC-AUC**, **0.572 Gini**, **0.116 Brier score**, and **0.462 average precision** on a later-date holdout with **16.9% default prevalence**. These are results on a designed synthetic data-generating process, not real borrowers, a lending portfolio, or expected production performance. Always retain that qualification when describing this work.

## Interview talking points

- **Why chronological validation?** It better resembles scoring future originations and avoids the optimistic leakage risk of a random split across time. The split is grouped by date so one day does not land on both sides.
- **Why more than ROC-AUC?** Ranking does not establish probability accuracy or business value. Brier score, calibration, prevalence-relative average precision, threshold trade-offs, and portfolio loss views add context.
- **What does PSI mean here?** A binned train-versus-holdout distribution-shift screen. It is not a universal accept/reject threshold; bin definitions and business context matter.
- **What would production work need?** Representative observed data, censoring and seasoning design, leakage review, independent validation, calibration, subgroup and legal review, decision economics, governance, explainability, and ongoing drift monitoring.

## One-minute walkthrough

1. Run `python src/credit_risk.py` to regenerate the synthetic records and outputs.
2. Open `outputs/dashboard.html` for risk-grade, vintage, loss, calibration, and model views.
3. Explain the out-of-time split, diagnostics, and why no synthetic metric should be presented as real-world model performance.
