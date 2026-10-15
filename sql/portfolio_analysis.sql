-- SQL examples for data/synthetic_loan_book.csv.
-- Load the CSV into a table named loan_book before running these queries.

-- Portfolio summary: incidence and balance-weighted loss rates.
SELECT
    COUNT(*) AS loan_count,
    SUM(loan_amount) AS total_originations,
    AVG(default_90dpd_12m * 1.0) AS default_rate,
    SUM(gross_chargeoff) AS gross_chargeoffs,
    SUM(recoveries) AS recoveries,
    (SUM(gross_chargeoff) - SUM(recoveries)) / NULLIF(SUM(loan_amount), 0) AS net_loss_rate
FROM loan_book;

-- Risk-tier performance: compare pricing, event rates, and net loss.
SELECT
    risk_grade,
    COUNT(*) AS loan_count,
    SUM(loan_amount) AS originations,
    AVG(interest_rate_pct) AS average_apr_pct,
    AVG(default_90dpd_12m * 1.0) AS default_rate,
    (SUM(gross_chargeoff) - SUM(recoveries)) / NULLIF(SUM(loan_amount), 0) AS net_loss_rate
FROM loan_book
GROUP BY risk_grade
ORDER BY risk_grade;

-- Monthly vintage tracking. All synthetic records have a full 12-month window.
SELECT
    SUBSTR(origination_date, 1, 7) AS origination_month,
    COUNT(*) AS loan_count,
    SUM(loan_amount) AS originations,
    AVG(default_90dpd_12m * 1.0) AS default_rate,
    (SUM(gross_chargeoff) - SUM(recoveries)) / NULLIF(SUM(loan_amount), 0) AS net_loss_rate
FROM loan_book
GROUP BY SUBSTR(origination_date, 1, 7)
ORDER BY origination_month;

-- Segment-level performance by contractual term.
SELECT
    term_months,
    COUNT(*) AS loan_count,
    AVG(default_90dpd_12m * 1.0) AS default_rate,
    AVG(interest_rate_pct) AS average_apr_pct,
    (SUM(gross_chargeoff) - SUM(recoveries)) / NULLIF(SUM(loan_amount), 0) AS net_loss_rate
FROM loan_book
GROUP BY term_months
ORDER BY term_months;
