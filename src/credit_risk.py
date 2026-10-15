from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'synthetic_loan_book.csv'
OUT = ROOT / 'outputs'
SEED = 20261015
N = 12000


def make_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Create a deterministic, fictional consumer installment-loan portfolio."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.date_range('2021-01-01', '2024-06-30', freq='D'), n))
    grades = rng.choice(['A', 'B', 'C', 'D', 'E'], n, p=[.16, .27, .30, .19, .08])
    grade_num = pd.Series(grades).map({'A':0,'B':1,'C':2,'D':3,'E':4}).to_numpy()
    term = rng.choice([36, 60], n, p=[.68,.32])
    income = np.clip(rng.lognormal(np.log(72000), .48, n), 18000, 350000).round().astype(int)
    dti = np.clip(rng.beta(2.2, 5.2, n) + grade_num*.012, .01, .78)
    utilization = np.clip(rng.beta(2.0, 3.5, n) + grade_num*.022, .01, .99)
    inquiries = np.clip(rng.poisson(1.1 + grade_num*.35, n), 0, 12)
    delinq = np.clip(rng.poisson(.12 + grade_num*.14, n), 0, 5)
    loan = np.clip(rng.lognormal(np.log(14500), .55, n), 1000, 60000).round().astype(int)
    apr = np.clip(7.0 + grade_num*3.1 + rng.normal(0, 1.4, n), 5.5, 29.9).round(2)
    # All records have a full 12-month observation window; no right censoring.
    logit = (-4.65 + .63*grade_num + 2.15*dti + 1.05*utilization + .20*inquiries + .39*delinq
             + .20*(term == 60) + .10*np.log(loan/15000) - .000003*(income-70000)
             + rng.normal(0, .28, n))
    prob = 1/(1+np.exp(-logit))
    default = rng.binomial(1, prob)
    # Simulated performance and collections fields, useful for portfolio monitoring.
    chargeoff_rate = np.where(default == 1, rng.uniform(.25,.90,n), 0)
    balance = np.round(loan * rng.uniform(.35,.95,n), 2)
    recoveries = np.round(balance * chargeoff_rate * rng.uniform(.02,.28,n), 2)
    return pd.DataFrame({
        'loan_id':[f'LN{i:06d}' for i in range(1,n+1)], 'origination_date':dates.strftime('%Y-%m-%d'),
        'risk_grade':grades, 'term_months':term, 'annual_income':income, 'debt_to_income':dti.round(4),
        'revolving_utilization':utilization.round(4), 'inquiries_6m':inquiries, 'prior_delinquencies':delinq,
        'loan_amount':loan, 'interest_rate_pct':apr, 'default_90dpd_12m':default,
        'ending_balance':balance, 'gross_chargeoff':np.round(balance*chargeoff_rate,2), 'recoveries':recoveries,
    })


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-35,35)))


def fit_logistic(x, y, steps=1800, lr=.08, ridge=.03):
    mu, sd = x.mean(0), x.std(0); sd[sd == 0] = 1
    z = (x-mu)/sd; z = np.column_stack([np.ones(len(z)),z]); w=np.zeros(z.shape[1])
    for _ in range(steps):
        p=sigmoid(z@w); grad=z.T@(p-y)/len(y); grad[1:]+=ridge*w[1:]; w-=lr*grad
    return w,mu,sd


def auc(y, p):
    order=np.argsort(p); ranks=np.empty(len(p),float); ranks[order]=np.arange(1,len(p)+1)
    pos=y==1; n1=pos.sum(); n0=len(y)-n1
    return float((ranks[pos].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else float('nan')


def main():
    OUT.mkdir(exist_ok=True); DATA.parent.mkdir(exist_ok=True)
    df=make_data(); df.to_csv(DATA,index=False)
    features=['risk_grade','term_months','annual_income','debt_to_income','revolving_utilization','inquiries_6m','prior_delinquencies','loan_amount','interest_rate_pct']
    x=pd.DataFrame({'risk_grade':df.risk_grade.map({'A':0,'B':1,'C':2,'D':3,'E':4}),
        'term_months':df.term_months,'annual_income':np.log(df.annual_income),'debt_to_income':df.debt_to_income,
        'revolving_utilization':df.revolving_utilization,'inquiries_6m':df.inquiries_6m,'prior_delinquencies':df.prior_delinquencies,
        'loan_amount':np.log(df.loan_amount),'interest_rate_pct':df.interest_rate_pct})[features].to_numpy(float)
    # Chronological validation: train earlier loans, evaluate the newest 20%.
    dates=df.origination_date.to_numpy(); unique_dates=np.unique(dates); cutoff=unique_dates[int(len(unique_dates)*.8)]; train=np.flatnonzero(dates<cutoff); test=np.flatnonzero(dates>=cutoff)
    w,mu,sd=fit_logistic(x[train],df.default_90dpd_12m.to_numpy()[train])
    p=sigmoid(np.column_stack([np.ones(len(test)),(x[test]-mu)/sd])@w); y=df.default_90dpd_12m.to_numpy()[test]
    pred=(p>=.5).astype(int); tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    metrics={'test_rows':len(test),'test_default_rate':float(y.mean()),'roc_auc':auc(y,p),'brier_score':float(np.mean((p-y)**2)),
      'threshold':.5,'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),'train_start':df.origination_date.iloc[train].min(),
      'train_end':df.origination_date.iloc[train].max(),'test_start':df.origination_date.iloc[test].min(),'test_end':df.origination_date.iloc[test].max()}
    pd.DataFrame([metrics]).to_csv(OUT/'model_metrics.csv',index=False)
    # Portfolio summaries
    d=df.copy(); d['origination_month']=pd.to_datetime(d.origination_date).dt.to_period('M').astype(str)
    monthly=d.groupby('origination_month').agg(loans=('loan_id','count'),originations=('loan_amount','sum'),default_rate=('default_90dpd_12m','mean'),gross_chargeoff=('gross_chargeoff','sum'),recoveries=('recoveries','sum')).reset_index()
    grade=d.groupby('risk_grade',observed=True).agg(loans=('loan_id','count'),average_apr_pct=('interest_rate_pct','mean'),average_loan_amount=('loan_amount','mean'),default_rate=('default_90dpd_12m','mean'),gross_chargeoff=('gross_chargeoff','sum'),recoveries=('recoveries','sum')).reset_index()
    grade.to_csv(OUT/'risk_grade_performance.csv',index=False); monthly.to_csv(OUT/'monthly_performance.csv',index=False)
    scored=pd.DataFrame({'loan_id':df.loan_id.iloc[test].to_numpy(),'actual_default':y,'predicted_default_probability':p})
    scored['risk_band']=pd.cut(scored.predicted_default_probability,[-.001,.05,.10,.20,1.01],labels=['Low (<5%)','Moderate (5–10%)','Elevated (10–20%)','High (>20%)'])
    bands=scored.groupby('risk_band',observed=True).agg(loans=('loan_id','count'),average_predicted_risk=('predicted_default_probability','mean'),observed_default_rate=('actual_default','mean')).reset_index()
    bands.to_csv(OUT/'validation_risk_bands.csv',index=False)
    # Compact HTML dashboard: native SVG charts, no external JS/CDN.
    def bars(labels, values, fmt, color='#45d6b5'):
        mx=max(values) if len(values) else 1; width=560; row=34; height=row*len(values)+12
        out=[f'<svg viewBox="0 0 {width} {height}" role="img">']
        for i,(lab,val) in enumerate(zip(labels,values)):
            yy=8+i*row; bw=max(2,(width-180)*val/mx)
            out.append(f'<text x="0" y="{yy+18}">{lab}</text><rect x="150" y="{yy}" width="{bw:.1f}" height="22" rx="5" fill="{color}"/><text x="{min(155+bw,510):.1f}" y="{yy+17}" class="value">{fmt(val)}</text>')
        out.append('</svg>'); return ''.join(out)
    labels=grade.risk_grade.tolist(); dr=(grade.default_rate*100).tolist()
    mlabels=monthly.origination_month.tolist(); mvals=(monthly.default_rate*100).tolist()
    kpis=[('Loans',f'{len(df):,}'),('12-month defaults',f'{df.default_90dpd_12m.mean():.1%}'),('Originations',f'${df.loan_amount.sum()/1e6:,.1f}M'),('Net charge-offs',f'${(df.gross_chargeoff.sum()-df.recoveries.sum())/1e6:,.2f}M'),('Validation ROC-AUC',f'{metrics["roc_auc"]:.3f}')]
    cards=''.join(f'<div class="card"><small>{a}</small><b>{b}</b></div>' for a,b in kpis)
    tab=grade.to_html(index=False,float_format=lambda v:f'{v:,.2f}',classes='tbl')
    bandtab=bands.to_html(index=False,float_format=lambda v:f'{v:.1%}' if v<1 else f'{v:.3f}',classes='tbl')
    html=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Credit Risk & Loan Performance</title><style>
    :root{{--bg:#0c111b;--panel:#151d2b;--ink:#e9f0fa;--muted:#9baac0;--mint:#45d6b5;--line:#263449}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Segoe UI,Arial,sans-serif}}main{{max-width:1150px;margin:auto;padding:42px 26px}}h1{{font-size:32px;line-height:1.15;margin:10px 0}}h2{{margin:0 0 14px;font-size:19px}}p,.sub{{color:var(--muted)}}.tag{{color:var(--mint);text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:26px 0}}.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}}.card small{{display:block;color:var(--muted);margin-bottom:8px}}.card b{{font-size:25px}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}svg{{width:100%;overflow:visible}}svg text{{fill:var(--muted);font-size:12px}}svg text.value{{fill:var(--ink)}}.tbl{{border-collapse:collapse;width:100%;font-size:13px}}.tbl th,.tbl td{{padding:9px 7px;border-bottom:1px solid var(--line);text-align:right}}.tbl th:first-child,.tbl td:first-child{{text-align:left}}.tbl th{{color:var(--mint)}}.note{{border-left:3px solid var(--mint);padding:2px 14px;margin:20px 0;color:var(--muted)}}@media(max-width:750px){{.cols{{grid-template-columns:1fr}}main{{padding:28px 16px}}}}</style></head><body><main><div class="tag">Portfolio intelligence · synthetic case study</div><h1>Credit Risk & Loan Performance</h1><p class="sub">A reproducible analysis of 12,000 fictional installment loans. Outcomes represent fully observed 12-month performance.</p><section class="grid">{cards}</section><section class="cols"><div class="panel"><h2>Default rate by risk grade</h2>{bars(labels,dr,lambda v:f'{v:.1f}%','#45d6b5')}</div><div class="panel"><h2>Monthly vintage default rate</h2>{bars(mlabels,mvals,lambda v:f'{v:.1f}%','#80a7ff')}</div></section><section class="panel" style="margin-top:14px"><h2>Risk grade performance</h2>{tab}</section><section class="cols" style="margin-top:14px"><div class="panel"><h2>Out-of-time validation risk bands</h2>{bandtab}</div><div class="panel"><h2>Model validation</h2><p>Regularized logistic regression; chronological 80/20 split.</p><p><b>ROC-AUC:</b> {metrics['roc_auc']:.3f}<br><b>Brier score:</b> {metrics['brier_score']:.3f}<br><b>Test period:</b> {metrics['test_start']} – {metrics['test_end']}<br><b>Threshold shown:</b> 50%</p><p>Review ranking, calibration, and decision costs together; 50% is not a recommended lending cutoff.</p></div></section><div class="note"><b>Interpretation:</b> All rows are simulated. Relationships are designed for demonstration and must not be used to price, approve, decline, or provision real loans. No protected attributes are included; this does not establish fairness.</div><p class="sub">Rebuild with <code>python src/credit_risk.py</code>. Definitions and limitations are in README.md.</p></main></body></html>'''
    (OUT/'dashboard.html').write_text(html,encoding='utf-8')
    # Executive summary in markdown
    summary=f'''# Credit Risk & Loan Performance — Findings\n\n**Dataset:** {len(df):,} deterministic synthetic loans; no customer or lender data. Each loan has a complete 12-month outcome window.\n\n## Portfolio snapshot\n\n- Total originations: **${df.loan_amount.sum():,.0f}** across **{len(df):,}** loans.\n- 12-month 90+ days-past-due default incidence: **{df.default_90dpd_12m.mean():.2%}**.\n- Simulated net charge-offs after recoveries: **${df.gross_chargeoff.sum()-df.recoveries.sum():,.0f}**.\n- Risk-grade defaults range from **{grade.default_rate.min():.2%}** (grade {grade.iloc[grade.default_rate.argmin()].risk_grade}) to **{grade.default_rate.max():.2%}** (grade {grade.iloc[grade.default_rate.argmax()].risk_grade}).\n\n## Model assessment\n\nA regularized logistic regression ranks loans using nine application-time features. The chronological holdout includes **{len(test):,}** later originations ({metrics['test_start']} to {metrics['test_end']}). ROC-AUC is **{metrics['roc_auc']:.3f}**, Brier score **{metrics['brier_score']:.3f}**, and the holdout default incidence is **{metrics['test_default_rate']:.2%}**. At an illustrative 50% threshold, precision is {metrics['precision']:.1%} and recall is {metrics['recall']:.1%}; this threshold is not an operating recommendation.\n\n## Practical readout\n\n- Track vintage performance with consistent observation windows; do not compare recent unseasoned loans with fully matured cohorts.\n- Monitor grade-level incidence, charge-offs, recoveries, pricing, and approval mix together.\n- Before production use, validate on representative real data, check calibration and subgroup outcomes, quantify decision costs, establish governance, and monitor drift.\n\n## Definitions and limitations\n\n`default_90dpd_12m` is a simulated binary indicator for reaching 90+ days past due within 12 months. Gross charge-offs and recoveries are simulated balances, not accounting projections. There is no censoring, macroeconomic cycle, fraud, prepayment, servicing effect, or selection bias. The data-generating process intentionally links risk grades and financial features to outcomes; validation on it measures recovery of a designed signal, not real-world predictive value. Protected attributes are excluded; that alone does not demonstrate fairness.\n\n## Reproduction\n\nInstall `numpy` and `pandas`, then run `python src/credit_risk.py` from the project root. The script regenerates the CSV input and all dashboard and summary tables in `outputs/`.\n'''
    (OUT/'findings.md').write_text(summary,encoding='utf-8')
    from portfolio_insights import build_extras
    build_extras(df, x, features, train, test, w, mu, sd, p, y, metrics, grade, monthly, bands)
    print(f'Built {len(df)} rows; overall default {df.default_90dpd_12m.mean():.3%}; AUC {metrics["roc_auc"]:.3f}; outputs {OUT}')

if __name__=='__main__': main()



