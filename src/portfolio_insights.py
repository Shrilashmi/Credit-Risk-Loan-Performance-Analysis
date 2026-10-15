"""Extended portfolio diagnostics and a self-contained project dashboard."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def _calibration(y, p):
    d = pd.DataFrame({'actual': y, 'predicted': p})
    d['decile'] = pd.qcut(d.predicted.rank(method='first'), 10, labels=False) + 1
    return d.groupby('decile', as_index=False).agg(
        loans=('actual', 'size'), mean_predicted=('predicted', 'mean'),
        observed_default_rate=('actual', 'mean'))


def _stability(x_train, x_test, names):
    rows = []
    for j, name in enumerate(names):
        cuts = np.unique(np.quantile(x_train[:, j], [0, .2, .4, .6, .8, 1]))
        if len(cuts) < 3:
            continue
        edges = np.r_[-np.inf, cuts[1:-1], np.inf]
        a = np.histogram(x_train[:, j], bins=edges)[0] / len(x_train)
        b = np.histogram(x_test[:, j], bins=edges)[0] / len(x_test)
        psi = float(np.sum((b-a) * np.log((b+1e-6)/(a+1e-6))))
        rows.append({'feature': name, 'train_mean': x_train[:, j].mean(),
                     'test_mean': x_test[:, j].mean(), 'psi': psi})
    return pd.DataFrame(rows).sort_values('psi', ascending=False)


def _table(frame, formats=None):
    return frame.to_html(index=False, classes='data-table', border=0, formatters=formats or {})


def build_extras(df, x, features, train, test, weights, mu, sd, p, y, metrics, grade, monthly, bands):
    """Write expanded model diagnostics, portfolio cuts, and refreshed dashboard."""
    OUT.mkdir(exist_ok=True)
    # Add discrimination and threshold diagnostics while keeping the same temporal split.
    pos = np.sort(p[y == 1]); neg = np.sort(p[y == 0])
    ks = float(np.max(np.abs(np.arange(1, len(pos)+1)/len(pos) - np.searchsorted(neg, pos, side='right')/len(neg))))
    order = np.argsort(-p); yy = y[order]
    ap = float((np.cumsum(yy)/np.arange(1, len(y)+1) * yy).sum()/max(yy.sum(), 1))
    metrics.update({'gini': 2*metrics['roc_auc']-1, 'ks': ks, 'average_precision': ap,
                    'train_default_rate': float(df.default_90dpd_12m.iloc[train].mean())})
    for t in [.10, .20, .30, .50]:
        pred = p >= t; tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum()); fn = int(((~pred) & (y == 1)).sum())
        metrics[f'precision_at_{int(t*100)}pct'] = tp/max(tp+fp, 1)
        metrics[f'recall_at_{int(t*100)}pct'] = tp/max(tp+fn, 1)
    threshold_rows = []
    for t in [.10, .20, .30, .50]:
        flagged = p >= t
        tp = int(((flagged) & (y == 1)).sum())
        threshold_rows.append({'illustrative_score_cutoff': t,
            'share_above_cutoff': float(flagged.mean()),
            'precision_among_flagged': tp/max(int(flagged.sum()), 1),
            'share_of_defaults_captured': tp/max(int(y.sum()), 1)})
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT/'threshold_sensitivity.csv', index=False)
    pd.DataFrame([metrics]).to_csv(OUT/'model_metrics.csv', index=False)

    # Balance-weighted loss ratios and additional portfolio cuts.
    grade['originations'] = df.groupby('risk_grade').loan_amount.sum().reindex(grade.risk_grade).to_numpy()
    grade['chargeoff_rate'] = grade.gross_chargeoff / grade.originations
    grade['net_loss_rate'] = (grade.gross_chargeoff-grade.recoveries) / grade.originations
    term = df.groupby('term_months', as_index=False).agg(
        loans=('loan_id','count'), originations=('loan_amount','sum'),
        average_apr_pct=('interest_rate_pct','mean'), average_loan_amount=('loan_amount','mean'),
        default_rate=('default_90dpd_12m','mean'), gross_chargeoff=('gross_chargeoff','sum'), recoveries=('recoveries','sum'))
    term['chargeoff_rate'] = term.gross_chargeoff / term.originations
    term['net_loss_rate'] = (term.gross_chargeoff-term.recoveries) / term.originations
    monthly['net_chargeoff'] = monthly.gross_chargeoff-monthly.recoveries
    monthly['net_loss_rate'] = monthly.net_chargeoff/monthly.originations
    grade.to_csv(OUT/'risk_grade_performance.csv', index=False)
    term.to_csv(OUT/'term_performance.csv', index=False)
    monthly.to_csv(OUT/'monthly_performance.csv', index=False)

    calibration = _calibration(y, p)
    calibration.to_csv(OUT/'calibration_deciles.csv', index=False)
    coef = pd.DataFrame({'feature': features, 'standardized_coefficient': weights[1:],
                         'odds_ratio_per_sd': np.exp(weights[1:])})
    coef = coef.sort_values('standardized_coefficient', key=abs, ascending=False)
    coef.to_csv(OUT/'model_coefficients.csv', index=False)
    _stability(x[train], x[test], features).to_csv(OUT/'feature_stability_psi.csv', index=False)
    scored = pd.DataFrame({'loan_id': df.loan_id.iloc[test].to_numpy(), 'actual_default': y,
                           'predicted_default_probability': p})
    scored.to_csv(OUT/'holdout_loan_scores.csv', index=False)

    pct = lambda v: f'{v:.1%}'
    money = lambda v: f'${v:,.0f}'
    percent_fmt = {'average_apr_pct': lambda v: f'{v:.2f}%', 'average_loan_amount': money,
        'default_rate': pct, 'chargeoff_rate': pct, 'net_loss_rate': pct,
        'gross_chargeoff': money, 'recoveries': money, 'originations': money}
    grade_html = _table(grade, percent_fmt)
    term_html = _table(term, percent_fmt)
    band_html = _table(bands, {'average_predicted_risk': pct, 'observed_default_rate': pct})
    cal_html = _table(calibration, {'mean_predicted': pct, 'observed_default_rate': pct})
    cutoff_html = _table(thresholds, {'illustrative_score_cutoff':pct,'share_above_cutoff':pct,
        'precision_among_flagged':pct,'share_of_defaults_captured':pct})
    coef_html = _table(coef, {'standardized_coefficient': lambda v:f'{v:+.3f}',
                              'odds_ratio_per_sd': lambda v:f'{v:.2f}x'})
    score = pd.DataFrame([{k: metrics[k] for k in ['test_rows','test_default_rate','roc_auc','gini','ks','average_precision','brier_score']}])
    score_html = _table(score, {'roc_auc':lambda v:f'{v:.3f}', 'gini':lambda v:f'{v:.3f}',
        'ks':lambda v:f'{v:.3f}', 'brier_score':lambda v:f'{v:.3f}',
        'average_precision':lambda v:f'{v:.3f}', 'test_default_rate':pct})
    chart_data = json.dumps({'gradeLabels':grade.risk_grade.tolist(),
        'gradeRates':(grade.default_rate*100).round(2).tolist(),
        'vintageLabels':monthly.origination_month.tolist(),
        'vintageRates':(monthly.default_rate*100).round(2).tolist(),
        'calPred':(calibration.mean_predicted*100).round(2).tolist(),
        'calActual':(calibration.observed_default_rate*100).round(2).tolist()})
    orig = df.loan_amount.sum(); netloss = df.gross_chargeoff.sum()-df.recoveries.sum()
    kpis = ''.join(f'<article class="kpi"><span>{label}</span><strong>{value}</strong><small>{sub}</small></article>' for label,value,sub in [
        ('Portfolio size',f'{len(df):,}','fictional loans'),('12-month defaults',pct(df.default_90dpd_12m.mean()),'simulated incidence'),
        ('Originations',f'${orig/1e6:,.1f}M','original principal'),('Net charge-offs',f'${netloss/1e6:,.2f}M','after simulated recoveries'),
        ('Out-of-time AUC',f'{metrics["roc_auc"]:.3f}','later-date holdout')])
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Credit Risk | Portfolio Lab</title><style>
:root{--bg:#0b1220;--panel:#142239;--ink:#edf4ff;--muted:#9aacc4;--line:#2a3d58;--mint:#54e0bf;--blue:#82aaff;--amber:#ffc36b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 75% -10%,#233655 0,transparent 45%),var(--bg);color:var(--ink);font:14px/1.55 Segoe UI,Arial,sans-serif}.wrap{max-width:1200px;margin:auto;padding:30px 26px 55px}.top{display:flex;justify-content:space-between}.brand{font-weight:750;letter-spacing:.08em}.pill{border:1px solid #3a675f;border-radius:99px;padding:5px 10px;color:var(--mint);font-size:10px;letter-spacing:.12em;text-transform:uppercase}.hero{padding:46px 0 26px}.eyebrow{color:var(--mint);font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700}h1{font-size:clamp(32px,5vw,54px);line-height:1.05;letter-spacing:-.04em;margin:12px 0}.lede{font-size:16px;color:#bdcada;max-width:760px}.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0 30px}.kpi,.panel{background:linear-gradient(145deg,#172844,var(--panel));border:1px solid var(--line);border-radius:14px}.kpi{padding:16px;min-height:116px}.kpi span,.kpi small{display:block;color:var(--muted)}.kpi strong{display:block;font-size:25px;margin:8px 0 3px}.kpi small{font-size:11px}.head{margin:30px 0 12px;display:flex;justify-content:space-between;align-items:end}.head h2{margin:0;font-size:19px}.caption{color:var(--muted);font-size:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{padding:18px;min-width:0}.panel h3{font-size:15px;margin:0 0 3px}.chart{height:250px;margin-top:12px}.chart svg{width:100%;height:100%;overflow:visible}.gridline{stroke:#30445f;stroke-dasharray:3 5}.axislabel{fill:#9aacc4;font:11px Segoe UI,Arial}.data-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px}.data-table th,.data-table td{padding:8px 7px;text-align:right;border-bottom:1px solid #2b3d56;white-space:nowrap}.data-table th{color:#b5c6db}.data-table th:first-child,.data-table td:first-child{text-align:left}.scroll{overflow:auto}.note{padding:16px;border:1px solid #635132;background:#241f17;border-radius:12px;color:#dfd0b3}.note b{color:#ffe1a6}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.insights article{padding:15px;background:#111d2e;border-left:2px solid var(--mint);color:#bdcada}.insights b{display:block;color:var(--ink);margin-bottom:5px}.footer{color:var(--muted);font-size:12px;margin-top:30px}@media(max-width:850px){.kpis{grid-template-columns:repeat(2,1fr)}.cols{grid-template-columns:1fr}.insights{grid-template-columns:1fr}}@media(max-width:520px){.wrap{padding:20px 14px 40px}.hero{padding-top:35px}.chart{height:220px}}
</style></head><body><main class="wrap"><div class="top"><div class="brand">◈ &nbsp; PORTFOLIO LAB</div><div class="pill">Synthetic case study</div></div><header class="hero"><div class="eyebrow">Credit risk · portfolio performance · model validation</div><h1>Understand risk across the loan lifecycle.</h1><p class="lede">A reproducible portfolio analytics project, from deterministic loan generation to vintage monitoring and chronological model validation. All loans and outcomes are fictional.</p></header><section class="kpis">__KPIS__</section>
<div class="head"><h2>Portfolio performance</h2><span class="caption">Simulated 12-month outcomes</span></div><section class="cols"><article class="panel"><h3>Default incidence by risk grade</h3><div class="caption">90+ days past due within 12 months</div><div id="grade" class="chart"></div></article><article class="panel"><h3>Monthly origination vintages</h3><div class="caption">Cohorts are fully observed by construction</div><div id="vintage" class="chart"></div></article></section>
<div class="head"><h2>Portfolio mix & loss</h2><span class="caption">Charge-off ratios use original principal</span></div><section class="cols"><article class="panel"><h3>Risk grade economics</h3><div class="caption">Volume, pricing, defaults, and loss</div><div class="scroll">__GRADE__</div></article><article class="panel"><h3>Term performance</h3><div class="caption">Compare 36-month and 60-month loans</div><div class="scroll">__TERM__</div></article></section>
<div class="head"><h2>Chronological model validation</h2><span class="caption">Train through __TRAIN_END__ · holdout __TEST_START__ to __TEST_END__</span></div><section class="cols"><article class="panel"><h3>Predicted versus observed risk</h3><div class="caption">Ten equal-count groups; diagonal is perfect calibration</div><div id="cal" class="chart"></div></article><article class="panel"><h3>Holdout scorecard</h3><div class="caption">Regularized logistic regression · nine features</div><div class="scroll">__SCORE__</div><p class="caption">ROC-AUC measures ranking, Brier score probability error. Neither alone defines a lending decision.</p></article></section>
<div class="head"><h2>Risk band calibration</h2><span class="caption">Holdout predicted risk compared with observed outcomes</span></div><section class="cols"><article class="panel"><h3>Predicted risk bands</h3><div class="scroll">__BANDS__</div></article><article class="panel"><h3>Calibration by decile</h3><div class="scroll">__CAL__</div></article></section><section class="panel" style="margin-top:14px"><h3>Illustrative score-cutoff sensitivity</h3><div class="caption">Share above score cutoff, precision among flagged, and share of defaults captured; diagnostic only, not policy</div><div class="scroll">__CUTOFFS__</div></section>
<div class="head"><h2>Model explainability</h2><span class="caption">Standardized coefficients and odds ratio per standard deviation</span></div><section class="panel"><div class="scroll">__COEF__</div><p class="caption">Fitted associations in synthetic data; these are not causal effects.</p></section>
<div class="head"><h2>What the analysis suggests</h2></div><section class="insights"><article><b>Risk segmentation</b>Default incidence rises across designed A–E tiers. Review pricing and loss together; score rank is not policy.</article><article><b>Vintage context</b>Monthly outcomes are noisy in smaller cohorts. Real vintages need seasoning and censoring controls.</article><article><b>Decision economics</b>A 50% threshold is illustrative only. Real cutoffs need loss severity, funding cost, and customer impact.</article></section>
<div class="head"><h2>Reproducibility & responsible use</h2></div><section class="cols"><article class="panel"><h3>Rebuild the project</h3><p>Run <code>python src/credit_risk.py</code> from the project folder. The seeded pipeline regenerates data, portfolio cuts, validation files, and this standalone dashboard.</p><p class="caption">NumPy and pandas only · no external dashboard assets or network calls.</p></article><aside class="note"><b>Fictional data only.</b> Relationships are intentionally designed. Results do not represent lender performance or support approvals, declines, pricing, provisioning, fairness, or compliance claims. Real use requires representative data, independent validation, legal and subgroup review, decision-cost analysis, governance, and monitoring.</aside></section><div class="footer">Credit Risk & Loan Performance · reproducible synthetic case study · no customer data</div></main><script>
const D=__DATA__;const NS='http://www.w3.org/2000/svg';function el(t,a={},s=''){let n=document.createElementNS(NS,t);for(const [k,v] of Object.entries(a))n.setAttribute(k,v);if(s)n.textContent=s;return n}function setup(id){let host=document.getElementById(id),w=Math.max(300,host.clientWidth),h=host.clientHeight,svg=el('svg',{viewBox:`0 0 ${w} ${h}`});host.replaceChildren(svg);return {svg,w,h}}
function bars(){let {svg:s,w,h}=setup('grade'),L=38,R=8,T=8,B=32,pw=w-L-R,ph=h-T-B,m=Math.max(...D.gradeRates)*1.15;for(let j=0;j<=4;j++){let y=T+ph*j/4;s.append(el('line',{x1:L,y1:y,x2:w-R,y2:y,class:'gridline'}));s.append(el('text',{x:0,y:y+4,class:'axislabel'},`${(m*(1-j/4)).toFixed(0)}%`))}let gap=pw/D.gradeLabels.length,bw=gap*.52;D.gradeRates.forEach((v,i)=>{let hh=v/m*ph,x=L+gap*i+(gap-bw)/2,r=el('rect',{x,y:T+ph-hh,width:bw,height:hh,rx:4,fill:'#54e0bf'});r.append(el('title',{},`${D.gradeLabels[i]}: ${v.toFixed(1)}%`));s.append(r);s.append(el('text',{x:L+gap*(i+.5),y:h-9,'text-anchor':'middle',class:'axislabel'},D.gradeLabels[i]))})}
function vintage(){let {svg:s,w,h}=setup('vintage'),L=40,R=8,T=8,B=30,pw=w-L-R,ph=h-T-B,m=Math.ceil(Math.max(...D.vintageRates)/5)*5;for(let j=0;j<=4;j++){let y=T+ph*j/4;s.append(el('line',{x1:L,y1:y,x2:w-R,y2:y,class:'gridline'}));s.append(el('text',{x:0,y:y+4,class:'axislabel'},`${(m*(1-j/4)).toFixed(0)}%`))}let pts=D.vintageRates.map((v,i)=>[L+i*pw/(D.vintageRates.length-1),T+ph-v/m*ph]);s.append(el('polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:'#82aaff','stroke-width':2.5}));pts.forEach((p,i)=>{let c=el('circle',{cx:p[0],cy:p[1],r:3,fill:'#82aaff'});c.append(el('title',{},`${D.vintageLabels[i]}: ${D.vintageRates[i].toFixed(1)}%`));s.append(c);if(i%6===0||i===pts.length-1)s.append(el('text',{x:p[0],y:h-8,'text-anchor':'middle',class:'axislabel'},D.vintageLabels[i]))})}
function calibration(){let {svg:s,w,h}=setup('cal'),L=42,R=8,T=8,B=29,pw=w-L-R,ph=h-T-B,m=Math.ceil(Math.max(...D.calPred,...D.calActual)/10)*10;for(let j=0;j<=4;j++){let y=T+ph*j/4;s.append(el('line',{x1:L,y1:y,x2:w-R,y2:y,class:'gridline'}));s.append(el('text',{x:0,y:y+4,class:'axislabel'},`${(m*(1-j/4)).toFixed(0)}%`))}s.append(el('line',{x1:L,y1:T+ph,x2:L+pw,y2:T,stroke:'#ffc36b','stroke-dasharray':'5 5'}));D.calActual.forEach((v,i)=>{let x=L+D.calPred[i]/m*pw,y=T+ph-v/m*ph,c=el('circle',{cx:x,cy:y,r:4,fill:'#54e0bf'});c.append(el('title',{},`Decile ${i+1} · predicted ${D.calPred[i].toFixed(1)}% · observed ${v.toFixed(1)}%`));s.append(c)});s.append(el('text',{x:L+pw/2,y:h-7,'text-anchor':'middle',class:'axislabel'},'Predicted default rate →'))}
function draw(){bars();vintage();calibration()}draw();window.addEventListener('resize',()=>{clearTimeout(window.rt);window.rt=setTimeout(draw,120)});
</script></body></html>'''
    replacements={'__KPIS__':kpis,'__GRADE__':grade_html,'__TERM__':term_html,'__BANDS__':band_html,'__CAL__':cal_html,'__CUTOFFS__':cutoff_html,'__COEF__':coef_html,'__SCORE__':score_html,
        '__TRAIN_END__':str(metrics.get('train_end','')),'__TEST_START__':str(metrics.get('test_start','')),'__TEST_END__':str(metrics.get('test_end','')),'__DATA__':chart_data}
    for key,value in replacements.items(): html=html.replace(key,value)
    (OUT/'dashboard.html').write_text(html,encoding='utf-8')

    # Refresh the summary with expanded diagnostic metrics and clear boundaries.
    report=f'''# Credit Risk & Loan Performance — Findings

> **Synthetic case study:** all {len(df):,} loans and outcomes are generated data. No lender or customer records are used.

## Portfolio snapshot

- Originations: **${orig:,.0f}** across {len(df):,} loans.
- Simulated 12-month 90+ DPD incidence: **{df.default_90dpd_12m.mean():.2%}**.
- Simulated net charge-offs: **${netloss:,.0f}** after recoveries.
- Grade A to E default incidence: **{grade.default_rate.iloc[0]:.1%} to {grade.default_rate.iloc[-1]:.1%}**.

## Out-of-time validation

Regularized logistic regression on nine application-time features. Split by origination date, with training through **{metrics.get('train_end')}** and holdout **{metrics.get('test_start')}–{metrics.get('test_end')}** ({len(test):,} loans). Holdout ROC-AUC **{metrics['roc_auc']:.3f}**, Gini **{metrics['gini']:.3f}**, KS **{metrics['ks']:.3f}**, average precision **{metrics['average_precision']:.3f}** versus {metrics['test_default_rate']:.1%} prevalence, and Brier score **{metrics['brier_score']:.3f}**.

At an illustrative 50% score threshold, precision is {metrics['precision']:.1%} and recall is {metrics['recall']:.1%}; see `model_metrics.csv` for additional thresholds. These are diagnostics, not lending cutoffs. Review `calibration_deciles.csv` and `validation_risk_bands.csv` with ranking metrics.

## Monitoring and interpretation

- Compare grade-level default incidence, pricing, gross charge-offs, recoveries, and net loss rates together.
- Monthly vintage performance is noisy for smaller cohorts. Every synthetic loan is stipulated to have a full 12-month window; real analyses need seasoning and censoring controls.
- `feature_stability_psi.csv` is a screening aid, not a universal pass/fail rule. Interpret shifts using domain and data-lineage context.

## Reproduction and limitations

Run `python src/credit_risk.py` from the project root. The deterministic seed regenerates the dataset and all `outputs/` artifacts. Outcome relationships are deliberately designed, so model performance measures recovery of that signal, not predictive power on real borrowers. The project omits macroeconomic scenarios, censoring, prepayment, fraud, selection effects, servicing behavior, and decision economics. No protected attributes are generated; that does not establish fairness. This is not an underwriting tool, policy recommendation, accounting estimate, or regulatory-compliance claim. Real use requires representative data, independent validation, legal and subgroup review, decision-cost analysis, governance, explainability, and monitoring.
'''
    (OUT/'findings.md').write_text(report,encoding='utf-8')

