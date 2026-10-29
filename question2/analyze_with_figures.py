from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve

# This script is intended to replace D:\bigdadta\question2\analyze.py
R = Path(__file__).resolve().parents[1]
if not (R / '吉隆县历史灾害_干净版.csv').exists():
    # Also works when downloaded elsewhere and executed with project root as cwd.
    R = Path.cwd()
O = R / 'question2'
O.mkdir(parents=True, exist_ok=True)

TRAIN_END = '2021-07-31'
TEST_START = '2021-08-16'  # 15-day embargo, Aug 1–15
STUDY_START, STUDY_END = '2021-04-01', '2021-09-30'
BASE_COLS = ['P24', 'P360', 'D72']
COMPARE_COLS = {'P24_D72': ['P24', 'D72'], 'P72_D72': ['P72', 'D72'],
                'P360_D72': ['P360', 'D72']}


def safe_float(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def safe_metric(y, score, kind):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score) if kind == 'auc' else average_precision_score(y, score))


def fit_and_evaluate(train, test, cols, name):
    if train.empty or test.empty or train['y'].nunique() < 2:
        return {'name': name, 'features': cols, 'status': '数据不足，未能估计'}, None
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
    model.fit(train[cols], train['y'])
    tr_score = model.predict_proba(train[cols])[:, 1]
    fpr, tpr, thresholds = roc_curve(train['y'], tr_score)
    # sklearn includes an infinite threshold; prefer a finite threshold if tied.
    youden = tpr - fpr
    finite = np.isfinite(thresholds)
    candidate = np.flatnonzero(finite)
    if len(candidate) == 0:
        return {'name': name, 'features': cols, 'status': '训练阈值不可识别'}, None
    best = candidate[np.argmax(youden[candidate])]
    cut = float(thresholds[best])
    scores = model.predict_proba(test[cols])[:, 1]
    alert = scores >= cut
    tn, fp, fn, tp = confusion_matrix(test['y'], alert, labels=[0, 1]).ravel()
    scaler, clf = model.steps[0][1], model.steps[1][1]
    result = {
        'name': name, 'status': 'ok', 'features': cols,
        'train_n': int(len(train)), 'train_positive_days': int(train['y'].sum()),
        'test_n': int(len(test)), 'test_positive_days': int(test['y'].sum()),
        'threshold_from_training': cut,
        'test_auc': safe_metric(test['y'], scores, 'auc'),
        'test_ap': safe_metric(test['y'], scores, 'ap'),
        'test_prevalence': float(test['y'].mean()),
        'test_confusion_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
        'test_event_recall': float(tp / (tp + fn)) if tp + fn else None,
        'test_background_alarm_rate': float(fp / (fp + tn)) if fp + tn else None,
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'coefficients_standardized': clf.coef_[0].tolist(),
        'intercept_standardized': float(clf.intercept_[0]),
        'note': '模型输出为病例—背景条件评分，不是校准后的真实灾害概率。'
    }
    daily = pd.DataFrame({'date': test.index.strftime('%Y-%m-%d'),
                          'y': test['y'].to_numpy(), 'score': scores,
                          'alert': alert.astype(int)})
    daily.to_csv(O / f'temporal_holdout_{name}.csv', index=False, encoding='utf-8-sig')
    return result, model


def glacier_diagnostics():
    path = R / 'glacier.nc'
    result = {'file': str(path), 'status': 'not_found',
              'interpretation': '全文件统计不能替代已确认冰崩释放区统计，也不能用作2021年事件标签。'}
    if not path.exists():
        return result
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            result['status'] = 'ok'
            result['dataset_dimensions'] = {str(k): int(v) for k, v in ds.sizes.items()}
            result['variables'] = {}
            for name in ['thickness', 'v', 'slope']:
                if name not in ds:
                    result['variables'][name] = {'status': 'missing'}
                    continue
                da = ds[name]
                # Preserve file-defined missing value masks where xarray decodes them.
                arr = np.asarray(da.values, dtype=float)
                arr[~np.isfinite(arr)] = np.nan
                for attr in ['_FillValue', 'missing_value']:
                    val = da.attrs.get(attr)
                    if val is not None and np.isscalar(val):
                        arr[arr == float(val)] = np.nan
                # This is an explicit heuristic, not instrument quality control.
                arr[arr <= -999] = np.nan
                if name in ('thickness', 'v', 'slope'):
                    arr[arr < 0] = np.nan
                valid = arr[np.isfinite(arr)]
                result['variables'][name] = {
                    'status': 'ok' if valid.size else 'no_valid_cells',
                    'valid_cells': int(valid.size), 'units_attribute': da.attrs.get('units'),
                    'min': safe_float(np.min(valid)) if valid.size else None,
                    'p10': safe_float(np.percentile(valid, 10)) if valid.size else None,
                    'median': safe_float(np.median(valid)) if valid.size else None,
                    'p90': safe_float(np.percentile(valid, 90)) if valid.size else None,
                    'max': safe_float(np.max(valid)) if valid.size else None,
                }
    except Exception as ex:
        result['status'] = 'error'
        result['error'] = f'{type(ex).__name__}: {ex}'
    return result


def main():
    event_path = R / '吉隆县历史灾害_干净版.csv'
    if not event_path.exists():
        raise FileNotFoundError(f'未找到灾害台账: {event_path}')
    events = pd.read_csv(event_path)
    events['date'] = pd.to_datetime(events['发生日期'], errors='coerce').dt.normalize()
    events = events.dropna(subset=['date'])
    met_files = sorted((R / 'data').glob('MET_*.csv'))
    if not met_files:
        raise FileNotFoundError(f'未找到气象文件: {R / "data" / "MET_*.csv"}')
    met = pd.concat([pd.read_csv(p) for p in met_files], ignore_index=True)
    met.index = pd.to_datetime(met.pop('Timestamp'), format='mixed', errors='coerce')
    met = met.loc[met.index.notna()].sort_index()
    if met.index.duplicated().any():
        raise ValueError('MET 时间戳存在重复，需先核对重复来源，不能静默去重。')
    p = pd.to_numeric(met['Precip(mm)'], errors='coerce')
    t = pd.to_numeric(met['Ta_2m(degree C)'], errors='coerce')
    p = p.where((p >= 0) & (p < 999))
    t = t.where((t > -80) & (t < 60))
    h = pd.DataFrame({'P': p, 'T': t}).asfreq('h')
    features = pd.DataFrame(index=h.index)
    for n in [24, 72, 360]:
        features[f'P{n}'] = h['P'].rolling(n, min_periods=n).sum().shift(1)
    features['D72'] = h['T'].clip(lower=0).rolling(72, min_periods=72).sum().shift(1) / 24
    # Only midnight predictions; no weather observations from the event day.
    daily = features.loc[features.index.hour == 0].copy()
    event_days = pd.DatetimeIndex(events['date'].unique())
    daily['y'] = daily.index.normalize().isin(event_days).astype(int)
    daily.to_csv(O / 'daily_features.csv', encoding='utf-8-sig')
    audit = {
        'records': int(len(events)),
        'records_by_year': {str(k): int(v) for k, v in events['date'].dt.year.value_counts().sort_index().items()},
        'unique_event_days': int(events['date'].nunique()),
        'events_2021': int(events.loc[events['date'].dt.year == 2021, 'date'].nunique()),
        'timestamp_assumption': '保留原始shift(1)：假设Timestamp为小时开始标签；若为小时结束标签需按元数据重新核对。',
        'qc_note': '仅执行简单数值范围过滤；不插值。气象仪器QC、时区和小时累计语义尚待核验。',
        'glacier_gate_in_dynamic_model': False,
    }
    v = daily.loc[STUDY_START:STUDY_END].copy()
    audit['missing_event_features'] = [str(x.date()) for x in v.index[(v.y == 1) & v[BASE_COLS].isna().any(axis=1)]]
    # Compare all models on the same complete-case dates, to keep metrics comparable.
    common_cols = sorted(set(BASE_COLS).union(*[set(c) for c in COMPARE_COLS.values()]))
    v = v.dropna(subset=common_cols)
    v.to_csv(O / 'model_samples.csv', encoding='utf-8-sig')
    train = v.loc[:TRAIN_END].copy()
    test = v.loc[TEST_START:].copy()
    if train.empty or test.empty:
        raise ValueError('训练集或测试集为空，请检查时间覆盖和气象缺测。')
    audit['split'] = {'train_end': TRAIN_END, 'test_start': TEST_START,
                      'embargo': '2021-08-01 through 2021-08-15',
                      'common_complete_case_features': common_cols}
    audit['train_n'] = int(len(train))
    audit['train_positive_days'] = int(train.y.sum())
    audit['test_n'] = int(len(test))
    audit['test_positive_days'] = int(test.y.sum())
    baseline, baseline_model = fit_and_evaluate(train, test, BASE_COLS, 'baseline_P24_P360_D72')
    audit['baseline'] = baseline
    audit['comparisons'] = {}
    for name, cols in COMPARE_COLS.items():
        result, _ = fit_and_evaluate(train, test, cols, name)
        audit['comparisons'][name] = result
    # Keep the original filename for the baseline holdout as well.
    source = O / 'temporal_holdout_baseline_P24_P360_D72.csv'
    if source.exists():
        (O / 'temporal_holdout.csv').write_bytes(source.read_bytes())
    if baseline_model is not None:
        case_date = pd.Timestamp('2026-08-26')
        if case_date in daily.index:
            case = daily.loc[case_date, BASE_COLS]
            audit['case_2026_features'] = {k: safe_float(v) for k, v in case.items()}
            if case.notna().all():
                score = baseline_model.predict_proba(pd.DataFrame([case], columns=BASE_COLS))[:, 1][0]
                audit['case_2026_score'] = float(score)
                audit['case_2026_alert'] = bool(score >= baseline['threshold_from_training'])
        audit['case_2026_note'] = '2026年案例仅用于外部气象条件诊断；未参与训练，也不验证冰崩预测。'
    audit['glacier_diagnostics'] = glacier_diagnostics()
    positive = v.loc[v.y == 1, BASE_COLS]
    if not positive.empty:
        positive.describe(percentiles=[0.1, 0.5, 0.9]).to_csv(O / 'event_feature_distribution.csv', encoding='utf-8-sig')
    (O / 'results_modified.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    rows = []
    for result in [baseline, *audit['comparisons'].values()]:
        rows.append({k: result.get(k) for k in ['name', 'status', 'train_n', 'train_positive_days',
                    'test_n', 'test_positive_days', 'threshold_from_training', 'test_auc',
                    'test_ap', 'test_event_recall', 'test_background_alarm_rate']})
    pd.DataFrame(rows).to_csv(O / 'model_comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print('\n模型对比表：', O / 'model_comparison.csv')
    print('完整结果：', O / 'results_modified.json')
    create_figures(events, h, daily, v, train, test, audit)


# ==================== 论文图件：全部来自实际数据 ====================
def create_figures(events, hourly, daily, samples, train, test, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve
    import matplotlib.dates as mdates

    font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC', 'DejaVu Sans']
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.sans-serif'] = [f for f in font_candidates if f in available] or ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 240
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.22
    out = O / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'{name}.{ext}', bbox_inches='tight')
        plt.close(fig)
        manifest.append(name)

    # 01: 原始记录年度数量
    years = events.date.dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(years.index.astype(str), years.values, color='#4078a8')
    ax.bar_label(bars); ax.set(xlabel='年份', ylabel='台账记录数', title='历史灾害台账年度记录数（不是独立事件日）')
    save(fig, '01_历史灾害记录年度分布')

    # 02: 独立事件日月度分布
    ev_days = pd.DatetimeIndex(events.date.drop_duplicates().sort_values())
    ev21 = ev_days[ev_days.year == 2021]
    counts = pd.Series(ev21.month).value_counts().reindex(range(1,13), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(counts.index, counts.values, color='#e69b42')
    ax.bar_label(bars); ax.set(xticks=range(1,13), xlabel='月份', ylabel='独立记录事件日数', title='2021年记录事件日月度分布')
    save(fig, '02_2021事件日月度分布')

    # 03: 2021气象特征与记录日（双轴）
    dd = daily.loc[STUDY_START:STUDY_END]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(dd.index, dd.P72, color='#2a77b8', lw=1.3, label='P72（mm）')
    ax.set_ylabel('72小时累计降雨（mm）', color='#2a77b8')
    ax2 = ax.twinx(); ax2.plot(dd.index, dd.D72, color='#df7847', lw=1.2, alpha=.8, label='D72（℃·日）')
    ax2.set_ylabel('72小时正积温（℃·日）', color='#df7847')
    for dt in ev21:
        if dd.index.min() <= dt <= dd.index.max(): ax.axvline(dt, color='#b32929', alpha=.3, lw=.8)
    ax.set_title('2021年气象特征与记录事件日（竖线；非因果关联）')
    save(fig, '03_2021气象特征时间序列')

    # 04: 气象变量分布（背景日/事件日）
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, col in zip(axes, ['P24', 'P72', 'P360', 'D72']):
        a = samples.loc[samples.y == 0, col].dropna().to_numpy()
        b = samples.loc[samples.y == 1, col].dropna().to_numpy()
        ax.boxplot([a,b], tick_labels=['背景日','记录日'], showfliers=False)
        ax.set_title(col); ax.set_ylabel('℃·日' if col=='D72' else 'mm')
    fig.suptitle('气象特征分布（背景日不等于确认无灾日）')
    save(fig, '04_事件背景特征箱线图')

    # 05: 特征相关矩阵
    cols = ['P24','P72','P360','D72']
    corr = samples[cols].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(corr, vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(4), cols); ax.set_yticks(range(4), cols)
    for i in range(4):
        for j in range(4): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=10)
    fig.colorbar(im,ax=ax,label='Pearson相关系数'); ax.set_title('2021年有效日期气象特征相关性')
    save(fig, '05_特征相关矩阵')

    # Read holdout outputs, no refitting or fabricated points
    names = ['baseline_P24_P360_D72','P24_D72','P72_D72','P360_D72']
    labels = ['原始模型','P24+D72','P72+D72','P360+D72']
    holdouts = {}
    for name in names:
        path = O / f'temporal_holdout_{name}.csv'
        if path.exists(): holdouts[name] = pd.read_csv(path)

    # 06: ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, label in zip(names, labels):
        if name not in holdouts: continue
        h = holdouts[name]
        if h.y.nunique() != 2: continue
        fpr,tpr,_ = roc_curve(h.y,h.score)
        ax.plot(fpr,tpr,lw=1.8,label=f'{label} (AUC={roc_auc_score(h.y,h.score):.3f})')
    ax.plot([0,1],[0,1],'k--',alpha=.5)
    ax.set(xlabel='背景日报警比例（FPR）',ylabel='记录事件日召回率（TPR）',title='时间留出集ROC（仅3个记录事件日）')
    ax.legend(loc='lower right'); save(fig, '06_时间留出ROC对比')

    # 07: PR
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, label in zip(names, labels):
        if name not in holdouts: continue
        h=holdouts[name]
        precision,recall,_=precision_recall_curve(h.y,h.score)
        ax.plot(recall,precision,lw=1.8,label=f'{label} (AP={average_precision_score(h.y,h.score):.3f})')
    ax.axhline(test.y.mean(),ls='--',color='gray',label=f'测试记录率={test.y.mean():.3f}')
    ax.set(xlabel='召回率',ylabel='精确率',title='时间留出集PR曲线（背景标签未核实）')
    ax.legend(); save(fig, '07_时间留出PR对比')

    # 08: Metrics
    all_res = [audit['baseline'], *audit['comparisons'].values()]
    fields = [('test_auc','AUC'),('test_ap','AP'),('test_event_recall','事件召回率'),('test_background_alarm_rate','背景日报警比例')]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(key,title) in zip(axes.flat,fields):
        values=[r.get(key) or 0 for r in all_res]
        bars=ax.bar(range(4),values,color=['#688cb1','#76a5af','#d7a35c','#aaa3ba'])
        ax.set_xticks(range(4),labels,rotation=15); ax.set_ylim(0,1);ax.set_title(title)
        ax.bar_label(bars,fmt='%.3f',padding=2)
    fig.suptitle('相同训练测试日期上的模型对照（探索性比较）')
    save(fig, '08_四模型指标对比')

    # 09: P72-D72 scatter and trained decision boundary, reconstituted from saved coefficients
    result=audit['comparisons']['P72_D72']
    mu=np.asarray(result['scaler_mean']); scale=np.asarray(result['scaler_scale'])
    coef=np.asarray(result['coefficients_standardized']); intercept=result['intercept_standardized']
    cut=result['threshold_from_training']; logit=np.log(cut/(1-cut))
    fig,ax=plt.subplots(figsize=(8,6))
    for y,label,color in [(0,'背景日','#5287b5'),(1,'记录事件日','#cc4d47')]:
        q=samples.loc[samples.y==y]
        ax.scatter(q.P72,q.D72,s=26 if y==0 else 55,c=color,alpha=.6 if y==0 else .9,label=label)
    xmin,xmax=max(0,float(samples.P72.min())),float(samples.P72.max())
    xs=np.linspace(xmin,xmax,200)
    if abs(coef[1])>1e-10:
        ys=mu[1]+scale[1]/coef[1]*(logit-intercept-coef[0]*(xs-mu[0])/scale[0])
        ax.plot(xs,ys,'k--',lw=2,label=f'训练期评分分界 s={cut:.3f}')
    ax.set(xlabel='P72（mm）',ylabel='D72（℃·日）',title='P72+D72经验筛查边界（不是物理失稳阈值）')
    ax.legend(); save(fig, '09_P72_D72散点与经验临界线')

    # 10: P72-D72 continuous score surface (not calibrated probability)
    fig,ax=plt.subplots(figsize=(8,6))
    xx=np.linspace(0,max(1,float(samples.P72.max())*1.05),160)
    yy=np.linspace(float(samples.D72.min())-2,float(samples.D72.max())+2,160)
    X,Y=np.meshgrid(xx,yy)
    eta=intercept+coef[0]*(X-mu[0])/scale[0]+coef[1]*(Y-mu[1])/scale[1]
    score=1/(1+np.exp(-np.clip(eta,-700,700)))
    im=ax.contourf(X,Y,score,levels=np.linspace(0,1,21),cmap='RdYlBu_r',vmin=0,vmax=1)
    if score.min()<=cut<=score.max():ax.contour(X,Y,score,levels=[cut],colors='black',linewidths=2)
    for y,color in [(0,'#1c5385'),(1,'#c22c27')]:
        q=samples.loc[samples.y==y];ax.scatter(q.P72,q.D72,s=12 if y==0 else 42,c=color,edgecolors='white',linewidths=.3)
    fig.colorbar(im,ax=ax,label='模型评分（非发生概率）')
    ax.set(xlabel='P72（mm）',ylabel='D72（℃·日）',title='P72+D72评分曲面与训练分界')
    save(fig, '10_P72_D72模型评分曲面')

    # 11: chronological test scores
    fig,ax=plt.subplots(figsize=(12,4.5))
    for name,label in [('baseline_P24_P360_D72','原始模型'),('P72_D72','P72+D72')]:
        if name not in holdouts:continue
        q=holdouts[name]; dates=pd.to_datetime(q.date)
        ax.plot(dates,q.score,marker='.',ms=4,lw=1.1,label=label)
    q=holdouts.get('P72_D72')
    if q is not None:
        ax.scatter(pd.to_datetime(q.loc[q.y==1,'date']),q.loc[q.y==1,'score'],marker='*',s=140,color='#c22c27',label='记录事件日',zorder=5)
        ax.axhline(cut,color='black',ls='--',label='P72训练期阈值')
    ax.set(ylabel='模型评分',title='2021年测试期逐日评分与事件日期（条件回溯）')
    ax.legend();save(fig,'11_测试期逐日评分')

    # 12: 2026 case prior 360h meteorology; timestamp is midnight and raw-hour axis
    target=pd.Timestamp('2026-08-26')
    if target in daily.index:
        hcase=hourly.loc[target-pd.Timedelta(hours=360):target-pd.Timedelta(hours=1)]
        if not hcase.empty:
            fig,ax=plt.subplots(figsize=(12,4.5))
            ax.bar(hcase.index,hcase.P,width=.035,color='#4888bb',label='小时降雨')
            ax.set_ylabel('小时降雨（mm）')
            ax2=ax.twinx();ax2.plot(hcase.index,hcase["T"],color='#d66c43',alpha=.8,lw=1,label='小时气温')
            ax2.set_ylabel('台站气温（℃）');ax.set_title('2026-08-26零时之前360小时气象（不代表冰崩源区实测）')
            save(fig,'12_2026案例事前气象序列')

    # 13-14: glacier spatial arrays and histograms, retain original array coordinates only
    path=R/'glacier.nc'
    if path.exists():
        try:
            import xarray as xr
            with xr.open_dataset(path) as ds:
                fig,axes=plt.subplots(1,3,figsize=(15,4.8))
                fig2,axes2=plt.subplots(1,3,figsize=(15,4.8))
                for ax,ax2,name,cn in zip(axes,axes2,['thickness','v','slope'],['冰厚（m）','冰流速度（m/年）','坡度（°）']):
                    if name not in ds:ax.set_visible(False);ax2.set_visible(False);continue
                    da=ds[name]
                    arr=np.array(da.values,dtype=float,copy=True)
                    arr[~np.isfinite(arr)]=np.nan
                    for key in ['_FillValue','missing_value']:
                        val=da.attrs.get(key)
                        if val is not None and np.isscalar(val):arr[arr==float(val)]=np.nan
                    arr[(arr<=-999)|(arr<0)]=np.nan
                    if arr.ndim!=2:ax.text(.1,.5,'非二维数据，跳过');ax2.set_visible(False);continue
                    im=ax.imshow(arr,origin='lower',cmap='viridis')
                    ax.set_title(cn+'（原始栅格坐标）');ax.set_xlabel('列');ax.set_ylabel('行')
                    fig.colorbar(im,ax=ax,shrink=.75)
                    valid=arr[np.isfinite(arr)]
                    if valid.size:ax2.hist(valid,bins=35,color='#4c88b5',alpha=.85)
                    ax2.set(xlabel=cn,ylabel='像元数量',title=f'{cn}分布（n={len(valid)}）')
                fig.suptitle('glacier.nc空间分布：非已确认冰崩释放区')
                fig2.suptitle('glacier.nc全文件有效像元统计：不同变量掩膜可能不同')
                save(fig,'13_冰川三变量空间栅格')
                save(fig2,'14_冰川三变量直方图')
        except Exception as exc:
            warnings.warn(f'冰川图件绘制失败：{exc}')

    # 15: normalized feature coefficient plot for each model
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,res,label in zip(axes.flat,all_res,labels):
        co=np.asarray(res['coefficients_standardized']); feat=res['features']
        ax.barh(feat,co,color=['#4b82b3' if x>=0 else '#d65c56' for x in co]);ax.axvline(0,color='black',lw=.8)
        ax.set_title(label);ax.set_xlabel('标准化空间回归系数')
    fig.suptitle('模型系数对照（不同特征组合，非因果效应）')
    save(fig,'15_标准化回归系数对比')

    (out/'图件目录.txt').write_text('\n'.join(f'{i:02d}. {n}.png / .pdf' for i,n in enumerate(manifest,1)),encoding='utf-8')
    print(f'\n已保存 {len(manifest)} 组真实数据图件（PNG+PDF）至：{out}')

if __name__ == '__main__':
    main()
