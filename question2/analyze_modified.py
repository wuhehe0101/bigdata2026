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

if __name__ == '__main__':
    main()
