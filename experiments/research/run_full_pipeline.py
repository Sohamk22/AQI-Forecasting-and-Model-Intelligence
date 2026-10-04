#!/usr/bin/env python3
"""
AQI Research Pipeline: Heterogeneous Stacked Ensemble Evaluation
Multi-city, strictly causal, temporal OOF stacking
"""
import sys, os, time, json, pickle, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Setup paths BEFORE other imports
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
import pandas as pd
import torch
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.xgboost_model import XGBoostModel
from src.models.extra_trees_model import ExtraTreesModel
from src.models.bilstm_model import BiLSTMModel
from src.models.bigru_model import BiGRUModel
from src.models.tcn_model import TCNModel

torch.set_num_threads(2)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings('ignore')

DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_INTERIM = PROJECT_ROOT / 'data' / 'interim'
DATA_PROC = PROJECT_ROOT / 'data' / 'processed'
RES = PROJECT_ROOT / 'results' / 'research'
for d in [DATA_INTERIM, DATA_PROC, RES/'metrics', RES/'predictions', RES/'models', RES/'figures']:
    d.mkdir(parents=True, exist_ok=True)

CITY_COORDS = {
    'Delhi': (28.6139, 77.2090), 'Bengaluru': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707), 'Hyderabad': (17.3850, 78.4867),
    'Lucknow': (26.8467, 80.9462), 'Patna': (25.6093, 85.1376),
    'Gurugram': (28.4595, 77.0266), 'Ahmedabad': (23.0225, 72.5714),
    'Mumbai': (19.0760, 72.8777), 'Kolkata': (22.5726, 88.3639),
    'Visakhapatnam': (17.6868, 83.2185), 'Amritsar': (31.6340, 74.8723),
    'Jaipur': (26.9124, 75.7873),
}

XGB_P = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.04,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': SEED, 'n_jobs': 2}
ET_P = {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'random_state': SEED, 'n_jobs': 2}
RNN_P = {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.2, 'learning_rate': 0.001,
         'batch_size': 32, 'epochs': 30, 'patience': 6}
TCN_P = {'num_channels': [32, 32, 64, 64], 'kernel_size': 3, 'dropout': 0.2,
         'learning_rate': 0.001, 'batch_size': 32, 'epochs': 30, 'patience': 6}
META_P = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5, 'random_state': SEED, 'n_jobs': 2}

SEQ_LEN = 14
SEQ_COLS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed', 'AQI']
POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
WEATHER_COLS = ['Temperature', 'Humidity', 'Wind_Speed']
FEATURE_COLS = POLLUTANT_COLS + WEATHER_COLS

# ============================================================
# PHASE 1: City Selection & Weather
# ============================================================

def select_cities(df: pd.DataFrame) -> List[str]:
    t0 = time.time()
    print("\n=== PHASE 1: City Selection ===")
    rows = []
    for city, g in df.groupby('City'):
        aqi_v = int(g['AQI'].notna().sum())
        rows.append({'City': city, 'total': len(g), 'aqi_valid': aqi_v,
                     'aqi_pct': aqi_v/len(g), 'start': g['Date'].min(), 'end': g['Date'].max()})
    summary = pd.DataFrame(rows)
    sel = summary[(summary['aqi_valid'] >= 1200) & (summary['aqi_pct'] >= 0.70) &
                  (summary['City'].isin(CITY_COORDS))].sort_values('aqi_valid', ascending=False)
    print(sel.to_string(index=False))
    cities = sel['City'].tolist()
    print(f"Selected {len(cities)} cities in {time.time()-t0:.1f}s")
    return cities

def download_weather(city: str, lat: float, lon: float) -> pd.DataFrame:
    if city == 'Delhi':
        p = DATA_RAW / 'weather_daily.csv'
        if p.exists():
            df = pd.read_csv(p); df['Date'] = pd.to_datetime(df['Date']); return df
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date=2015-01-01&end_date=2020-07-01"
           f"&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean"
           f"&timezone=Asia/Kolkata")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60); r.raise_for_status(); d = r.json()['daily']
            return pd.DataFrame({'Date': pd.to_datetime(d['time']),
                                 'Temperature': d['temperature_2m_mean'],
                                 'Humidity': d['relative_humidity_2m_mean'],
                                 'Wind_Speed': d['wind_speed_10m_mean']})
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {city}: {e}"); time.sleep(3)
    return pd.DataFrame()

def get_weather(cities: List[str]) -> pd.DataFrame:
    cache = DATA_INTERIM / 'weather_all_cities.csv'
    if cache.exists():
        df = pd.read_csv(cache); df['Date'] = pd.to_datetime(df['Date'])
        cached_cities = set(df['City'].unique())
        if set(cities).issubset(cached_cities):
            print(f"Loaded cached weather for {len(cached_cities)} cities")
            return df[df['City'].isin(cities)]
    print("Downloading per-city ERA5 weather...")
    parts = []
    for c in cities:
        lat, lon = CITY_COORDS[c]
        print(f"  {c} ({lat}, {lon})...", end=' ')
        w = download_weather(c, lat, lon)
        if w.empty:
            print("FAILED"); continue
        w['City'] = c; parts.append(w); print(f"{len(w)} days")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(cache, index=False)
    return df

# ============================================================
# PHASE 2-4: Data Preparation, Split, Imputation
# ============================================================

def prepare_data(cities: List[str], df_poll: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 2: Data Preparation ===")
    df_poll['Date'] = pd.to_datetime(df_poll['Date'])
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    parts = []
    for c in cities:
        p = df_poll[df_poll['City'] == c].copy()
        w = df_weather[df_weather['City'] == c].copy()
        if w.empty:
            print(f"  {c}: NO WEATHER DATA — skipping"); continue
        # handle potential duplicate column from merge
        m = p.merge(w[['Date','Temperature','Humidity','Wind_Speed']], on='Date', how='inner',
                    suffixes=('','_w'))
        for col in WEATHER_COLS:
            if f'{col}_w' in m.columns:
                m[col] = m[f'{col}_w']; m.drop(columns=[f'{col}_w'], inplace=True)
        keep = ['City','Date'] + POLLUTANT_COLS + WEATHER_COLS + ['AQI']
        m = m[[c for c in keep if c in m.columns]]
        m = m.dropna(subset=['AQI']).sort_values('Date').reset_index(drop=True)
        print(f"  {c}: {len(m)} rows  {m['Date'].min().date()} → {m['Date'].max().date()}")
        parts.append(m)
    return pd.concat(parts, ignore_index=True).sort_values(['City','Date']).reset_index(drop=True)

def chrono_split(df: pd.DataFrame):
    print("\n=== PHASE 3: Chronological Split ===")
    tr = df[df['Date'] < '2019-03-01'].copy()
    va = df[(df['Date'] >= '2019-03-01') & (df['Date'] <= '2019-08-31')].copy()
    te = df[df['Date'] >= '2019-09-01'].copy()
    for n, d in [('Train', tr), ('Val', va), ('Test', te)]:
        print(f"  {n}: {len(d)} rows, {d['Date'].min().date()} → {d['Date'].max().date()}, "
              f"{d['City'].nunique()} cities")
    return tr, va, te

def impute(tr, va, te):
    print("\n=== PHASE 4: Imputation ===")
    cols = FEATURE_COLS
    for split in [tr, va, te]:
        for col in cols:
            if col in split.columns:
                split[col] = split.groupby('City')[col].transform(lambda x: x.ffill().bfill())
    # carry forward from train→val, val→test per city
    for prev, cur in [(tr, va), (va, te)]:
        for city in cur['City'].unique():
            prev_city = prev[prev['City'] == city]
            if prev_city.empty: continue
            last = prev_city.iloc[-1]
            for col in cols:
                if col in cur.columns:
                    mask = (cur['City'] == city) & cur[col].isna()
                    cur.loc[mask, col] = last.get(col, np.nan)
    for split in [tr, va, te]:
        for col in cols:
            if col in split.columns:
                split[col] = split.groupby('City')[col].transform(lambda x: x.ffill().bfill())
    crit = ['PM2.5', 'AQI']
    tr = tr.dropna(subset=crit).reset_index(drop=True)
    va = va.dropna(subset=crit).reset_index(drop=True)
    te = te.dropna(subset=crit).reset_index(drop=True)
    print(f"  After imputation: Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    return tr, va, te

# ============================================================
# PHASE 5: Feature Engineering
# ============================================================

def engineer_features(df: pd.DataFrame, all_cities: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Engineer ~70 tabular features. all_cities ensures consistent one-hot across splits."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Calendar (11)
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
    season_map = {1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:3,11:3,12:3}
    df['season'] = df['month'].map(season_map).astype(float)
    df['sin_doy'] = np.sin(2*np.pi*df['day_of_year']/365.25)
    df['cos_doy'] = np.cos(2*np.pi*df['day_of_year']/365.25)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12)
    df['sin_dow'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['cos_dow'] = np.cos(2*np.pi*df['day_of_week']/7)

    # City one-hot (consistent across splits)
    for c in sorted(all_cities):
        df[f'city_{c}'] = (df['City'] == c).astype(float)

    # Lags — per city
    g = df.groupby('City')
    # AQI lags
    for lag in [1,2,3,7]:
        df[f'AQI_lag_{lag}'] = g['AQI'].shift(lag)
    # Pollutant + weather lag-1
    for col in POLLUTANT_COLS + WEATHER_COLS:
        df[f'{col}_lag_1'] = g[col].shift(1)
    # PM2.5 and PM10 lag-2
    df['PM2.5_lag_2'] = g['PM2.5'].shift(2)
    df['PM10_lag_2'] = g['PM10'].shift(2)

    # Rolling — per city, on shifted series
    def roll(col, w, stat):
        shifted = g[col].shift(1)
        if stat == 'mean': return shifted.rolling(w, min_periods=1).mean()
        elif stat == 'std': return shifted.rolling(w, min_periods=1).std().fillna(0)
        elif stat == 'min': return shifted.rolling(w, min_periods=1).min()
        elif stat == 'max': return shifted.rolling(w, min_periods=1).max()
        elif stat == 'ewm': return shifted.ewm(span=w, min_periods=1).mean()

    # AQI rolling (9)
    for s in ['mean','std','min','max']:
        df[f'AQI_roll_{s}_7'] = roll('AQI',7,s)
    for s in ['mean','std','min','max']:
        df[f'AQI_roll_{s}_14'] = roll('AQI',14,s)
    df['AQI_ewma_7'] = roll('AQI',7,'ewm')

    # PM2.5 rolling (7)
    for s in ['mean','std','min','max']:
        df[f'PM2.5_roll_{s}_7'] = roll('PM2.5',7,s)
    df['PM2.5_roll_mean_14'] = roll('PM2.5',14,'mean')
    df['PM2.5_roll_std_14'] = roll('PM2.5',14,'std')
    df['PM2.5_ewma_7'] = roll('PM2.5',7,'ewm')

    # Other pollutant rolling (10)
    for col in ['PM10','NO2','SO2','CO','O3']:
        df[f'{col}_roll_mean_7'] = roll(col,7,'mean')
        df[f'{col}_ewma_7'] = roll(col,7,'ewm')

    # Weather rolling (6)
    df['Temp_roll_mean_7'] = roll('Temperature',7,'mean')
    df['Temp_roll_std_7'] = roll('Temperature',7,'std')
    df['Temp_ewma_7'] = roll('Temperature',7,'ewm')
    df['Humidity_roll_mean_7'] = roll('Humidity',7,'mean')
    df['Humidity_ewma_7'] = roll('Humidity',7,'ewm')
    df['WindSpeed_roll_mean_7'] = roll('Wind_Speed',7,'mean')

    # Drop NaN rows from lag/rolling warmup
    df = df.dropna().reset_index(drop=True)

    # Identify tabular columns (everything except raw + meta)
    raw = {'City','Date','AQI'} | set(POLLUTANT_COLS) | set(WEATHER_COLS)
    tab_cols = [c for c in df.columns if c not in raw]
    return df, tab_cols

# ============================================================
# PHASE 6: Scaling & Sequences
# ============================================================

def build_arrays(df_tr, df_va, df_te, tab_cols):
    print("\n=== PHASE 6: Building arrays ===")
    scaler_tab = MinMaxScaler().fit(df_tr[tab_cols])
    scaler_seq = MinMaxScaler().fit(df_tr[SEQ_COLS])

    def make_arrays(df_curr, df_prev=None):
        X_tab, X_seq, Y, Dates, Cities = [], [], [], [], []
        for city in sorted(df_curr['City'].unique()):
            cc = df_curr[df_curr['City'] == city].copy()
            if df_prev is not None:
                cp = df_prev[df_prev['City'] == city]
                if not cp.empty:
                    ctx = cp.iloc[-SEQ_LEN:]
                    cc = pd.concat([ctx, cc]).reset_index(drop=True)
                    start_idx = len(ctx)
                else:
                    start_idx = SEQ_LEN
            else:
                start_idx = SEQ_LEN

            if len(cc) <= SEQ_LEN:
                continue
            tab_s = scaler_tab.transform(cc[tab_cols])
            seq_s = scaler_seq.transform(cc[SEQ_COLS])
            y_raw = cc['AQI'].values
            dates = cc['Date'].values

            for i in range(max(SEQ_LEN, start_idx if df_prev is not None else SEQ_LEN), len(cc)):
                X_seq.append(seq_s[i-SEQ_LEN:i])
                X_tab.append(tab_s[i])
                Y.append(y_raw[i])
                Dates.append(dates[i])
                Cities.append(city)
        return (np.array(X_tab, dtype=np.float32), np.array(X_seq, dtype=np.float32),
                np.array(Y, dtype=np.float32), np.array(Dates), np.array(Cities))

    Xt_tr, Xs_tr, y_tr, d_tr, c_tr = make_arrays(df_tr)
    Xt_va, Xs_va, y_va, d_va, c_va = make_arrays(df_va, df_tr)
    Xt_te, Xs_te, y_te, d_te, c_te = make_arrays(df_te, df_va)

    print(f"  Train: tab={Xt_tr.shape} seq={Xs_tr.shape} y={y_tr.shape}")
    print(f"  Val:   tab={Xt_va.shape} seq={Xs_va.shape} y={y_va.shape}")
    print(f"  Test:  tab={Xt_te.shape} seq={Xs_te.shape} y={y_te.shape}")

    bundle = dict(Xt_tr=Xt_tr, Xs_tr=Xs_tr, y_tr=y_tr, d_tr=d_tr, c_tr=c_tr,
                  Xt_va=Xt_va, Xs_va=Xs_va, y_va=y_va, d_va=d_va, c_va=c_va,
                  Xt_te=Xt_te, Xs_te=Xs_te, y_te=y_te, d_te=d_te, c_te=c_te,
                  tab_cols=tab_cols, scaler_tab=scaler_tab, scaler_seq=scaler_seq)
    with open(DATA_PROC / 'research_bundle.pkl', 'wb') as f:
        pickle.dump(bundle, f)
    return bundle

# ============================================================
# PHASE 7: Standalone Training
# ============================================================

def train_standalones(B):
    print("\n=== PHASE 7: Standalone Models ===")
    preds_te, preds_va, models = {}, {}, {}
    indim = B['Xs_tr'].shape[2]

    t0 = time.time()
    print("  XGBoost...", end=' ', flush=True)
    m = XGBoostModel(params=XGB_P); m.fit(B['Xt_tr'], B['y_tr'])
    preds_va['XGBoost'] = m.predict(B['Xt_va'])
    preds_te['XGBoost'] = m.predict(B['Xt_te'])
    m.save(str(RES/'models'/'xgboost.pkl')); models['XGBoost'] = m
    print(f"{time.time()-t0:.1f}s")

    t0 = time.time(); print("  ExtraTrees...", end=' ', flush=True)
    m = ExtraTreesModel(params=ET_P); m.fit(B['Xt_tr'], B['y_tr'])
    preds_va['ExtraTrees'] = m.predict(B['Xt_va'])
    preds_te['ExtraTrees'] = m.predict(B['Xt_te'])
    m.save(str(RES/'models'/'extra_trees.pkl')); models['ExtraTrees'] = m
    print(f"{time.time()-t0:.1f}s")

    for name, Cls, params in [('BiLSTM', BiLSTMModel, RNN_P),
                               ('BiGRU', BiGRUModel, RNN_P),
                               ('TCN', TCNModel, TCN_P)]:
        t0 = time.time(); print(f"  {name}...", end=' ', flush=True)
        torch.manual_seed(SEED)
        m = Cls(input_dim=indim, params=params)
        m.fit(B['Xs_tr'], B['y_tr'], B['Xs_va'], B['y_va'], verbose=False)
        preds_va[name] = m.predict(B['Xs_va'])
        preds_te[name] = m.predict(B['Xs_te'])
        m.save(str(RES/'models'/f'{name.lower()}.pt')); models[name] = m
        print(f"{time.time()-t0:.1f}s")

    return models, preds_te, preds_va

# ============================================================
# PHASE 8: OOF Stacking
# ============================================================

def generate_oof(B, components: List[str], n_splits=5):
    """Generate OOF predictions for given components. Returns dict of arrays."""
    indim = B['Xs_tr'].shape[2]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = {c: [] for c in components}
    oof_y = []
    oof_idx = []

    for fold, (tr_i, va_i) in enumerate(tscv.split(B['Xt_tr'])):
        print(f"    Fold {fold+1}/{n_splits} (train={len(tr_i)}, val={len(va_i)})...", flush=True)
        yt, yv = B['y_tr'][tr_i], B['y_tr'][va_i]

        for c in components:
            torch.manual_seed(SEED + fold)
            if c == 'XGBoost':
                m = XGBoostModel(params=XGB_P); m.fit(B['Xt_tr'][tr_i], yt)
                oof[c].append(m.predict(B['Xt_tr'][va_i]))
            elif c == 'BiLSTM':
                m = BiLSTMModel(input_dim=indim, params=RNN_P)
                m.fit(B['Xs_tr'][tr_i], yt, B['Xs_tr'][va_i], yv, verbose=False)
                oof[c].append(m.predict(B['Xs_tr'][va_i]))
            elif c == 'BiGRU':
                m = BiGRUModel(input_dim=indim, params=RNN_P)
                m.fit(B['Xs_tr'][tr_i], yt, B['Xs_tr'][va_i], yv, verbose=False)
                oof[c].append(m.predict(B['Xs_tr'][va_i]))
            elif c == 'TCN':
                m = TCNModel(input_dim=indim, params=TCN_P)
                m.fit(B['Xs_tr'][tr_i], yt, B['Xs_tr'][va_i], yv, verbose=False)
                oof[c].append(m.predict(B['Xs_tr'][va_i]))
        oof_y.append(yv)
        oof_idx.append(va_i)

    # Concatenate
    for c in components:
        oof[c] = np.concatenate(oof[c])
    oof_y = np.concatenate(oof_y)
    return oof, oof_y

def run_stacking(B, preds_te):
    print("\n=== PHASE 8: OOF Stacking ===")
    t0 = time.time()

    # Generate OOF for all 4 base models at once
    all_comps = ['XGBoost', 'BiLSTM', 'BiGRU', 'TCN']
    print("  Generating OOF predictions for all components...")
    oof, oof_y = generate_oof(B, all_comps)
    print(f"  OOF done in {time.time()-t0:.1f}s. OOF samples: {len(oof_y)}")

    # SE-1: XGBoost + BiLSTM + TCN
    se1_comps = ['XGBoost', 'BiLSTM', 'TCN']
    Z_oof_se1 = np.column_stack([oof[c] for c in se1_comps])
    meta_se1 = ExtraTreesRegressor(**META_P)
    meta_se1.fit(Z_oof_se1, oof_y)
    Z_te_se1 = np.column_stack([preds_te[c] for c in se1_comps])
    preds_te['SE-1'] = meta_se1.predict(Z_te_se1)
    print(f"  SE-1 meta-learner trained. Feature importances: "
          f"{dict(zip(se1_comps, [f'{v:.3f}' for v in meta_se1.feature_importances_]))}")

    # SE-2: XGBoost + BiGRU + TCN
    se2_comps = ['XGBoost', 'BiGRU', 'TCN']
    Z_oof_se2 = np.column_stack([oof[c] for c in se2_comps])
    meta_se2 = ExtraTreesRegressor(**META_P)
    meta_se2.fit(Z_oof_se2, oof_y)
    Z_te_se2 = np.column_stack([preds_te[c] for c in se2_comps])
    preds_te['SE-2'] = meta_se2.predict(Z_te_se2)
    print(f"  SE-2 meta-learner trained. Feature importances: "
          f"{dict(zip(se2_comps, [f'{v:.3f}' for v in meta_se2.feature_importances_]))}")

    return oof, oof_y, preds_te

# ============================================================
# PHASE 9: Ablations
# ============================================================

def run_ablations(oof, oof_y, preds_te, y_te):
    print("\n=== PHASE 9: Ablation Study ===")

    se1_configs = {
        'H1-A1 (BiLSTM+TCN, no XGB)':    ['BiLSTM', 'TCN'],
        'H1-A2 (XGB+BiLSTM, no TCN)':     ['XGBoost', 'BiLSTM'],
        'H1-A3 (XGB+TCN, no BiLSTM)':     ['XGBoost', 'TCN'],
        'H1-A4 (Full SE-1)':              ['XGBoost', 'BiLSTM', 'TCN'],
    }
    se2_configs = {
        'H2-A1 (BiGRU+TCN, no XGB)':      ['BiGRU', 'TCN'],
        'H2-A2 (XGB+BiGRU, no TCN)':      ['XGBoost', 'BiGRU'],
        'H2-A3 (XGB+TCN, no BiGRU*)':     ['XGBoost', 'TCN'],  # Guide inconsistency: see note
        'H2-A4 (Full SE-2)':              ['XGBoost', 'BiGRU', 'TCN'],
    }

    def run_configs(configs, label):
        results = []
        for name, comps in configs.items():
            Z_oof = np.column_stack([oof[c] for c in comps])
            m = ExtraTreesRegressor(**META_P).fit(Z_oof, oof_y)
            Z_te = np.column_stack([preds_te[c] for c in comps])
            p = m.predict(Z_te)
            metrics = compute_metrics(y_te, p)
            metrics['Config'] = name
            metrics['Components'] = '+'.join(comps)
            results.append(metrics)
            print(f"  {name}: RMSE={metrics['RMSE']:.2f}  R²={metrics['R2']:.4f}")
        return results

    print("  --- Hybrid 1 (SE-1) Ablations ---")
    se1_abl = run_configs(se1_configs, 'SE-1')
    print("\n  --- Hybrid 2 (SE-2) Ablations ---")
    print("  NOTE: H2-A3 guide says 'without BiLSTM' but BiLSTM is not in SE-2.")
    print("        Using parallel interpretation: XGBoost+TCN (without BiGRU).")
    se2_abl = run_configs(se2_configs, 'SE-2')

    return se1_abl, se2_abl

# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    mae = float(np.mean(np.abs(yt-yp)))
    mask = yt > 0
    mape = float(np.mean(np.abs(yt[mask]-yp[mask])/yt[mask])*100) if mask.any() else 0.0
    ss_res = np.sum((yt-yp)**2); ss_tot = np.sum((yt-np.mean(yt))**2)
    r2 = float(1-ss_res/ss_tot) if ss_tot > 0 else 0.0
    return {'RMSE': round(rmse,2), 'MAE': round(mae,2), 'MAPE': round(mape,2), 'R2': round(r2,4)}

# ============================================================
# PHASE 10-13: Evaluate, Save, Plot, Report
# ============================================================

def evaluate_and_report(B, preds_te, se1_abl, se2_abl, all_cities):
    print("\n=== PHASE 10: Evaluation ===")
    y_te, c_te, d_te = B['y_te'], B['c_te'], B['d_te']

    # Main comparison
    main = []
    for name in ['XGBoost','ExtraTrees','BiLSTM','BiGRU','TCN','SE-1','SE-2']:
        if name in preds_te:
            m = compute_metrics(y_te, preds_te[name]); m['Model'] = name; main.append(m)
    df_main = pd.DataFrame(main)
    df_main.to_csv(RES/'metrics'/'main_comparison.csv', index=False)
    print("\nMain Comparison:")
    print(df_main[['Model','RMSE','MAE','MAPE','R2']].to_string(index=False))

    # Per-city
    city_rows = []
    for city in sorted(np.unique(c_te)):
        mask = c_te == city
        if not mask.any(): continue
        for name, p in preds_te.items():
            m = compute_metrics(y_te[mask], p[mask]); m['City'] = city; m['Model'] = name
            city_rows.append(m)
    df_city = pd.DataFrame(city_rows)
    df_city.to_csv(RES/'metrics'/'per_city_breakdown.csv', index=False)

    # Extreme AQI
    ext_mask = y_te > 300
    ext_rows = []
    if ext_mask.any():
        print(f"\nExtreme AQI (>300): {ext_mask.sum()} samples")
        for name, p in preds_te.items():
            m = compute_metrics(y_te[ext_mask], p[ext_mask]); m['Model'] = name
            # underestimation analysis
            under = (p[ext_mask] < y_te[ext_mask])
            m['Underestimate_Rate'] = round(float(under.mean())*100, 1)
            m['Avg_Underestimate'] = round(float(np.mean(y_te[ext_mask][under] - p[ext_mask][under])), 1) if under.any() else 0
            ext_rows.append(m)
        df_ext = pd.DataFrame(ext_rows)
        df_ext.to_csv(RES/'metrics'/'extreme_aqi_analysis.csv', index=False)
        print(df_ext[['Model','RMSE','R2','Underestimate_Rate']].to_string(index=False))
    else:
        print("\nNo extreme AQI (>300) in test set. Trying threshold >200...")
        ext_mask = y_te > 200
        if ext_mask.any():
            for name, p in preds_te.items():
                m = compute_metrics(y_te[ext_mask], p[ext_mask]); m['Model'] = name
                under = (p[ext_mask] < y_te[ext_mask])
                m['Underestimate_Rate'] = round(float(under.mean())*100, 1)
                m['Avg_Underestimate'] = round(float(np.mean(y_te[ext_mask][under] - p[ext_mask][under])), 1) if under.any() else 0
                ext_rows.append(m)
            df_ext = pd.DataFrame(ext_rows)
            df_ext.to_csv(RES/'metrics'/'extreme_aqi_analysis.csv', index=False)

    # Ablation CSVs
    pd.DataFrame(se1_abl).to_csv(RES/'metrics'/'se1_ablation.csv', index=False)
    pd.DataFrame(se2_abl).to_csv(RES/'metrics'/'se2_ablation.csv', index=False)
    print("\nSE-1 Ablation:")
    print(pd.DataFrame(se1_abl)[['Config','RMSE','MAE','MAPE','R2']].to_string(index=False))
    print("\nSE-2 Ablation:")
    print(pd.DataFrame(se2_abl)[['Config','RMSE','MAE','MAPE','R2']].to_string(index=False))

    # Predictions CSV
    df_preds = pd.DataFrame({'Date': d_te, 'City': c_te, 'AQI_actual': y_te})
    for name, p in preds_te.items():
        df_preds[name] = p
    df_preds.to_csv(RES/'predictions'/'all_test_predictions.csv', index=False)

    # ---- FIGURES ----
    print("\n=== PHASE 11: Figures ===")
    sns.set_style('whitegrid')

    # Fig 1: Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = df_main.sort_values('RMSE')['Model'].tolist()
    sns.barplot(data=df_main, x='Model', y='RMSE', order=order, ax=axes[0], palette='viridis')
    axes[0].set_title('Test RMSE by Model'); axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(data=df_main, x='Model', y='R2', order=order, ax=axes[1], palette='viridis')
    axes[1].set_title('Test R² by Model'); axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.savefig(RES/'figures'/'fig_01_model_comparison.png', dpi=300, bbox_inches='tight'); plt.close()

    # Fig 2: Actual vs predicted for Delhi + one other city
    plot_cities = ['Delhi']
    other = [c for c in sorted(np.unique(c_te)) if c != 'Delhi']
    if other: plot_cities.append(other[0])
    fig, axes = plt.subplots(len(plot_cities), 1, figsize=(15, 5*len(plot_cities)))
    if len(plot_cities) == 1: axes = [axes]
    for ax, city in zip(axes, plot_cities):
        mask = c_te == city
        dates = pd.to_datetime(d_te[mask])
        ax.plot(dates, y_te[mask], 'k-', alpha=0.7, label='Actual', linewidth=1.2)
        for mname, color in [('XGBoost','tab:blue'),('SE-1','tab:orange'),('SE-2','tab:green')]:
            if mname in preds_te:
                ax.plot(dates, preds_te[mname][mask], alpha=0.7, label=mname, linewidth=0.9, color=color)
        ax.set_title(f'{city}: Actual vs Predicted AQI'); ax.legend(); ax.set_ylabel('AQI')
    plt.tight_layout(); plt.savefig(RES/'figures'/'fig_02_actual_vs_predicted.png', dpi=300, bbox_inches='tight'); plt.close()

    # Fig 3 & 4: Ablation charts
    for tag, abl_data, title in [('se1', se1_abl, 'Hybrid 1 (SE-1) Ablation'),
                                  ('se2', se2_abl, 'Hybrid 2 (SE-2) Ablation')]:
        df_a = pd.DataFrame(abl_data)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(data=df_a, x='Config', y='RMSE', ax=axes[0], palette='Set2')
        axes[0].set_title(f'{title}: RMSE'); axes[0].tick_params(axis='x', rotation=30)
        sns.barplot(data=df_a, x='Config', y='R2', ax=axes[1], palette='Set2')
        axes[1].set_title(f'{title}: R²'); axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        fig_num = '03' if tag == 'se1' else '04'
        plt.savefig(RES/'figures'/f'fig_{fig_num}_{tag}_ablation.png', dpi=300, bbox_inches='tight'); plt.close()

    # Fig 5: City R² heatmap
    if len(df_city) > 0:
        pivot = df_city.pivot_table(index='City', columns='Model', values='R2')
        plt.figure(figsize=(12, max(6, len(pivot)*0.8)))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5)
        plt.title('R² Score by City and Model')
        plt.tight_layout(); plt.savefig(RES/'figures'/'fig_05_city_heatmap.png', dpi=300, bbox_inches='tight'); plt.close()

    # Fig 6: Extreme AQI scatter
    threshold = 300 if (y_te > 300).any() else 200
    ext_m = y_te > threshold
    if ext_m.any():
        plt.figure(figsize=(8, 8))
        for mname, marker in [('XGBoost','o'),('SE-1','s'),('SE-2','^')]:
            if mname in preds_te:
                plt.scatter(y_te[ext_m], preds_te[mname][ext_m], alpha=0.6, label=mname, marker=marker, s=50)
        lims = [threshold, max(y_te)+20]
        plt.plot(lims, lims, 'r--', alpha=0.5)
        plt.xlabel('Actual AQI'); plt.ylabel('Predicted AQI')
        plt.title(f'Extreme AQI (>{threshold}) Predictions'); plt.legend()
        plt.tight_layout(); plt.savefig(RES/'figures'/'fig_06_extreme_aqi.png', dpi=300, bbox_inches='tight'); plt.close()

    # Fig 7: Residual box plot
    res_data = []
    for mname in ['XGBoost','ExtraTrees','BiLSTM','BiGRU','TCN','SE-1','SE-2']:
        if mname in preds_te:
            res = y_te - preds_te[mname]
            for r in res: res_data.append({'Model': mname, 'Residual': r})
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(res_data), x='Model', y='Residual', palette='Set3')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Prediction Residual Distribution'); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(RES/'figures'/'fig_07_residual_distribution.png', dpi=300, bbox_inches='tight'); plt.close()

    print(f"  Saved 7 figures to {RES/'figures'}")

    # ---- FINAL SUMMARY ----
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    summary_lines = []
    summary_lines.append(f"Cities: {sorted(np.unique(c_te))}")
    summary_lines.append(f"Feature count: {len(B['tab_cols'])} tabular + {len(SEQ_COLS)} sequence channels")
    summary_lines.append(f"Train: {len(B['y_tr'])} samples")
    summary_lines.append(f"Val:   {len(B['y_va'])} samples")
    summary_lines.append(f"Test:  {len(B['y_te'])} samples")
    summary_lines.append("")
    summary_lines.append("MAIN MODEL COMPARISON:")
    summary_lines.append(df_main[['Model','RMSE','MAE','MAPE','R2']].to_string(index=False))
    summary_lines.append("")
    summary_lines.append("SE-1 (Hybrid 1) ABLATION:")
    summary_lines.append(pd.DataFrame(se1_abl)[['Config','RMSE','MAE','MAPE','R2']].to_string(index=False))
    summary_lines.append("")
    summary_lines.append("SE-2 (Hybrid 2) ABLATION:")
    summary_lines.append("NOTE: H2-A3 guide inconsistency — guide says 'without BiLSTM' but BiLSTM")
    summary_lines.append("      is not in SE-2. Used parallel interpretation: without BiGRU.")
    summary_lines.append(pd.DataFrame(se2_abl)[['Config','RMSE','MAE','MAPE','R2']].to_string(index=False))

    summary_text = '\n'.join(summary_lines)
    print(summary_text)
    with open(RES / 'RESULTS_SUMMARY.txt', 'w') as f:
        f.write(summary_text)
    print(f"\nAll results saved to: {RES}")
    print("Pipeline complete.")

# ============================================================
# MAIN
# ============================================================

def main():
    T0 = time.time()
    print("="*70)
    print("AQI RESEARCH PIPELINE — Heterogeneous Stacked Ensemble Evaluation")
    print("="*70)

    # Load raw data
    df_raw = pd.read_csv(DATA_RAW / 'city_day.csv')
    cities = select_cities(df_raw)

    # Weather
    df_weather = get_weather(cities)

    # Prepare
    df_combined = prepare_data(cities, df_raw, df_weather)

    # Split
    df_tr, df_va, df_te = chrono_split(df_combined)

    # Impute
    df_tr, df_va, df_te = impute(df_tr, df_va, df_te)

    # Feature engineering (use consistent city list across all splits)
    all_cities = sorted(set(df_tr['City'].unique()) | set(df_va['City'].unique()) | set(df_te['City'].unique()))
    print(f"\n=== PHASE 5: Feature Engineering ===")
    print(f"  Consistent city list ({len(all_cities)}): {all_cities}")
    df_tr, tab_cols = engineer_features(df_tr, all_cities)
    df_va, _ = engineer_features(df_va, all_cities)
    df_te, _ = engineer_features(df_te, all_cities)
    print(f"  Tabular features: {len(tab_cols)}")
    print(f"  After feature eng: Train={len(df_tr)}, Val={len(df_va)}, Test={len(df_te)}")

    # Build arrays
    B = build_arrays(df_tr, df_va, df_te, tab_cols)

    # Train standalones
    models, preds_te, preds_va = train_standalones(B)

    # OOF Stacking
    oof, oof_y, preds_te = run_stacking(B, preds_te)

    # Ablations
    se1_abl, se2_abl = run_ablations(oof, oof_y, preds_te, B['y_te'])

    # Evaluate & Report
    evaluate_and_report(B, preds_te, se1_abl, se2_abl, all_cities)

    print(f"\nTotal pipeline time: {time.time()-T0:.1f}s")

if __name__ == '__main__':
    main()
