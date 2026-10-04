#!/usr/bin/env python3
"""Phase 7-13: Train, Stack, Ablate, Evaluate — using cached research_bundle.pkl"""
import sys, os, pickle, time, traceback, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(2)
np.random.seed(42); torch.manual_seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from src.models.xgboost_model import XGBoostModel
from src.models.extra_trees_model import ExtraTreesModel
from src.models.bilstm_model import BiLSTMModel
from src.models.bigru_model import BiGRUModel
from src.models.tcn_model import TCNModel

RES = str(PROJECT_ROOT / 'results' / 'research')
SEED = 42

XGB_P = {'n_estimators':200,'max_depth':5,'learning_rate':0.04,'subsample':0.8,
         'colsample_bytree':0.8,'random_state':SEED,'n_jobs':1}
ET_P = {'n_estimators':100,'max_depth':8,'min_samples_split':5,'random_state':SEED,'n_jobs':1}
RNN_P = {'hidden_dim':64,'num_layers':1,'dropout':0.2,'learning_rate':0.001,
         'batch_size':32,'epochs':30,'patience':6}
TCN_P = {'num_channels':[32,32,64,64],'kernel_size':3,'dropout':0.2,
         'learning_rate':0.001,'batch_size':32,'epochs':30,'patience':6}
META_P = {'n_estimators':100,'max_depth':5,'min_samples_split':5,'random_state':SEED,'n_jobs':1}

def metrics(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    mae = float(np.mean(np.abs(yt-yp)))
    mask = yt > 0
    mape = float(np.mean(np.abs(yt[mask]-yp[mask])/yt[mask])*100) if mask.any() else 0.
    ss_r = np.sum((yt-yp)**2); ss_t = np.sum((yt-np.mean(yt))**2)
    r2 = float(1 - ss_r/ss_t) if ss_t > 0 else 0.
    return {'RMSE':round(rmse,2),'MAE':round(mae,2),'MAPE':round(mape,2),'R2':round(r2,4)}

def main():
    T0 = time.time()
    log = open(f'{RES}/pipeline_log.txt','w')
    def pr(s):
        print(s, flush=True); log.write(s+'\n'); log.flush()

    pr("=== Loading cached research bundle ===")
    with open(PROJECT_ROOT / 'data' / 'processed' / 'research_bundle.pkl','rb') as f:
        B = pickle.load(f)
    pr(f"Train: tab={B['Xt_tr'].shape} seq={B['Xs_tr'].shape} y={B['y_tr'].shape}")
    pr(f"Val:   tab={B['Xt_va'].shape} seq={B['Xs_va'].shape} y={B['y_va'].shape}")
    pr(f"Test:  tab={B['Xt_te'].shape} seq={B['Xs_te'].shape} y={B['y_te'].shape}")
    indim = B['Xs_tr'].shape[2]

    # ============ PHASE 7: STANDALONE MODELS ============
    pr("\n=== PHASE 7: Standalone Models ===")
    preds_te, preds_va = {}, {}

    t=time.time()
    m=XGBoostModel(params=XGB_P); m.fit(B['Xt_tr'],B['y_tr'])
    preds_te['XGBoost']=m.predict(B['Xt_te']); preds_va['XGBoost']=m.predict(B['Xt_va'])
    m.save(f'{RES}/models/xgboost.pkl')
    pr(f"  XGBoost: {time.time()-t:.1f}s  RMSE={np.sqrt(np.mean((B['y_te']-preds_te['XGBoost'])**2)):.2f}")

    t=time.time()
    m=ExtraTreesModel(params=ET_P); m.fit(B['Xt_tr'],B['y_tr'])
    preds_te['ExtraTrees']=m.predict(B['Xt_te']); preds_va['ExtraTrees']=m.predict(B['Xt_va'])
    m.save(f'{RES}/models/extra_trees.pkl')
    pr(f"  ExtraTrees: {time.time()-t:.1f}s  RMSE={np.sqrt(np.mean((B['y_te']-preds_te['ExtraTrees'])**2)):.2f}")

    for name, Cls, params in [('BiLSTM',BiLSTMModel,RNN_P),('BiGRU',BiGRUModel,RNN_P),('TCN',TCNModel,TCN_P)]:
        t=time.time(); torch.manual_seed(SEED)
        m=Cls(input_dim=indim, params=params)
        m.fit(B['Xs_tr'],B['y_tr'],B['Xs_va'],B['y_va'],verbose=False)
        preds_te[name]=m.predict(B['Xs_te']); preds_va[name]=m.predict(B['Xs_va'])
        m.save(f'{RES}/models/{name.lower()}.pt')
        pr(f"  {name}: {time.time()-t:.1f}s  RMSE={np.sqrt(np.mean((B['y_te']-preds_te[name])**2)):.2f}")

    # ============ PHASE 8: OOF STACKING ============
    pr("\n=== PHASE 8: OOF Stacking ===")
    tscv = TimeSeriesSplit(n_splits=5)
    all_comps = ['XGBoost','BiLSTM','BiGRU','TCN']
    oof = {c: [] for c in all_comps}
    oof_y = []

    for fold, (tr_i, va_i) in enumerate(tscv.split(B['Xt_tr'])):
        pr(f"  Fold {fold+1}/5 (train={len(tr_i)}, val={len(va_i)})...")
        yt, yv = B['y_tr'][tr_i], B['y_tr'][va_i]

        torch.manual_seed(SEED+fold)
        m=XGBoostModel(params=XGB_P); m.fit(B['Xt_tr'][tr_i],yt)
        oof['XGBoost'].append(m.predict(B['Xt_tr'][va_i]))

        torch.manual_seed(SEED+fold)
        m=BiLSTMModel(input_dim=indim,params=RNN_P)
        m.fit(B['Xs_tr'][tr_i],yt,B['Xs_tr'][va_i],yv,verbose=False)
        oof['BiLSTM'].append(m.predict(B['Xs_tr'][va_i]))

        torch.manual_seed(SEED+fold)
        m=BiGRUModel(input_dim=indim,params=RNN_P)
        m.fit(B['Xs_tr'][tr_i],yt,B['Xs_tr'][va_i],yv,verbose=False)
        oof['BiGRU'].append(m.predict(B['Xs_tr'][va_i]))

        torch.manual_seed(SEED+fold)
        m=TCNModel(input_dim=indim,params=TCN_P)
        m.fit(B['Xs_tr'][tr_i],yt,B['Xs_tr'][va_i],yv,verbose=False)
        oof['TCN'].append(m.predict(B['Xs_tr'][va_i]))

        oof_y.append(yv)
        pr(f"    Fold {fold+1} complete.")

    for c in all_comps:
        oof[c] = np.concatenate(oof[c])
    oof_y = np.concatenate(oof_y)
    pr(f"  OOF predictions: {len(oof_y)} samples")

    # SE-1: XGBoost + BiLSTM + TCN
    se1_c = ['XGBoost','BiLSTM','TCN']
    Z1 = np.column_stack([oof[c] for c in se1_c])
    meta1 = ExtraTreesRegressor(**META_P); meta1.fit(Z1, oof_y)
    Z1_te = np.column_stack([preds_te[c] for c in se1_c])
    preds_te['SE-1'] = meta1.predict(Z1_te)
    fi1 = dict(zip(se1_c, [f'{v:.3f}' for v in meta1.feature_importances_]))
    pr(f"  SE-1 meta-learner trained. Importances: {fi1}")
    pr(f"  SE-1 RMSE={np.sqrt(np.mean((B['y_te']-preds_te['SE-1'])**2)):.2f}")

    # SE-2: XGBoost + BiGRU + TCN
    se2_c = ['XGBoost','BiGRU','TCN']
    Z2 = np.column_stack([oof[c] for c in se2_c])
    meta2 = ExtraTreesRegressor(**META_P); meta2.fit(Z2, oof_y)
    Z2_te = np.column_stack([preds_te[c] for c in se2_c])
    preds_te['SE-2'] = meta2.predict(Z2_te)
    fi2 = dict(zip(se2_c, [f'{v:.3f}' for v in meta2.feature_importances_]))
    pr(f"  SE-2 meta-learner trained. Importances: {fi2}")
    pr(f"  SE-2 RMSE={np.sqrt(np.mean((B['y_te']-preds_te['SE-2'])**2)):.2f}")

    # ============ PHASE 9: ABLATIONS ============
    pr("\n=== PHASE 9: Ablation Study ===")

    pr("  --- Hybrid 1 (SE-1) Ablations ---")
    se1_ablations = [
        ('H1-A1 (BiLSTM+TCN, no XGB)', ['BiLSTM','TCN']),
        ('H1-A2 (XGB+BiLSTM, no TCN)', ['XGBoost','BiLSTM']),
        ('H1-A3 (XGB+TCN, no BiLSTM)', ['XGBoost','TCN']),
        ('H1-A4 (Full SE-1)', ['XGBoost','BiLSTM','TCN']),
    ]
    se1_abl_results = []
    for name, comps in se1_ablations:
        Zo = np.column_stack([oof[c] for c in comps])
        m = ExtraTreesRegressor(**META_P).fit(Zo, oof_y)
        Zt = np.column_stack([preds_te[c] for c in comps])
        p = m.predict(Zt)
        met = metrics(B['y_te'], p); met['Config']=name; met['Components']='+'.join(comps)
        se1_abl_results.append(met)
        pr(f"    {name}: RMSE={met['RMSE']}, R²={met['R2']}")

    pr("\n  --- Hybrid 2 (SE-2) Ablations ---")
    pr("  NOTE: H2-A3 guide says 'without BiLSTM' but BiLSTM is NOT in SE-2.")
    pr("        Using parallel interpretation: XGBoost+TCN (without BiGRU).")
    se2_ablations = [
        ('H2-A1 (BiGRU+TCN, no XGB)', ['BiGRU','TCN']),
        ('H2-A2 (XGB+BiGRU, no TCN)', ['XGBoost','BiGRU']),
        ('H2-A3 (XGB+TCN, no BiGRU*)', ['XGBoost','TCN']),
        ('H2-A4 (Full SE-2)', ['XGBoost','BiGRU','TCN']),
    ]
    se2_abl_results = []
    for name, comps in se2_ablations:
        Zo = np.column_stack([oof[c] for c in comps])
        m = ExtraTreesRegressor(**META_P).fit(Zo, oof_y)
        Zt = np.column_stack([preds_te[c] for c in comps])
        p = m.predict(Zt)
        met = metrics(B['y_te'], p); met['Config']=name; met['Components']='+'.join(comps)
        se2_abl_results.append(met)
        pr(f"    {name}: RMSE={met['RMSE']}, R²={met['R2']}")

    # ============ PHASE 10: EVALUATION ============
    pr("\n=== PHASE 10: Full Evaluation ===")
    y_te, c_te, d_te = B['y_te'], B['c_te'], B['d_te']

    # Main comparison table
    main_rows = []
    for name in ['XGBoost','ExtraTrees','BiLSTM','BiGRU','TCN','SE-1','SE-2']:
        if name in preds_te:
            m = metrics(y_te, preds_te[name]); m['Model']=name; main_rows.append(m)
    df_main = pd.DataFrame(main_rows)
    df_main.to_csv(f'{RES}/metrics/main_comparison.csv', index=False)
    pr("\nMAIN COMPARISON TABLE:")
    pr(df_main[['Model','RMSE','MAE','MAPE','R2']].to_string(index=False))

    # Per-city breakdown
    city_rows = []
    for city in sorted(np.unique(c_te)):
        mask = c_te == city
        for name, p in preds_te.items():
            m = metrics(y_te[mask], p[mask]); m['City']=city; m['Model']=name
            city_rows.append(m)
    df_city = pd.DataFrame(city_rows)
    df_city.to_csv(f'{RES}/metrics/per_city_breakdown.csv', index=False)
    pr("\nPER-CITY R² (selected models):")
    pivot = df_city.pivot_table(index='City',columns='Model',values='R2')
    pr(pivot[['XGBoost','SE-1','SE-2']].to_string() if all(c in pivot.columns for c in ['XGBoost','SE-1','SE-2']) else pivot.to_string())

    # Extreme AQI
    for thresh in [300, 200, 150]:
        ext_mask = y_te > thresh
        if ext_mask.sum() >= 10:
            pr(f"\nEXTREME AQI (>{thresh}): {ext_mask.sum()} samples")
            ext_rows = []
            for name, p in preds_te.items():
                m = metrics(y_te[ext_mask], p[ext_mask]); m['Model']=name
                under = p[ext_mask] < y_te[ext_mask]
                m['Underest_Rate%'] = round(float(under.mean())*100,1)
                m['Avg_Underest'] = round(float(np.mean(y_te[ext_mask][under]-p[ext_mask][under])),1) if under.any() else 0
                ext_rows.append(m)
            df_ext = pd.DataFrame(ext_rows)
            df_ext.to_csv(f'{RES}/metrics/extreme_aqi_analysis.csv', index=False)
            pr(df_ext[['Model','RMSE','MAE','R2','Underest_Rate%']].to_string(index=False))
            break

    # Ablation CSVs
    pd.DataFrame(se1_abl_results).to_csv(f'{RES}/metrics/se1_ablation.csv', index=False)
    pd.DataFrame(se2_abl_results).to_csv(f'{RES}/metrics/se2_ablation.csv', index=False)

    # Predictions CSV
    df_preds = pd.DataFrame({'Date':d_te,'City':c_te,'AQI_actual':y_te})
    for name, p in preds_te.items():
        df_preds[name] = p
    df_preds.to_csv(f'{RES}/predictions/all_test_predictions.csv', index=False)

    # ============ PHASE 11: FIGURES ============
    pr("\n=== PHASE 11: Generating Figures ===")
    sns.set_style('whitegrid')

    # Fig 1: Model comparison
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    order = df_main.sort_values('RMSE')['Model'].tolist()
    sns.barplot(data=df_main,x='Model',y='RMSE',order=order,ax=axes[0],palette='viridis')
    axes[0].set_title('Test RMSE by Model'); axes[0].tick_params(axis='x',rotation=45)
    sns.barplot(data=df_main,x='Model',y='R2',order=order,ax=axes[1],palette='viridis')
    axes[1].set_title('Test R² by Model'); axes[1].tick_params(axis='x',rotation=45)
    plt.tight_layout(); plt.savefig(f'{RES}/figures/fig_01_model_comparison.png',dpi=300,bbox_inches='tight'); plt.close()

    # Fig 2: Actual vs predicted
    plot_cities = ['Delhi']
    other = [c for c in sorted(np.unique(c_te)) if c != 'Delhi']
    if other: plot_cities.append(other[0])
    fig, axes = plt.subplots(len(plot_cities),1,figsize=(15,5*len(plot_cities)))
    if len(plot_cities)==1: axes=[axes]
    for ax, city in zip(axes, plot_cities):
        mask = c_te==city; dates = pd.to_datetime(d_te[mask])
        ax.plot(dates,y_te[mask],'k-',alpha=0.7,label='Actual',linewidth=1.2)
        for mn,color in [('XGBoost','tab:blue'),('SE-1','tab:orange'),('SE-2','tab:green')]:
            if mn in preds_te: ax.plot(dates,preds_te[mn][mask],alpha=0.7,label=mn,linewidth=0.9,color=color)
        ax.set_title(f'{city}: Actual vs Predicted AQI'); ax.legend(); ax.set_ylabel('AQI')
    plt.tight_layout(); plt.savefig(f'{RES}/figures/fig_02_actual_vs_predicted.png',dpi=300,bbox_inches='tight'); plt.close()

    # Fig 3 & 4: Ablation
    for tag,data,title in [('se1',se1_abl_results,'Hybrid 1 (SE-1)'),('se2',se2_abl_results,'Hybrid 2 (SE-2)')]:
        df_a = pd.DataFrame(data)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        sns.barplot(data=df_a,x='Config',y='RMSE',ax=axes[0],palette='Set2')
        axes[0].set_title(f'{title}: RMSE'); axes[0].tick_params(axis='x',rotation=30)
        sns.barplot(data=df_a,x='Config',y='R2',ax=axes[1],palette='Set2')
        axes[1].set_title(f'{title}: R²'); axes[1].tick_params(axis='x',rotation=30)
        plt.tight_layout()
        n='03' if tag=='se1' else '04'
        plt.savefig(f'{RES}/figures/fig_{n}_{tag}_ablation.png',dpi=300,bbox_inches='tight'); plt.close()

    # Fig 5: City R² heatmap
    pivot = df_city.pivot_table(index='City',columns='Model',values='R2')
    plt.figure(figsize=(12,max(6,len(pivot)*0.8)))
    sns.heatmap(pivot,annot=True,fmt='.3f',cmap='YlOrRd',linewidths=0.5)
    plt.title('R² Score by City and Model')
    plt.tight_layout(); plt.savefig(f'{RES}/figures/fig_05_city_heatmap.png',dpi=300,bbox_inches='tight'); plt.close()

    # Fig 6: Extreme AQI scatter
    for thresh in [300,200,150]:
        ext_m = y_te > thresh
        if ext_m.sum() >= 5:
            plt.figure(figsize=(8,8))
            for mn,marker in [('XGBoost','o'),('SE-1','s'),('SE-2','^')]:
                if mn in preds_te: plt.scatter(y_te[ext_m],preds_te[mn][ext_m],alpha=0.6,label=mn,marker=marker,s=50)
            lims = [thresh,max(y_te)+20]; plt.plot(lims,lims,'r--',alpha=0.5)
            plt.xlabel('Actual AQI'); plt.ylabel('Predicted AQI')
            plt.title(f'Extreme AQI (>{thresh}) Predictions'); plt.legend()
            plt.tight_layout(); plt.savefig(f'{RES}/figures/fig_06_extreme_aqi.png',dpi=300,bbox_inches='tight'); plt.close()
            break

    # Fig 7: Residual box
    res_data = []
    for mn in ['XGBoost','ExtraTrees','BiLSTM','BiGRU','TCN','SE-1','SE-2']:
        if mn in preds_te:
            for r in (y_te-preds_te[mn]): res_data.append({'Model':mn,'Residual':r})
    plt.figure(figsize=(12,6))
    sns.boxplot(data=pd.DataFrame(res_data),x='Model',y='Residual',palette='Set3')
    plt.axhline(y=0,color='r',linestyle='--',alpha=0.5)
    plt.title('Prediction Residual Distribution'); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(f'{RES}/figures/fig_07_residual_distribution.png',dpi=300,bbox_inches='tight'); plt.close()

    pr("  All 7 figures saved.")

    # ============ SAVE OOF + META INFO ============
    meta_info = {
        'se1_importances': dict(zip(se1_c, meta1.feature_importances_.tolist())),
        'se2_importances': dict(zip(se2_c, meta2.feature_importances_.tolist())),
        'cities': sorted(np.unique(c_te).tolist()),
        'n_features': B['Xt_tr'].shape[1],
        'train_samples': len(B['y_tr']),
        'val_samples': len(B['y_va']),
        'test_samples': len(B['y_te']),
    }
    with open(f'{RES}/meta_info.json','w') as f:
        json.dump(meta_info, f, indent=2)

    # ============ FINAL SUMMARY ============
    pr("\n" + "="*70)
    pr("FINAL RESULTS SUMMARY")
    pr("="*70)
    pr(f"\nCities ({len(meta_info['cities'])}): {meta_info['cities']}")
    pr(f"Feature count: {meta_info['n_features']} tabular + 10 sequence channels")
    pr(f"Train: {meta_info['train_samples']} samples (< 2019-03-01)")
    pr(f"Val:   {meta_info['val_samples']} samples (2019-03-01 to 2019-08-31)")
    pr(f"Test:  {meta_info['test_samples']} samples (2019-09-01 to 2020-07-01)")
    pr(f"\nMAIN COMPARISON:")
    pr(df_main[['Model','RMSE','MAE','MAPE','R2']].to_string(index=False))
    pr(f"\nSE-1 ABLATION (Hybrid 1):")
    pr(pd.DataFrame(se1_abl_results)[['Config','RMSE','MAE','MAPE','R2']].to_string(index=False))
    pr(f"\nSE-2 ABLATION (Hybrid 2):")
    pr("NOTE: H2-A3 guide says 'without BiLSTM' but BiLSTM is NOT in SE-2.")
    pr("      Used parallel interpretation: XGBoost+TCN (without BiGRU).")
    pr(pd.DataFrame(se2_abl_results)[['Config','RMSE','MAE','MAPE','R2']].to_string(index=False))
    pr(f"\nMETA-LEARNER FEATURE IMPORTANCES:")
    pr(f"  SE-1: {fi1}")
    pr(f"  SE-2: {fi2}")
    pr(f"\nTotal pipeline time: {time.time()-T0:.1f}s")
    pr("Pipeline complete.")
    log.close()

if __name__=='__main__':
    main()
