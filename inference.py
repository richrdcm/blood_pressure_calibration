"""
ppg_only_estimator.py

Load saved biophysical parameters (alpha, beta0) and estimate BP
from PPG-only signals using fiducial point timing as PTT proxy.

PPG-only PTT proxies:
    ptt_proxy_a   : crest time  (a → systolic peak)   — fast, easy
    ptt_proxy_e   : (a → e)                            — more stable
    ibi           : inter-beat interval (a[i] → a[i+1]) — heart period
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import DataLoader
from utils.preprocessing import Preprocessor
from src.extractor.morphology import MorphologyExtractor
from schemas.bp_schema import BPSample


# ── 1. Load saved biophysical parameters ─────────────────────────────────────

def load_biophysical_params(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize patient ID column name
    if 'Patient ID' in df.columns:
        df = df.rename(columns={'Patient ID': 'patient_id'})
    print(f"[Loader] Loaded biophysical params for {len(df)} patient(s) from '{csv_path}'")
    return df


# ── 2. PPG-only PTT proxy extraction ─────────────────────────────────────────

def extract_ppg_only_features(samples: list, ppg_waves: list) -> pd.DataFrame:
    """
    Extracts PTT proxies from PPG fiducial points only — no ECG needed.

    PTT proxies:
        crest_time_ms : a → systolic peak (ms)  ← proxy for PTT_a
        a_to_e_ms     : a → e (ms)              ← proxy for PTT_e
        ibi_ms        : a[i] → a[i+1] (ms)      ← inter-beat interval
    """
    rows = []

    for sample, ppg_wave in zip(samples, ppg_waves):
        vpg, apg, waves = ppg_wave
        ts = sample.ppg_timestamps  # ms

        if not waves:
            print(f"[PPG-only] Patient {sample.patient_id}: no waves found.")
            continue

        for i, w in enumerate(waves):
            a_idx    = w.get('a')
            sys_idx  = w.get('ppg_peak')
            e_idx    = w.get('e')

            if a_idx is None or sys_idx is None:
                continue
            if a_idx >= len(ts) or sys_idx >= len(ts):
                continue

            # Crest time: a → systolic peak
            crest_time_ms = ts[sys_idx] - ts[a_idx]

            # a → e
            a_to_e_ms = (ts[e_idx] - ts[a_idx]) if (e_idx is not None and e_idx < len(ts)) else None

            # IBI: time from this 'a' to next 'a'
            ibi_ms = None
            if i + 1 < len(waves):
                next_a = waves[i + 1].get('a')
                if next_a is not None and next_a < len(ts):
                    ibi_ms = ts[next_a] - ts[a_idx]

            rows.append({
                'patient_id':    sample.patient_id,
                'beat_idx':      i,
                'a_idx':         a_idx,
                'crest_time_ms': round(crest_time_ms, 2),
                'a_to_e_ms':     round(a_to_e_ms, 2) if a_to_e_ms else None,
                'ibi_ms':        round(ibi_ms, 2)    if ibi_ms    else None,
                'label_sbp':     sample.bps,
                'label_dbp':     sample.bpd,
            })

    df = pd.DataFrame(rows)
    print(f"[PPG-only] Extracted {len(df)} beats from {df['patient_id'].nunique()} patient(s).")
    return df


# ── 3. BP estimation from saved params + PPG-only PTT proxy ──────────────────

def estimate_bp_ppg_only(
    features_df: pd.DataFrame,
    bio_params_df: pd.DataFrame,
    ptt_proxy_col: str = 'crest_time_ms',
    map_proxy_col: str = 'a_to_e_ms',
    n_monte_carlo: int = 5000,
) -> pd.DataFrame:

    # ── Column name mapping from save_results() renamed columns ──────────
    COL_PID       = 'patient_id'
    COL_ALPHA_SBP = 'Stiffness α (mean)'
    COL_BETA0_SBP = 'Intercept β₀ (mean)'

    # MAP params — not saved separately, fall back to SBP params
    # (the calibrator uses the same alpha/beta0 structure for both)
    COL_ALPHA_MAP = COL_ALPHA_SBP
    COL_BETA0_MAP = COL_BETA0_SBP

    results = []

    for _, params in bio_params_df.iterrows():
        pid = str(params[COL_PID])

        alpha_sbp = params.get(COL_ALPHA_SBP)
        beta0_sbp = params.get(COL_BETA0_SBP)
        alpha_map = params.get(COL_ALPHA_MAP, alpha_sbp)
        beta0_map = params.get(COL_BETA0_MAP, beta0_sbp)

        if pd.isna(alpha_sbp) or pd.isna(beta0_sbp):
            print(f"[Estimator] Patient {pid}: missing alpha/beta0 — skipping.")
            continue

        alpha_sbp = float(alpha_sbp)
        beta0_sbp = float(beta0_sbp)
        alpha_map = float(alpha_map)
        beta0_map = float(beta0_map)

        # Get beats for this patient
        p_feats = features_df[
            features_df['patient_id'] == pid
        ].dropna(subset=[ptt_proxy_col])

        if len(p_feats) < 5:
            print(f"[Estimator] Patient {pid}: only {len(p_feats)} valid beats — skipping.")
            continue

        # ── SBP from ptt_proxy_col ────────────────────────────────────────
        ptt_a_sec     = p_feats[ptt_proxy_col].values / 1000.0
        ptt_a_mu      = np.mean(ptt_a_sec)
        ptt_a_sigma   = max(np.std(ptt_a_sec), 0.001)
        sampled_ptt_a = np.random.normal(ptt_a_mu, ptt_a_sigma, n_monte_carlo)
        sampled_ptt_a = sampled_ptt_a[sampled_ptt_a > 0]

        sim_sbp   = (2 / alpha_sbp) * (beta0_sbp - np.log(sampled_ptt_a))
        clean_sbp = sim_sbp[(sim_sbp > 40) & (sim_sbp < 250)]
        sbp_mu, sbp_std = norm.fit(clean_sbp) if len(clean_sbp) > 100 else (float(np.mean(sim_sbp)), 10.0)

        # ── MAP from map_proxy_col ────────────────────────────────────────
        map_feats = p_feats.dropna(subset=[map_proxy_col])
        if len(map_feats) >= 5:
            ptt_e_sec     = map_feats[map_proxy_col].values / 1000.0
            ptt_e_mu      = np.mean(ptt_e_sec)
            ptt_e_sigma   = max(np.std(ptt_e_sec), 0.001)
            sampled_ptt_e = np.random.normal(ptt_e_mu, ptt_e_sigma, n_monte_carlo)
            sampled_ptt_e = sampled_ptt_e[sampled_ptt_e > 0]

            sim_map   = (2 / alpha_map) * (beta0_map - np.log(sampled_ptt_e))
            clean_map = sim_map[(sim_map > 40) & (sim_map < 180)]
            map_mu, map_std = norm.fit(clean_map) if len(clean_map) > 100 else (float(np.mean(sim_map)), 10.0)
        else:
            map_mu  = sbp_mu * 0.75
            map_std = sbp_std * 0.75

        # ── DBP = (3×MAP − SBP) / 2 ──────────────────────────────────────
        dbp_mu  = (3 * map_mu  - sbp_mu)  / 2.0
        dbp_std = np.sqrt((9 * map_std**2 + sbp_std**2) / 4.0)

        # ── Ground truth ──────────────────────────────────────────────────
        ref_sbp = float(p_feats['label_sbp'].dropna().iloc[0]) if p_feats['label_sbp'].notna().any() else None
        ref_dbp = float(p_feats['label_dbp'].dropna().iloc[0]) if p_feats['label_dbp'].notna().any() else None
        ref_map = round(ref_dbp + (ref_sbp - ref_dbp) / 3.0, 2) if (ref_sbp and ref_dbp) else None

        results.append({
            'patient_id':   pid,
            'n_beats':      len(p_feats),
            'sbp_est_mean': round(float(sbp_mu),  2),
            'sbp_est_std':  round(float(sbp_std), 2),
            'map_est_mean': round(float(map_mu),  2),
            'map_est_std':  round(float(map_std), 2),
            'dbp_est_mean': round(float(dbp_mu),  2),
            'dbp_est_std':  round(float(dbp_std), 2),
            'ref_sbp':      ref_sbp,
            'ref_dbp':      ref_dbp,
            'ref_map':      ref_map,
            'error_sbp':    round(float(sbp_mu) - ref_sbp, 2) if ref_sbp else None,
            'error_dbp':    round(float(dbp_mu) - ref_dbp, 2) if ref_dbp else None,
            'error_map':    round(float(map_mu) - ref_map, 2) if ref_map else None,
            'ptt_proxy':    ptt_proxy_col,
            'model_r2':     float(params.get('Model R²', 0.0)),
        })

        print(f"[Estimator] Patient {pid}: "
              f"SBP={sbp_mu:.1f}±{sbp_std:.1f}, "
              f"DBP={dbp_mu:.1f}±{dbp_std:.1f} mmHg "
              f"(ref: {ref_sbp}/{ref_dbp})")

    return pd.DataFrame(results)


# ── 4. KPIs ───────────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Reuses the same KPI logic as HemodynamicEvaluator."""
    kpis = {}
    metrics = {
        'sbp': ('sbp_est_mean', 'ref_sbp', 'error_sbp'),
        'dbp': ('dbp_est_mean', 'ref_dbp', 'error_dbp'),
        'map': ('map_est_mean', 'ref_map', 'error_map'),
    }
    for metric, (est_col, ref_col, err_col) in metrics.items():
        if not all(c in df.columns for c in [est_col, ref_col, err_col]):
            continue
        valid = df[df[err_col].notna() & df[ref_col].notna()]
        if len(valid) < 2:
            continue
        mae       = float(np.mean(np.abs(valid[err_col])))
        bias      = float(np.mean(valid[err_col]))
        precision = float(np.std(valid[err_col]))
        corr, _   = pearsonr(valid[est_col], valid[ref_col])
        kpis[metric] = {
            'MAE':           round(mae, 2),
            'Bias':          round(bias, 2),
            'Precision_SD':  round(precision, 2),
            'Correlation_r': round(corr, 3),
            'N_samples':     len(valid),
        }
        print(f"[KPI] {metric.upper()}: MAE={mae:.2f}, Bias={bias:.2f}, "
              f"SD={precision:.2f}, r={corr:.3f}, N={len(valid)}")
    return kpis


# ── 5. Visualisation ──────────────────────────────────────────────────────────

def plot_ppg_only_evaluation(df_results: pd.DataFrame, output_dir: str = "results"):
    """
    Plots estimated vs reference BP — reuses the same style as plot_estimation_performance.
    One scatter plot per metric (SBP, MAP, DBP) with identity line and error bars.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ('sbp', 'SBP', '#00FF00', 'ref_sbp', 'sbp_est_mean', 'sbp_est_std'),
        ('map', 'MAP', '#FFA500', 'ref_map', 'map_est_mean', 'map_est_std'),
        ('dbp', 'DBP', '#0074D9', 'ref_dbp', 'dbp_est_mean', 'dbp_est_std'),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["SBP: Estimated vs Reference",
                        "MAP: Estimated vs Reference",
                        "DBP: Estimated vs Reference"],
        horizontal_spacing=0.08,
    )

    for col_idx, (metric, label, color, ref_col, est_col, std_col) in enumerate(metrics, start=1):
        valid = df_results[df_results[ref_col].notna() & df_results[est_col].notna()]
        if valid.empty:
            continue

        ref_vals = valid[ref_col].values
        est_vals = valid[est_col].values
        std_vals = valid[std_col].values if std_col in valid.columns else np.zeros_like(est_vals)

        # Identity line
        bp_min = min(ref_vals.min(), est_vals.min()) - 5
        bp_max = max(ref_vals.max(), est_vals.max()) + 5
        fig.add_trace(go.Scatter(
            x=[bp_min, bp_max], y=[bp_min, bp_max],
            mode='lines', line=dict(color='white', dash='dash', width=1),
            showlegend=False, name='Identity',
        ), row=1, col=col_idx)

        # Scatter with error bars
        fig.add_trace(go.Scatter(
            x=ref_vals, y=est_vals,
            mode='markers',
            error_y=dict(type='data', array=std_vals, visible=True, color=color),
            marker=dict(color=color, size=10, line=dict(width=1, color='white')),
            text=valid['patient_id'],
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Ref {label}: %{{x:.1f}} mmHg<br>"
                f"Est {label}: %{{y:.1f}} mmHg<br>"
                "<extra></extra>"
            ),
            name=label,
            showlegend=True,
        ), row=1, col=col_idx)

        fig.update_xaxes(title_text=f"Reference {label} (mmHg)", row=1, col=col_idx)
        fig.update_yaxes(title_text=f"Estimated {label} (mmHg)", row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="PPG-Only BP Estimation vs Reference", x=0.5),
        template="plotly_dark",
        height=500,
    )

    path = os.path.join(output_dir, "ppg_only_evaluation.html")
    fig.write_html(path)
    print(f"[Plot] Saved → {path}")
    fig.show()


# ── 6. Main script ────────────────────────────────────────────────────────────

def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    BIO_PARAMS_PATH = "test/fihmi/biophysical_parameters.csv"
    DATA_PATH       = "datasets/raw/mcs/ECG_Calib_64hz/fihmi/fihmi_calib_macAddress_6c1deb04a9ce_pt"
    DATASET_TYPE    = "mcs"
    OUTPUT_DIR      = "test/fihmi/inference"

    # ── Load biophysical parameters ────────────────────────────────────────
    bio_params_df = load_biophysical_params(BIO_PARAMS_PATH)

    # ── Load PPG-only data (no ECG needed but load as usual) ───────────────
    loader  = DataLoader()
    samples = loader.load_from_csv(
        file_path=DATA_PATH,
        dataset_type=DATASET_TYPE,
        index_from=139, index_to=145,
        each_file_is_own_patient=False,
        max_duration_msec=10000,
    )

    if not samples:
        print("No samples loaded.")
        return

    # ── Preprocess PPG (ECG optional — used only for SQI if present) ───────
    processor       = Preprocessor()
    cleaned_samples = processor.clean_signals(samples)

    # ── Extract morphology from PPG only ───────────────────────────────────
    morphology = MorphologyExtractor()
    ppg_waves  = morphology.extract_samples_waves(cleaned_samples)

    # ── Extract PPG-only PTT proxies ───────────────────────────────────────
    features_df = extract_ppg_only_features(cleaned_samples, ppg_waves)

    if features_df.empty:
        print("No features extracted.")
        return

    # ── Estimate BP using saved biophysical parameters ─────────────────────
    df_results = estimate_bp_ppg_only(
        features_df   = features_df,
        bio_params_df = bio_params_df,
        ptt_proxy_col = 'crest_time_ms',  # a → systolic peak ← proxy for PTT_a
        map_proxy_col = 'a_to_e_ms',      # a → e            ← proxy for PTT_e
    )

    if df_results.empty:
        print("No estimates produced.")
        return

    # ── KPIs ───────────────────────────────────────────────────────────────
    print("\n=== PPG-Only Estimation KPIs ===")
    kpis = compute_kpis(df_results)
    for metric, vals in kpis.items():
        print(f"  {metric.upper()}: {vals}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "ppg_only_results.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/ppg_only_results.csv")

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_ppg_only_evaluation(df_results, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()