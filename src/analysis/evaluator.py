import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from src.calibrator.SubjectCalibrator import SubjectCalibrator, FeatureExporter


class HemodynamicEvaluator:
    @staticmethod
    def process_population(samples, ppg_waves, time_features, calibrations, cleaned_samples=None):
        """
        Iterates through ALL samples, extracts features, calibrates each patient,
        and aggregates results.

        Parameters
        ----------
        samples         : list of raw BPSample (used for ground-truth BP labels)
        ppg_waves       : list of (vpg, apg, waves) tuples from MorphologyExtractor
        time_features   : list of beat dicts from TimeExtractor
        cleaned_samples : list of cleaned BPSample from Preprocessor.clean_signals()
                          If None, raw samples are used (may cause index mismatches)
        calibrations    :
        """
        if cleaned_samples is None:
            print("[HemodynamicEvaluator] WARNING: cleaned_samples not provided. "
                  "Using raw samples — wave indices may not match.")
            cleaned_samples = samples

        calibrator = SubjectCalibrator()
        print(f"[HemodynamicEvaluator] Starting batch processing for {len(samples)} samples...")

        # ── Step 1: Extract features for all samples ──────────────────────────
        df_features = FeatureExporter.extract_training_data(
            samples=cleaned_samples,
            ppg_waves=ppg_waves,
            time_features=time_features,
        )

        if df_features.empty:
            print("[HemodynamicEvaluator] No features extracted — aborting.")
            return pd.DataFrame()

        print(f"[HemodynamicEvaluator] Extracted {len(df_features)} beats "
              f"across {df_features['patient_id'].nunique()} patient(s).")

        # ── Step 3: Build results — join calibration with ground truth ─────────
        # Lookup from patient_id → raw sample for ground truth BP
        sample_lookup = {str(s.patient_id): s for s in samples}

        results = []
        for calib in calibrations:
            if not calib:
                continue

            patient_id = calib['patient_id']
            row = calib.copy()

            # Ground truth BP — may be None if no reference available
            raw_sample = sample_lookup.get(str(patient_id))
            ref_sbp = raw_sample.bps if raw_sample else None
            ref_dbp = raw_sample.bpd if raw_sample else None

            row['ref_sbp'] = ref_sbp
            row['ref_dbp'] = ref_dbp

            # SBP error
            row['error_sbp'] = (
                round(calib['sbp_est_mean'] - ref_sbp, 2)
                if ref_sbp is not None else None
            )

            # DBP error
            row['error_dbp'] = (
                round(calib['dbp_est_mean'] - ref_dbp, 2)
                if ref_dbp is not None else None
            )

            # MAP reference and error
            row['ref_map'] = (
                round(ref_dbp + (ref_sbp - ref_dbp) / 3.0, 2)
                if (ref_sbp is not None and ref_dbp is not None) else None
            )
            row['error_map'] = (
                round(calib['map_est_mean'] - row['ref_map'], 2)
                if row['ref_map'] is not None else None
            )

            results.append(row)

        df_results = pd.DataFrame(results)
        df_results.to_csv("hemodynamic_population_report.csv", index=False)
        print(f"[HemodynamicEvaluator] Complete. Saved {len(df_results)} records.")
        return df_results

    @staticmethod
    def compute_kpis(df: pd.DataFrame) -> dict:
        """
        Calculates AAMI Standard Metrics (MAE, Bias, SD) and Pearson correlation.
        Only computed for patients that have reference BP values.
        """
        kpis = {}

        # Compute ref_map inline if not already present
        if 'ref_map' not in df.columns:
            if 'ref_sbp' in df.columns and 'ref_dbp' in df.columns:
                df = df.copy()
                df['ref_map'] = df['ref_dbp'] + (df['ref_sbp'] - df['ref_dbp']) / 3.0

        metrics = {
            'sbp': ('sbp_est_mean', 'ref_sbp', 'error_sbp'),
            'dbp': ('dbp_est_mean', 'ref_dbp', 'error_dbp'),
            'map': ('map_est_mean', 'ref_map', 'error_map'),
        }

        for metric, (est_col, ref_col, err_col) in metrics.items():

            # Skip if required columns are missing
            missing = [c for c in [est_col, ref_col, err_col] if c not in df.columns]
            if missing:
                print(f"[KPI] {metric.upper()}: skipping — missing columns {missing}")
                continue

            # Only rows with valid reference and valid error
            valid = df[df[err_col].notna() & df[ref_col].notna()].copy()
            if len(valid) < 2:
                print(f"[KPI] {metric.upper()}: not enough samples with reference — skipping.")
                continue

            mae = float(np.mean(np.abs(valid[err_col])))
            bias = float(np.mean(valid[err_col]))
            precision = float(np.std(valid[err_col]))
            corr, p_val = pearsonr(valid[est_col], valid[ref_col])

            kpis[metric] = {
                'MAE': round(mae, 2),
                'Bias': round(bias, 2),
                'Precision_SD': round(precision, 2),
                'Correlation_r': round(corr, 3),
                'P_value': round(p_val, 4),
                'N_samples': len(valid),
            }

            print(f"[KPI] {metric.upper()}: MAE={mae:.2f}, Bias={bias:.2f}, "
                  f"SD={precision:.2f}, r={corr:.3f}, N={len(valid)}")

        return kpis

    @staticmethod
    def save_results(df_results: pd.DataFrame, output_dir: str = "results"):
        """
        Saves calibration results to CSV files.

        Output files:
            results/biophysical_parameters.csv  — mean and std of biophysical params per patient
            results/bp_estimates.csv            — BP estimates vs reference per patient
            results/population_report.csv       — full raw results
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── Biophysical parameters ─────────────────────────────────────────────
        bio_cols = {
            'patient_id': 'Patient ID',
            'alpha_mean': 'Stiffness α (mean)',
            'alpha_std': 'Stiffness α (std)',
            'beta0_mean': 'Intercept β₀ (mean)',
            'beta0_std': 'Intercept β₀ (std)',
            'e0_mean_kPa': 'Elasticity E₀ mean (kPa)',
            'e0_std_kPa': 'Elasticity E₀ std (kPa)',
            'hd_ratio_mean': 'Wall ratio h/d (mean)',
            'hd_ratio_std': 'Wall ratio h/d (std)',
            'model_r2': 'Model R²',
        }

        available_bio = {k: v for k, v in bio_cols.items() if k in df_results.columns}
        df_bio = df_results[list(available_bio.keys())].copy()
        df_bio.rename(columns=available_bio, inplace=True)
        bio_path = os.path.join(output_dir, "biophysical_parameters.csv")
        df_bio.to_csv(bio_path, index=False)
        print(f"[Results] Biophysical parameters saved → {bio_path}")

        # ── BP estimates vs reference ──────────────────────────────────────────
        bp_cols = {
            'patient_id': 'Patient ID',
            'sbp_est_mean': 'SBP Estimated (mmHg)',
            'sbp_est_std': 'SBP Std (mmHg)',
            'ref_sbp': 'SBP Reference (mmHg)',
            'error_sbp': 'SBP Error (mmHg)',
            'map_est_mean': 'MAP Estimated (mmHg)',
            'map_est_std': 'MAP Std (mmHg)',
            'ref_map': 'MAP Reference (mmHg)',
            'error_map': 'MAP Error (mmHg)',
            'dbp_est_mean': 'DBP Estimated (mmHg)',
            'dbp_est_std': 'DBP Std (mmHg)',
            'ref_dbp': 'DBP Reference (mmHg)',
            'error_dbp': 'DBP Error (mmHg)',
        }

        available_bp = {k: v for k, v in bp_cols.items() if k in df_results.columns}
        df_bp = df_results[list(available_bp.keys())].copy()
        df_bp.rename(columns=available_bp, inplace=True)
        bp_path = os.path.join(output_dir, "bp_estimates.csv")
        df_bp.to_csv(bp_path, index=False)
        print(f"[Results] BP estimates saved → {bp_path}")

        # ── Full report ────────────────────────────────────────────────────────
        full_path = os.path.join(output_dir, "population_report.csv")
        df_results.to_csv(full_path, index=False)
        print(f"[Results] Full report saved → {full_path}")