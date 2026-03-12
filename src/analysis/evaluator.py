import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from src.calibrator.SubjectCalibrator import SubjectCalibrator, FeatureExporter


class HemodynamicEvaluator:
    @staticmethod
    def process_population(samples, ppg_waves, time_features, fs=125):
        """
        Iterates through ALL samples, calibrates each, and aggregates results.
        Returns:
            - results_df: DataFrame with one row per patient
            - kpi_dict: Dictionary of aggregate accuracy metrics
        """
        results = []
        calibrator = SubjectCalibrator()

        print(f"Starting Batch Processing for {len(samples)} patients...")

        for i, (sample, ppg_wave, time_feature) in enumerate(zip(samples, ppg_waves, time_features)):

            # 1. Feature Extraction
            df_feat = FeatureExporter.extract_training_data(samples=[sample],
                                                            ppg_waves=[ppg_wave],
                                                            time_features=[time_feature])

            # 2. Calibration & Estimation
            # This returns the dictionary with 'sbp_est_mean', 'alpha_mean', etc.
            calib = calibrator.calibrate_patient(df_feat, sample.patient_id)

            if calib is None:
                continue

            # 3. Aggregate Data
            # We combine the Calibration results with the Ground Truth labels
            row = calib.copy()
            row['ref_sbp'] = sample.bps
            row['ref_dbp'] = sample.bpd

            # Calculate Individual Errors
            row['error_sbp'] = row['sbp_est_mean'] - sample.bps
            row['error_dbp'] = row['dbp_est_mean'] - sample.bpd

            results.append(row)

        # 4. Create DataFrame & Save CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv("hemodynamic_population_report.csv", index=False)
        print(f"Processing Complete. Saved {len(df_results)} patient records.")

        return df_results

    @staticmethod
    def compute_kpis(df):
        """
        Calculates AAMI Standard Metrics (MAE, STD) and BHS Correlation.
        """
        kpis = {}

        for metric in ['sbp', 'dbp']:
            est_col = f'{metric}_est_mean'
            ref_col = f'ref_{metric}'
            err_col = f'error_{metric}'

            # 1. Mean Absolute Error (MAE) - Accuracy
            mae = np.mean(np.abs(df[err_col]))

            # 2. Mean Error (Bias) & Std Dev (Precision)
            bias = np.mean(df[err_col])
            precision = np.std(df[err_col])

            # 3. Pearson Correlation (r)
            corr, _ = pearsonr(df[est_col], df[ref_col])

            kpis[metric] = {
                'MAE': round(mae, 2),
                'Bias': round(bias, 2),
                'Precision_SD': round(precision, 2),
                'Correlation_r': round(corr, 3),
                'N_Patients': len(df)
            }

        return kpis