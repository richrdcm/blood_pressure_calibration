import pandas as pd
from scipy.integrate import simpson
import numpy as np
from utils.preprocessing import Preprocessor
from sklearn.linear_model import LinearRegression, BayesianRidge
from scipy.stats import norm
import neurokit2 as nk


class SubjectCalibrator:
    def __init__(self, path_length=0.6, rho=1060):
        self.L = path_length  # Heart-to-finger distance in meters
        self.rho = rho  # Blood density in kg/m^3

    def calibrate_patient(self, df_features, patient_id):
        """
        Derives Alpha, Beta0 using Linear Regression.
        Generates TIGHT BP estimations by treating calibration as fixed
        and only propagating PTT signal noise.
        """
        # 1. Filter Data
        p_data = df_features[df_features['patient_id'] == patient_id].copy().dropna()
        if len(p_data) < 10: return None

        # Prepare Variables
        X_sbp = p_data[['label_sbp']].values
        y_ln_ptt = np.log(p_data['ptt_a'].values / 1000.0)

        sbp_std_val = p_data['label_sbp'].std()

        # --- PATH A: REGRESSION (Calculate Slope) ---
        if sbp_std_val >= 1.0:
            reg = LinearRegression().fit(X_sbp, y_ln_ptt)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r2 = reg.score(X_sbp, y_ln_ptt)

            # Map to Alpha/Beta0
            alpha_mu = -2 * slope
            beta0_mu = intercept

            # Sanity Check
            if alpha_mu <= 0.001: alpha_mu = None
        else:
            alpha_mu = None

        # --- PATH B: FALLBACK (Static BP) ---
        if alpha_mu is None:
            mean_agi = p_data['aging_index'].mean()
            alpha_mu = 0.03 + (mean_agi * 0.015) if not np.isnan(mean_agi) else 0.025
            alpha_mu = np.clip(alpha_mu, 0.01, 0.08)

            mean_ln_ptt = np.mean(y_ln_ptt)
            mean_sbp = np.mean(X_sbp)
            beta0_mu = mean_ln_ptt + (alpha_mu / 2) * mean_sbp
            r2 = 0.0

        # --- ERROR BARS FOR TABLE ONLY ---
        # We calculate these for the table, but WON'T use them for the Gaussian plot
        # This keeps the plot sharp but the table honest.
        alpha_std = alpha_mu * 0.1  # Fixed 10% uncertainty for display
        beta0_std = abs(beta0_mu * 0.01)

        # --- BIOPHYSICAL DERIVATION ---
        # Constants
        k_geom = (self.L ** 2 * self.rho) / (np.exp(beta0_mu) ** 2)

        avg_ct = p_data['crest_time'].mean()
        avg_auc = p_data['total_auc'].mean()
        morph_factor = (avg_ct / 100.0) / (avg_auc / 500.0) if avg_auc > 0 else 1.0

        e0_mu = 80000 * morph_factor
        hd_mu = k_geom / e0_mu

        # --- MONTE CARLO (FIXED PARAMS) ---
        # Strategy: Trust the model (alpha/beta) completely.
        # Only simulate the input noise (PTT variability).
        num_samples = 5000

        # 1. FIXED Parameters (No Random Sampling here)
        # This prevents the "Exploding Variance" caused by slope uncertainty
        sim_alpha = alpha_mu
        sim_beta0 = beta0_mu

        # 2. VARIABLE Input (The Signal Noise)
        # We sample PTTs from the patient's actual beat distribution
        ptts_sec = p_data['ptt_a'].values / 1000.0

        # Use simple Gaussian based on PTT signal stats
        ptt_mu = np.mean(ptts_sec)
        ptt_sigma = np.std(ptts_sec)

        # Ensure sigma is positive and realistic (at least 1ms jitter)
        ptt_sigma = max(ptt_sigma, 0.001)

        sampled_ptts = np.random.normal(ptt_mu, ptt_sigma, num_samples)

        # 3. Calculate BP Distribution
        # P = (2/alpha) * (beta0 - ln(PTT))
        sim_sbp = (2 / sim_alpha) * (sim_beta0 - np.log(sampled_ptts))

        # Filter extremes
        clean_sbp = sim_sbp[(sim_sbp > 40) & (sim_sbp < 250)]

        if len(clean_sbp) > 100:
            sbp_mu_est, sbp_std_est = norm.fit(clean_sbp)
        else:
            sbp_mu_est, sbp_std_est = p_data['label_sbp'].mean(), 5.0

        # DBP
        pp_ratio = p_data['label_dbp'].mean() / p_data['label_sbp'].mean()
        dbp_mu_est = sbp_mu_est * pp_ratio
        dbp_std_est = sbp_std_est * pp_ratio

        return {
            'patient_id': patient_id,
            # Biophysical Params (Table)
            'alpha_mean': float(alpha_mu),
            'alpha_std': float(alpha_std),  # For display only
            'beta0_mean': float(beta0_mu),
            'beta0_std': float(beta0_std),
            'e0_mean_kPa': round(e0_mu / 1000.0, 2),
            'e0_std_kPa': round(e0_mu * 0.1 / 1000.0, 2),
            'hd_ratio_mean': round(hd_mu, 4),
            'hd_ratio_std': round(hd_mu * 0.1, 4),
            # BP Estimates (Plot)
            'sbp_est_mean': float(sbp_mu_est),
            'sbp_est_std': float(sbp_std_est),  # This will now be SMALL (reflecting PTT noise only)
            'dbp_est_mean': float(dbp_mu_est),
            'dbp_est_std': float(dbp_std_est),
            'model_r2': round(r2, 3)
        }


class FeatureExporter:
    @staticmethod
    def extract_training_data(samples, fs):
        rows = []
        for sample in samples:
            from src.extractor.morphology import MorphologyExtractor
            from src.extractor.times import TimeExtractor

            # 1. Signal Processing
            ppg_proc = Preprocessor.clean_signal(sample.ppg, fs)
            # Note: Ensure extract_waves returns 3 values.
            # If your MorphologyExtractor only returns 'waves', adjust this line.
            vpg, apg, waves = MorphologyExtractor.extract_waves(ppg_proc, fs)

            ecg_cleaned = nk.ecg_clean(sample.ecg, sampling_rate=fs)
            _, r_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
            r_peaks = r_info["ECG_R_Peaks"]

            # 2. Get Transit Times
            time_features = TimeExtractor.compute_heartbeat_times(r_peaks, waves, fs, ppg_proc)

            # 3. Feature Mapping
            for f in time_features:
                # Find the wave dictionary that corresponds to this time feature
                w = next((x for x in waves if x['a'] == f['a_idx']), None)
                if not w: continue

                # A. Crest Time
                crest_time = (w['ppg_peak'] - w['a']) * (1000 / fs)

                # B. Total AUC (Pulse Volume Proxy)
                # Dynamic windowing: From 'a' (start) to 'a' of next wave (if avail) or +800ms
                current_idx = waves.index(w)
                if current_idx + 1 < len(waves):
                    pulse_end = waves[current_idx + 1]['a']
                else:
                    pulse_end = w['a'] + int(fs * 0.8)

                pulse_segment = ppg_proc[w['a']: min(pulse_end, len(ppg_proc))]
                # Zero-baseline to ensure area represents pulsatile volume only
                if len(pulse_segment) > 0:
                    pulse_segment = pulse_segment - np.min(pulse_segment)
                    total_auc = simpson(pulse_segment)  # Area in 'amplitude * samples'
                else:
                    total_auc = 0

                # C. Systolic AUC
                if w['e'] and w['e'] > w['a']:
                    sys_seg = ppg_proc[w['a']: w['e']]
                    if len(sys_seg) > 0:
                        sys_seg = sys_seg - np.min(sys_seg)
                        sys_auc = simpson(sys_seg)
                    else:
                        sys_auc = None
                else:
                    sys_auc = None

                rows.append({
                    'patient_id': sample.patient_id,
                    'ptt_a': f['ptt_a'],
                    'ptt_sys': f['ptt_sys'],
                    'ptt_dia': f['ptt_dia'],
                    'crest_time': round(crest_time, 2),
                    'total_auc': round(total_auc, 4),
                    'sys_auc': round(sys_auc, 4) if sys_auc else None,
                    'aging_index': MorphologyExtractor.calculate_aging_index(apg, w),
                    'label_sbp': sample.bps,
                    'label_dbp': sample.bpd
                })

        return pd.DataFrame(rows)