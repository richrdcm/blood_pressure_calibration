import pandas as pd
from scipy.integrate import simpson
import numpy as np
from utils.preprocessing import Preprocessor
from sklearn.linear_model import LinearRegression, BayesianRidge
from scipy.stats import norm
import neurokit2 as nk
from schemas.bp_schema import BPSample


class SubjectCalibrator:
    def __init__(self, path_length=0.6, rho=1060):
        self.L = path_length  # Heart-to-finger distance in meters
        self.rho = rho  # Blood density in kg/m^3

    def calibrate_patients(self, df_features):
        patient_calibrations = []
        for patient_id, p_data in df_features.groupby('patient_id'):
            calib = self.calibrate_patient(p_data, patient_id)
            if not calib:
                print(f"Calibration failed for Patient {patient_id}")
                patient_calibrations.append([])
                continue
            patient_calibrations.append(calib)
        return patient_calibrations

    def calibrate_patient(self, p_data, patient_id):
        """
        Derives SBP and DBP from PTT_a and PTT_e using the Moens-Korteweg model.

        Physiology:
            PTT_a  (R → 'a' wave foot)     → correlates with SBP
            PTT_e  (R → dicrotic notch)    → correlates with MAP
            DBP = 2*MAP - SBP              (standard clinical identity)

        No diastolic peak required.
        """
        if len(p_data) < 10:
            return None

        p_data = p_data.dropna(subset=['ptt_a', 'ptt_e'])
        if len(p_data) < 10:
            print(f"[Calibrator] Patient {patient_id}: insufficient beats "
                  f"with both ptt_a and ptt_e after dropna.")
            return None

        print(f"[Calibrator] Patient {patient_id}: {len(p_data)} beats with ptt_a + ptt_e.")

        # ── Variables ──────────────────────────────────────────────────────────
        X_sbp = p_data[['label_sbp']].values
        y_ln_ptt_a = np.log(p_data['ptt_a'].values / 1000.0)
        y_ln_ptt_e = np.log(p_data['ptt_e'].values / 1000.0)

        # MAP from labels only if BOTH sbp and dbp are available
        has_labels = (
                'label_sbp' in p_data.columns and
                'label_dbp' in p_data.columns and
                p_data['label_sbp'].notna().all() and
                p_data['label_dbp'].notna().all()
        )

        if has_labels:
            map_vals = p_data['label_dbp'].values + (
                    p_data['label_sbp'].values - p_data['label_dbp'].values
            ) / 3.0
            X_map = map_vals.reshape(-1, 1)
            X_sbp = p_data[['label_sbp']].values
        else:
            # No reference BP — use PTT_e self-supervised:
            # fit MAP axis as a scaled version of PTT_e distribution
            # (relative calibration only, absolute scale from population prior)
            ptt_e_sec = p_data['ptt_e'].values / 1000.0
            # MAP prior: assume population mean MAP ~93 mmHg, use PTT_e variance
            map_prior = 93.0
            map_vals = map_prior * (np.mean(ptt_e_sec) / ptt_e_sec)
            X_map = map_vals.reshape(-1, 1)
            X_sbp = X_map * 1.4  # SBP ≈ 1.4 × MAP as population prior

        # ── PATH A: Regression ─────────────────────────────────────────────────
        def _fit(X, y, label):
            X = np.array(X).reshape(-1, 1)
            std = float(np.std(X))
            if std >= 0.5:  # lower threshold when no reference labels
                reg = LinearRegression().fit(X, y)
                alpha_mu = -2 * reg.coef_[0]
                beta0_mu = float(reg.intercept_)
                r2 = reg.score(X, y)
                if alpha_mu > 0.001:
                    return float(alpha_mu), beta0_mu, round(r2, 3)
            return None, None, 0.0

        alpha_sbp, beta0_sbp, r2_sbp = _fit(X_sbp, y_ln_ptt_a, 'SBP')
        alpha_map, beta0_map, r2_map = _fit(X_map, y_ln_ptt_e, 'MAP')

        # ── PATH B: Fallback ───────────────────────────────────────────────────
        def _fallback(y_ln_ptt, X, aging_index_mean):
            alpha = 0.03 + (aging_index_mean * 0.015) if not np.isnan(aging_index_mean) else 0.025
            alpha = float(np.clip(alpha, 0.01, 0.08))
            beta0 = float(np.mean(y_ln_ptt) + (alpha / 2) * np.mean(X))
            return alpha, beta0

        mean_agi = p_data['aging_index'].mean() if 'aging_index' in p_data.columns else np.nan

        if alpha_sbp is None:
            alpha_sbp, beta0_sbp = _fallback(y_ln_ptt_a, X_sbp, mean_agi)
            r2_sbp = 0.0
        if alpha_map is None:
            alpha_map, beta0_map = _fallback(y_ln_ptt_e, X_map, mean_agi)
            r2_map = 0.0

        # ── Monte Carlo: SBP from PTT_a ────────────────────────────────────────
        ptt_a_sec = p_data['ptt_a'].values / 1000.0
        ptt_a_mu = np.mean(ptt_a_sec)
        ptt_a_sigma = max(np.std(ptt_a_sec), 0.001)
        sampled_ptt_a = np.random.normal(ptt_a_mu, ptt_a_sigma, 5000)
        sim_sbp = (2 / alpha_sbp) * (beta0_sbp - np.log(sampled_ptt_a))
        clean_sbp = sim_sbp[(sim_sbp > 40) & (sim_sbp < 250)]

        if len(clean_sbp) > 100:
            sbp_mu_est, sbp_std_est = norm.fit(clean_sbp)
        else:
            sbp_mu_est = float(p_data['label_sbp'].mean())
            sbp_std_est = 5.0

        # ── Monte Carlo: MAP from PTT_e ────────────────────────────────────────
        ptt_e_sec = p_data['ptt_e'].values / 1000.0
        ptt_e_mu = np.mean(ptt_e_sec)
        ptt_e_sigma = max(np.std(ptt_e_sec), 0.001)
        sampled_ptt_e = np.random.normal(ptt_e_mu, ptt_e_sigma, 5000)
        sim_map = (2 / alpha_map) * (beta0_map - np.log(sampled_ptt_e))
        clean_map = sim_map[(sim_map > 40) & (sim_map < 180)]

        if len(clean_map) > 100:
            map_mu_est, map_std_est = norm.fit(clean_map)
        else:
            map_mu_est = float(np.mean(X_map))
            map_std_est = 5.0

        # ── DBP from SBP and MAP: DBP = 3*MAP - 2*SBP ─────────────────────────
        # Derived from: MAP = DBP + (SBP - DBP) / 3
        #               MAP = (2*DBP + SBP) / 3
        #               DBP = (3*MAP - SBP) / 2
        dbp_mu_est = (3 * map_mu_est - sbp_mu_est) / 2.0
        # Error propagation: var(DBP) = (9*var(MAP) + var(SBP)) / 4
        dbp_std_est = np.sqrt((9 * map_std_est ** 2 + sbp_std_est ** 2) / 4.0)

        # ── Biophysical derivation ─────────────────────────────────────────────
        k_geom = (self.L ** 2 * self.rho) / (np.exp(beta0_sbp) ** 2)
        avg_ct = p_data['crest_time'].mean() if 'crest_time' in p_data.columns else 100.0
        avg_auc = p_data['total_auc'].mean() if 'total_auc' in p_data.columns else 500.0
        morph_factor = (avg_ct / 100.0) / (avg_auc / 500.0) if avg_auc > 0 else 1.0
        e0_mu = 80000 * morph_factor
        hd_mu = k_geom / e0_mu

        return {
            'patient_id': patient_id,
            # Model params
            'alpha_sbp': round(float(alpha_sbp), 5),
            'alpha_map': round(float(alpha_map), 5),
            'alpha_mean': round(float(alpha_sbp), 5),  # keep for dashboard compat
            'alpha_std': round(float(alpha_sbp) * 0.1, 5),
            'beta0_mean': round(float(beta0_sbp), 4),
            'beta0_std': round(abs(float(beta0_sbp) * 0.01), 4),
            'e0_mean_kPa': round(e0_mu / 1000.0, 2),
            'e0_std_kPa': round(e0_mu * 0.1 / 1000.0, 2),
            'hd_ratio_mean': round(hd_mu, 4),
            'hd_ratio_std': round(hd_mu * 0.1, 4),
            # BP estimates
            'sbp_est_mean': float(sbp_mu_est),
            'sbp_est_std': float(sbp_std_est),
            'map_est_mean': float(map_mu_est),
            'map_est_std': float(map_std_est),
            'dbp_est_mean': float(dbp_mu_est),
            'dbp_est_std': float(dbp_std_est),
            'r2_sbp': round(r2_sbp, 3),
            'r2_map': round(r2_map, 3),
            'model_r2': round(r2_sbp, 3),  # keep for dashboard compat
        }


class FeatureExporter:
    @staticmethod
    def extract_training_data(samples: list[BPSample], ppg_waves, time_features):
        rows = []
        for sample, ppg_wave, time_feature in zip(samples, ppg_waves, time_features):
            from src.extractor.morphology import MorphologyExtractor

            # 1. Signal Processing
            t = sample.ppg_timestamps
            ppg_proc = sample.ppg
            vpg, apg, waves = ppg_wave

            # 3. Feature Mapping
            for f in time_feature:
                # Find the wave dictionary that corresponds to this time feature
                w = next((x for x in waves if x['a'] == f['a_idx']), None)
                if not w: continue

                # A. Crest Time
                crest_time = (w['ppg_peak'] - w['a']) * (1 / sample.target_fs)

                # B. Total AUC (Pulse Volume Proxy)
                # Dynamic windowing: From 'a' (start) to 'a' of next wave (if avail) or +800ms
                current_idx = waves.index(w)
                if current_idx + 1 < len(waves):
                    pulse_end = waves[current_idx + 1]['a']
                else:
                    pulse_end = w['a'] + int(sample.target_fs * 0.8)

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
                    'ptt_e': f['ptt_e'],
                    'crest_time': round(crest_time, 2),
                    'total_auc': round(total_auc, 4),
                    'sys_auc': round(sys_auc, 4) if sys_auc else None,
                    'aging_index': MorphologyExtractor.calculate_aging_index(apg, w),
                    'label_sbp': sample.bps,
                    'label_dbp': sample.bpd
                })

        return pd.DataFrame(rows)