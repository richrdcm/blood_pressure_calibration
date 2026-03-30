import numpy as np
from scipy.stats import norm


class PhysiologicalWaveFinder:
    def __init__(self, fs=125):
        self.fs = fs
        # Default biophysical constants (calibrated per patient later)
        self.alpha = 0.017  # Stiffness coefficient
        self.L = 0.6  # Estimated heart-to-finger distance (meters)
        self.E_factor = 5.0  # Baseline elasticity factor

    def predict_ptt_physics(self, pressure):
        """Calculates expected PTT based on Moens-Korteweg physics."""
        # Simplified MK: PWV = sqrt(k * exp(alpha * P))
        pwv = self.E_factor * np.exp((self.alpha * pressure) / 2)
        ptt_seconds = self.L / pwv
        return ptt_seconds * 1000  # Convert to ms

    def find_waves_probabilistic(self, ppg_signal, r_peaks, bps, bpd):
        results = []

        # 1. Physics-based Expected Times
        t_sys_mu = self.predict_ptt_physics(bps)
        t_dia_mu = self.predict_ptt_physics(bpd) + 150  # Diastolic reflection delay

        # Standard deviation (uncertainty) - starts wider, narrows with calibration
        sigma = 25  # ms

        for r_idx in r_peaks:
            # 2. Define Search Window based on Gaussian Probability (3-sigma)
            window_size = int((3 * sigma / 1000) * self.fs)

            sys_center = int((t_sys_mu / 1000) * self.fs)
            dia_center = int((t_dia_mu / 1000) * self.fs)

            # 3. Find Maxima within Physical Constraints
            try:
                # Systolic Peak Search
                sys_slice = ppg_signal[r_idx + sys_center - window_size: r_idx + sys_center + window_size]
                actual_sys_idx = np.argmax(sys_slice) + (r_idx + sys_center - window_size)

                # Diastolic Peak Search
                dia_slice = ppg_signal[r_idx + dia_center - window_size: r_idx + dia_center + window_size]
                actual_dia_idx = np.argmax(dia_slice) + (r_idx + dia_center - window_size)

                # 4. Bayesian Update: Calculate the "Surprise" (Error)
                found_ptt = (actual_sys_idx - r_idx) * (1000 / self.fs)
                error = abs(found_ptt - t_sys_mu)

                # SQI Score based on how well it fits the physics
                # A peak that breaks the laws of physics gets a score of 0
                prob_fit = norm.pdf(found_ptt, loc=t_sys_mu, scale=sigma) / norm.pdf(t_sys_mu, loc=t_sys_mu,
                                                                                     scale=sigma)

                results.append({
                    'r_peak': r_idx,
                    'sys_peak': actual_sys_idx,
                    'dia_peak': actual_dia_idx,
                    'ptt_predicted': t_sys_mu,
                    'ptt_actual': found_ptt,
                    'phys_fit_score': round(prob_fit * 100, 2)
                })
            except Exception:
                continue

        return results