import numpy as np
from scipy.signal import find_peaks


class MorphologyExtractor:
    @staticmethod
    def extract_waves(ppg_signal, fs):
        """
        For a robust biophysical model, tune the distance based on the
        Physiological Range of the human heart.
        Scenario,       Max BPM,    Min Pulse Interval, Tuning Recommendation (fs=125)
        Resting,        100 BPM,    600ms,              distance = int(fs * 0.5) (Safe)
        Stress/Exercise,180 BPM,    333ms,              distance = int(fs * 0.3) (Aggressive)
        Bradycardia,    40 BPM,     1500ms,             distance = int(fs * 0.8) (Conservative)

        :param ppg_signal:
        :param fs:
        :return:
        """
        vpg = np.gradient(ppg_signal)
        apg = np.gradient(vpg)

        # Find global PPG peaks (Volume Peaks) for windowing
        ppg_peaks, _ = find_peaks(
            ppg_signal,
            distance=int(fs * 0.35),
            height=np.mean(ppg_signal) * 0.5,  # Height threshold helps ignore noise
            prominence=np.std(ppg_signal) * 0.2  # Prominence is key for PPG!
        )

        results = []
        for p_idx in ppg_peaks:
            # 1. 'a' wave: Max acceleration in the 150ms before PPG peak
            search_a = max(0, p_idx - int(fs * 0.15))
            a_idx = np.argmax(apg[search_a:p_idx]) + search_a

            # 2. Search for b, c, d, e in the 500ms after 'a' wave
            search_end = min(a_idx + int(fs * 0.5), len(apg))
            cycle_apg = apg[a_idx:search_end]

            # Local Maxima (c, e) and Local Minima (b, d)
            peaks, _ = find_peaks(cycle_apg)
            valleys, _ = find_peaks(-cycle_apg)

            # Map indices back to global time
            try:
                b = valleys[0] + a_idx if len(valleys) > 0 else None
                c = peaks[0] + a_idx if len(peaks) > 0 else None
                d = valleys[1] + a_idx if len(valleys) > 1 else None
                e = peaks[1] + a_idx if len(peaks) > 1 else None  # Dicrotic Notch

                results.append({
                    'ppg_peak': p_idx,
                    'a': a_idx, 'b': b, 'c': c, 'd': d, 'e': e
                })
            except IndexError:
                continue

        return vpg, apg, results

    @staticmethod
    def calculate_aging_index(apg_signal, wave_indices):
        """
        Calculates the Aging Index: (b - c - d - e) / a
        How to Interpret the Aging Index
        AGI Value	    Vascular Condition	    Typical Observation
        Below −0.6	    Excellent	            Young, highly elastic arteries. Large c and d waves.
        −0.6 to 0.0	    Good/Healthy	        Normal adult vascularity.
        0.0 to 0.5	    Stiffening	            Common in hypertension. The b wave becomes less negative.
        Above 0.5	    Atherosclerosis	        Very stiff arteries. The c,d,e waves often "flatten out.
        """
        w = wave_indices
        try:
            # Get amplitudes (y-values) from the APG signal
            a = apg_signal[w['a']]
            b = apg_signal[w['b']] if w['b'] else 0
            c = apg_signal[w['c']] if w['c'] else 0
            d = apg_signal[w['d']] if w['d'] else 0
            e = apg_signal[w['e']] if w['e'] else 0

            # The standard clinical formula
            agi = (b - c - d - e) / a
            return round(agi, 3)
        except (TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def find_diastolic_peak(ppg_signal, wave_indices, next_wave_indices, fs):
        # --- FIX: ROBUST KEY ACCESS ---
        # This works for dicts AND numpy object-arrays
        try:
            e_idx = wave_indices['e']
        except (KeyError, TypeError, IndexError):
            return None

        if e_idx is None:
            return None

        # Determine the 'Wall' (start of next heartbeat)
        if next_wave_indices is not None:
            try:
                wall_idx = next_wave_indices['a']
            except (KeyError, TypeError, IndexError):
                wall_idx = len(ppg_signal)
        else:
            # Fallback: 450ms search window
            wall_idx = min(e_idx + int(fs * 0.45), len(ppg_signal))

        # Define the Search Slice
        search_start = e_idx + 2
        search_end = wall_idx

        if search_start >= search_end:
            return None

        # Slice and find local maximum
        search_area = ppg_signal[search_start:search_end]
        if len(search_area) == 0:
            return None

        rel_idx = np.argmax(search_area)
        return rel_idx + search_start


class PhysiologicalWaveFinder:
    def __init__(self, alpha, intercept, fs=125):
        self.alpha = alpha
        self.intercept = intercept
        self.fs = fs

    def get_physical_belief(self, bps, r_peak_idx):
        """
        Predicts where the systolic peak SHOULD be based on physics.
        ln(PTT) = (-alpha/2) * BPS + Intercept
        """
        beta_1 = -self.alpha / 2
        pred_log_ptt = (beta_1 * bps) + self.intercept
        pred_ptt_ms = np.exp(pred_log_ptt)

        # Convert ms delay to sample index
        delay_samples = int((pred_ptt_ms / 1000) * self.fs)
        return r_peak_idx + delay_samples

    def find_waves_with_belief(self, ppg_signal, r_peaks, bps):
        """
        Search for peaks ONLY within the physical belief window.
        """
        predictions = []
        # Search window +/- 40ms around the physical prediction
        window_size = int(0.04 * self.fs)

        for r_idx in r_peaks:
            t_pred_idx = self.get_physical_belief(bps, r_idx)

            # Slice the search area
            start, end = t_pred_idx - window_size, t_pred_idx + window_size
            search_slice = ppg_signal[max(0, start): min(len(ppg_signal), end)]

            if len(search_slice) > 0:
                actual_idx = np.argmax(search_slice) + start
                predictions.append({
                    'r_peak': r_idx,
                    'predicted_idx': t_pred_idx,
                    'actual_idx': actual_idx,
                    'error_ms': (actual_idx - t_pred_idx) * (1000 / self.fs)
                })
        return predictions