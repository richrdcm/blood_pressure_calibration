import numpy as np
from schemas.bp_schema import BPSample
import neurokit2 as nk

class TimeExtractor:
    """
    PTTa (R→a): The gold standard for arterial stiffness and SBP.
    PTTsys(R→Peak): Total systolic transit.
    PTTe(R→e): Aortic valve closure/Mean pressure.
    PTTdia(R→dp): Reflected wave speed, crucial for DBP.
    Reflection Time (dp−sys): A measure of peripheral resistance.
    """

    def compute_ptts(self, ppg_derivatives: list, samples: list) -> list:
        ptts = []
        for (vpg, apg, waves), sample in zip(ppg_derivatives, samples):
            r_peaks = self.extract_ecg_r_peaks(ecg_signal=sample.ecg, fs=sample.ecg_fs)
            ptts.append(self.compute_heartbeat_times(r_peaks, waves, sample))
        return ptts

    @staticmethod
    def extract_ecg_r_peaks(ecg_signal: np.ndarray, fs: int) -> np.ndarray:
        _, r_peak_info = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method="neurokit")
        r_peaks = r_peak_info["ECG_R_Peaks"]
        return r_peaks

    @staticmethod
    def compute_heartbeat_times(r_peaks, waves, sample):
        time_features = []
        timestamps = sample.ppg_timestamps  # Use the actual unified time array!

        # THE FIX: Define the absolute boundary of our array
        max_idx = len(timestamps) - 1

        for w in waves:
            # 1. Safely get the a-wave index and ensure it's inside the array
            a_idx = w.get('a')
            if a_idx is None or a_idx > max_idx:
                continue  # Skip this beat completely if missing or out of bounds

            # Find the closest R-peak BEFORE this pulse that is also within bounds
            prev_r = [r for r in r_peaks if r < a_idx and r <= max_idx]
            if not prev_r:
                continue

            r_idx = prev_r[-1]
            r_time = timestamps[r_idx]

            # 2. PTT_a: Electrical-to-Physical Ejection
            ptt_a = (timestamps[a_idx] - r_time)

            # 3. PTT_sys: Pulse Arrival Time
            sys_idx = w.get('ppg_peak')
            ptt_sys = (timestamps[sys_idx] - r_time) if (sys_idx is not None and sys_idx <= max_idx) else None

            # 4. PTT_e: Time to Aortic Valve Closure
            e_idx = w.get('e')
            ptt_e = (timestamps[e_idx] - r_time) if (e_idx is not None and e_idx <= max_idx) else None

            # 5. PTT_dia: Time to Reflected Wave Arrival
            dp_idx = TimeExtractor._get_dp_for_beat(sample.ppg, w, waves, sample.ppg_fs)
            ptt_dia = (timestamps[dp_idx] - r_time) if (dp_idx is not None and dp_idx <= max_idx) else None

            # 6. Reflection Time (Internal PPG timing)
            reflection_time = ptt_dia - ptt_sys if (ptt_dia is not None and ptt_sys is not None) else None

            # Safely append only the valid points
            time_features.append({
                'r_peak': r_idx,
                'a_idx': a_idx,
                'sys_idx': sys_idx if (sys_idx is not None and sys_idx <= max_idx) else None,
                'e_idx': e_idx if (e_idx is not None and e_idx <= max_idx) else None,
                'dp_idx': dp_idx if (dp_idx is not None and dp_idx <= max_idx) else None,

                'ptt_a': round(ptt_a, 2),
                'ptt_sys': round(ptt_sys, 2) if ptt_sys is not None else None,
                'ptt_e': round(ptt_e, 2) if ptt_e is not None else None,
                'ptt_dia': round(ptt_dia, 2) if ptt_dia is not None else None,
                'reflection_time': round(reflection_time, 2) if reflection_time is not None else None
            })

        return time_features

    @staticmethod
    def _get_dp_for_beat(ppg_signal, current_w, all_waves, fs):
        """Helper to find the diastolic peak using your morphology class logic"""
        from src.extractor.morphology import MorphologyExtractor

        try:
            curr_idx = all_waves.index(current_w)
            next_w = all_waves[curr_idx + 1] if curr_idx + 1 < len(all_waves) else None
        except ValueError:
            next_w = None

        return MorphologyExtractor.find_diastolic_peak(ppg_signal, current_w, next_w, fs)