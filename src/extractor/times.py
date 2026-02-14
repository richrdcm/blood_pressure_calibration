import numpy as np


class TimeExtractor:
    """
    PTTa (R→a): The gold standard for arterial stiffness and SBP.
    PTTsys(R→Peak): Total systolic transit.
    PTTe(R→e): Aortic valve closure/Mean pressure.
    PTTdia(R→dp): Reflected wave speed, crucial for DBP.
    Reflection Time (dp−sys): A measure of peripheral resistance.
    """
    @staticmethod
    def compute_heartbeat_times(r_peaks, waves, fs, ppg_proc):
        """
        Computes the time intervals (PTTs) between the ECG R-peak and
        various fiducial points on the PPG signal.
        Metric,     Start → End,        Hemodynamic Meaning
        ptt_a,      R → a,                PEP + PTT: Measures the speed of the very first pressure wave front. Best for SBP.
        ptt_sys,    R → Systolic Peak,    Full Systolic Transit: Measures the arrival of maximum blood volume.
        ptt_e,      R → e,                Valve Dynamics: Represents the end of the heart's active ejection phase.
        ptt_dia,    R → dp,               "Reflected Wave Speed: Shows how fast the ""Echo"" returns from the lower body. Best for DBP."
        """
        time_features = []

        for w in waves:
            # Find the closest R-peak that occurred BEFORE this pulse a-wave
            # This ensures we are tethered to the correct electrical trigger
            prev_r = [r for r in r_peaks if r < w['a']]
            if not prev_r:
                continue
            r_idx = prev_r[-1]

            # 1. PTT_a: Electrical-to-Physical Ejection (Heart to Artery Start)
            # Highly sensitive to SBP and cardiac contractility
            ptt_a = (w['a'] - r_idx) * (1000 / fs)

            # 2. PTT_sys: Pulse Arrival Time (Heart to Finger Peak)
            # The standard PTT used for SBP estimation
            ptt_sys = (w['ppg_peak'] - r_idx) * (1000 / fs)

            # 3. PTT_e: Time to Aortic Valve Closure (Notch)
            # Proxy for Mean Arterial Pressure
            ptt_e = (w['e'] - r_idx) * (1000 / fs) if w['e'] else None

            # 4. PTT_dia: Time to Reflected Wave Arrival (Diastolic Peak)
            # Critical for DBP estimation and peripheral resistance
            # We find the DP using your existing MorphologyExtractor logic
            dp_idx = TimeExtractor._get_dp_for_beat(ppg_proc, w, waves, fs)
            ptt_dia = (dp_idx - r_idx) * (1000 / fs) if dp_idx else None

            # 5. Reflection Time (Internal PPG timing)
            # T_reflection = T_dp - T_sys
            reflection_time = ptt_dia - ptt_sys if (ptt_dia and ptt_sys) else None

            time_features.append({
                'r_peak': r_idx,
                'a_idx': w['a'],
                'ptt_a': round(ptt_a, 2),
                'ptt_sys': round(ptt_sys, 2),
                'ptt_e': round(ptt_e, 2) if ptt_e else None,
                'ptt_dia': round(ptt_dia, 2) if ptt_dia else None,
                'reflection_time': round(reflection_time, 2) if reflection_time else None
            })

        return time_features

    @staticmethod
    def _get_dp_for_beat(ppg_signal, current_w, all_waves, fs):
        """Helper to find the diastolic peak using your morphology class logic"""
        from src.extractor.morphology import MorphologyExtractor

        # Find the next wave to set the boundary wall
        try:
            curr_idx = all_waves.index(current_w)
            next_w = all_waves[curr_idx + 1] if curr_idx + 1 < len(all_waves) else None
        except ValueError:
            next_w = None

        return MorphologyExtractor.find_diastolic_peak(ppg_signal, current_w, next_w, fs)