import numpy as np
from scipy.signal import find_peaks
from schemas.bp_schema import BPSample


class MorphologyExtractor:
    def extract_samples_waves(self, samples: list[BPSample]):
        waves = []
        for sample in samples:
            waves.append(self.extract_sample_waves(ppg_signal=sample.ppg,
                                                   fs=sample.target_fs,
                                                   intervals=sample.joint_sqi))
        return waves

    @staticmethod
    def extract_sample_waves(ppg_signal, fs, intervals=None):
        """
        Extracts APG morphology waves from the PPG signal with robust,
        physiologically-constrained detection for each fiducial point.

        Correct anatomical ordering:
            a < b < ppg_peak < c < d < e (dicrotic notch) < dp (diastolic peak)

        Each point is searched within a physiological time window:
            a  — max APG in [150ms before ppg_peak]           → always positive
            b  — min APG in [a, a+150ms], before ppg_peak     → always negative
            c  — max APG in [ppg_peak, ppg_peak+200ms]        → positive (re-acceleration)
            d  — min APG in [c, c+200ms]                      → negative
            e  — max APG in [d, d+300ms]  (dicrotic notch)    → positive, lower than a
            dp — first local PPG max after e, before next 'a'
        """
        vpg = np.gradient(ppg_signal)
        apg = np.gradient(vpg)
        ppg_arr = np.array(ppg_signal)
        n = len(ppg_arr)

        ppg_peaks, _ = find_peaks(
            ppg_arr,
            distance=int(fs * 0.35),
            height=np.mean(ppg_arr) * 0.5,
            prominence=np.std(ppg_arr) * 0.2,
        )

        # Filter to joint SQI intervals
        if intervals:
            usable = np.zeros(n, dtype=bool)
            for s, e in intervals:
                usable[max(0, s): min(e, n - 1) + 1] = True
            ppg_peaks = ppg_peaks[usable[ppg_peaks]]
            if len(ppg_peaks) == 0:
                print("[MorphologyExtractor] No PPG peaks found within joint SQI intervals.")
                return vpg, apg, []

        # ── Helper: find best peak/valley in a window with constraints ────────
        def _find_extremum(signal, start, end, mode='peak', min_val=None, max_val=None):
            """
            Find the best local extremum in signal[start:end].
            mode: 'peak' (local max) or 'valley' (local min)
            min_val/max_val: optional amplitude constraints on the found point.
            Returns absolute index or None.
            """
            start = max(0, int(start))
            end = min(n, int(end))
            if end - start < 3:
                return None

            segment = signal[start:end]
            candidates, _ = find_peaks(segment if mode == 'peak' else -segment,
                                       prominence=np.std(segment) * 0.05)

            if len(candidates) == 0:
                # Fallback to argmax/argmin if no local extremum found
                idx = int(np.argmax(segment) if mode == 'peak' else np.argmin(segment))
                abs_idx = idx + start
            else:
                abs_idx = candidates[0] + start

            val = signal[abs_idx]
            if min_val is not None and val < min_val:
                return None
            if max_val is not None and val > max_val:
                return None
            return abs_idx

        # ── Extract raw waves with physiological windows ───────────────────────
        raw_results = []
        for p_idx in ppg_peaks:

            # ── 'a': max APG in [ppg_peak - 200ms, ppg_peak] ─────────────────
            a_start = max(0, p_idx - int(fs * 0.20))
            a_idx = _find_extremum(apg, a_start, p_idx, mode='peak', min_val=0)
            if a_idx is None:
                continue

            # ── 'b': min APG in [a, a + 150ms], must be before ppg_peak ──────
            b_end = min(p_idx, a_idx + int(fs * 0.15))
            b_idx = _find_extremum(apg, a_idx + 1, b_end, mode='valley', max_val=0)
            # b is optional — don't skip beat if missing

            # ── 'c': max APG in [ppg_peak, ppg_peak + 200ms] ─────────────────
            c_end = min(n, p_idx + int(fs * 0.20))
            c_idx = _find_extremum(apg, p_idx + 1, c_end, mode='peak', min_val=0)
            # c is optional

            # ── 'd': min APG in [c, c + 200ms] ───────────────────────────────
            d_idx = None
            if c_idx is not None:
                d_end = min(n, c_idx + int(fs * 0.20))
                d_idx = _find_extremum(apg, c_idx + 1, d_end, mode='valley', max_val=0)

            # ── 'e': max APG in [d, d + 300ms] — dicrotic notch ──────────────
            # Must be lower amplitude than 'a' (it's a secondary acceleration)
            e_idx = None
            if d_idx is not None:
                e_end = min(n, d_idx + int(fs * 0.30))
                a_amp = apg[a_idx]
                e_idx = _find_extremum(apg, d_idx + 1, e_end,
                                       mode='peak',
                                       min_val=0,
                                       max_val=a_amp * 0.9)  # e < a amplitude

            raw_results.append({
                'ppg_peak': int(p_idx),
                'a': int(a_idx),
                'b': int(b_idx) if b_idx is not None else None,
                'c': int(c_idx) if c_idx is not None else None,
                'd': int(d_idx) if d_idx is not None else None,
                'e': int(e_idx) if e_idx is not None else None,
            })

        # ── Morphology validation + PTT requirements + diastolic peak ─────────
        results = []
        rejected = {'morphology': 0, 'ptt': 0, 'diastolic': 0}

        # First pass: validate morphology and PTT only
        validated = []
        for wave in raw_results:
            if not MorphologyExtractor.validate_wave_morphology(wave, apg, ppg_arr, fs):
                rejected['morphology'] += 1
                continue
            if not MorphologyExtractor.has_ptt_requirements(wave):
                rejected['ptt'] += 1
                continue
            if not wave['e']:
                rejected['diastolic'] += 1
                continue
            validated.append(wave)

        # Build a lookup of raw ppg_peak positions for boundary purposes
        raw_peaks_sorted = sorted([w['ppg_peak'] for w in raw_results if w.get('ppg_peak') is not None])

        # Second pass: find diastolic peak using next RAW peak as hard wall
        """
        for i, wave in enumerate(validated):
            current_peak = wave['ppg_peak']

            # Find the next raw ppg_peak strictly after the current one
            next_raw_peak = next(
                (p for p in raw_peaks_sorted if p > current_peak), None
            )

            # Build a minimal boundary dict for find_diastolic_peak
            next_boundary = {'a': next_raw_peak} if next_raw_peak is not None else None

            dp_idx = MorphologyExtractor.find_diastolic_peak(ppg_arr, wave, next_boundary, fs)

            if dp_idx is None:
                rejected['diastolic'] += 1
                continue

            wave['dp'] = dp_idx
            results.append(wave)
        """
        print(f"[MorphologyExtractor] {len(validated)} valid beats from "
              f"{len(raw_results)} detected — rejected: "
              f"{rejected['morphology']} morphology, "
              f"{rejected['ptt']} missing PTT fields, "
              f"{rejected['diastolic']} no diastolic peak.")

        return vpg, apg, validated

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
    def _find_next_foot(ppg: np.ndarray, current_peak: int, next_peak: int = None, fs: float = None):
        """Find the foot (minimum) of the PPG after current_peak, before next_peak."""
        if next_peak is not None:
            end = next_peak
        else:
            # Reasonable maximum search: 1.2 seconds after systolic peak
            end = min(len(ppg), current_peak + int(1.2 * fs)) if fs else len(ppg)
        segment = ppg[current_peak:end]
        if len(segment) < 3:
            return None
        # The foot is the global minimum in this region (start of next upstroke)
        foot_rel = np.argmin(segment)
        return foot_rel + current_peak


    def find_diastolic_peak(ppg_signal: np.ndarray, wave: dict, next_wave: dict, fs: float, debug: bool = False):
        """
        Finds the diastolic peak (reflected wave) after the dicrotic notch (e).
        """
        ppg_arr = np.array(ppg_signal)
        e_idx = wave.get('e')
        ppg_peak = wave.get('ppg_peak')

        if e_idx is None or ppg_peak is None:
            return None

        # ── Search boundaries ────────────────────────────────────────────────
        # Hard wall: next beat's 'a' wave (never cross into next beat)
        next_a = next_wave.get('a') if next_wave is not None else None

        # Allow up to 700ms after systolic peak (covers elderly/stiff arteries)
        max_delay_samples = int(0.7 * fs)
        absolute_end = min(len(ppg_arr), ppg_peak + max_delay_samples)

        if next_a is not None and next_a > e_idx:
            search_end = min(next_a - 1, absolute_end)
        else:
            search_end = absolute_end

        search_start = e_idx + 1

        if debug:
            print(f"  [dp search] e_idx={e_idx}, ppg_peak={ppg_peak}, "
                  f"search=[{search_start}, {search_end}], "
                  f"window_ms={(search_end - search_start) * 1000 / fs:.0f}ms")

        if search_start >= search_end:
            if debug: print("  [dp] rejected: search window empty")
            return None

        search_area = ppg_arr[search_start:search_end]
        if len(search_area) < 3:
            if debug: print("  [dp] rejected: search area too short")
            return None

        # ── Find local peaks — use absolute height floor instead of std ───────
        # std-based prominence fails when the search area is nearly flat (old patients)
        e_amp = ppg_arr[e_idx]
        sys_amp = ppg_arr[ppg_peak]
        min_height = e_amp + (sys_amp - e_amp) * 0.02  # just 2% above notch

        local_peaks, props = find_peaks(
            search_area,
            height=min_height,
            prominence=(sys_amp - e_amp) * 0.02,  # 2% of systolic-notch range
            distance=int(fs * 0.05),  # peaks at least 50ms apart
        )

        if debug:
            print(f"  [dp] e_amp={e_amp:.4f}, sys_amp={sys_amp:.4f}, "
                  f"min_height={min_height:.4f}, local_peaks={local_peaks}")

        if len(local_peaks) > 0:
            dp_rel = local_peaks[0]
            dp_idx = dp_rel + search_start
        else:
            # Fallback: global max in search window
            max_rel = int(np.argmax(search_area))
            dp_idx = max_rel + search_start

            # Must be meaningfully above the notch, not just noise
            if ppg_arr[dp_idx] <= min_height:
                if debug: print(
                    f"  [dp] fallback rejected: argmax={ppg_arr[dp_idx]:.4f} <= min_height={min_height:.4f}")
                return None

        # ── Final validations ─────────────────────────────────────────────────
        if ppg_arr[dp_idx] >= sys_amp:
            if debug: print(f"  [dp] rejected: dp_amp={ppg_arr[dp_idx]:.4f} >= sys_amp={sys_amp:.4f}")
            return None

        if dp_idx <= e_idx:
            if debug: print(f"  [dp] rejected: dp_idx={dp_idx} <= e_idx={e_idx}")
            return None

        if next_a is not None and dp_idx >= next_a:
            if debug: print(f"  [dp] rejected: dp_idx={dp_idx} >= next_a={next_a}")
            return None

        # Physiological delay: 100–700ms after systolic peak
        delay_ms = (dp_idx - ppg_peak) * (1000.0 / fs)
        if not (100 <= delay_ms <= 700):
            if debug: print(f"  [dp] rejected: delay_ms={delay_ms:.1f} outside [100, 700]")
            return None

        if debug:
            print(f"  [dp] FOUND at idx={dp_idx}, delay={delay_ms:.1f}ms, "
                  f"amp={ppg_arr[dp_idx]:.4f}")

        return dp_idx

    @staticmethod
    def validate_wave_morphology(wave: dict, apg: np.ndarray, ppg: np.ndarray, fs: float) -> bool:
        """
        Validates that a detected wave dict is physiologically plausible.

        Correct anatomical ordering:
            a < b < ppg_peak < c < d < e (dicrotic notch) < dp (diastolic peak)

        Where:
            a  — max acceleration (start of systolic upstroke)         [pre-peak]
            b  — min acceleration (end of systolic upstroke)           [pre-peak]
            ppg_peak — systolic peak
            c  — re-acceleration after systolic peak                   [post-peak]
            d  — deceleration after c                                  [post-peak]
            e  — dicrotic notch (aortic valve closure)                 [post-peak]
            dp — diastolic peak (reflected wave)                       [post-peak, after e]

        Checks:
          1. Required indices present   — ppg_peak and 'a' must exist
          2. Pre-peak ordering          — a < b < ppg_peak
          3. Post-peak ordering         — ppg_peak < c < d < e
          4. APG polarity               — a > 0 (positive), b < 0 (negative)
          5. Crest time plausibility    — a → ppg_peak in [20, 400] ms
          6. b plausibility             — b is between a and ppg_peak
          7. e plausibility             — dicrotic notch within [50, 600] ms after ppg_peak
        """
        ppg_peak = wave.get('ppg_peak')
        a_idx = wave.get('a')
        b_idx = wave.get('b')
        c_idx = wave.get('c')
        d_idx = wave.get('d')
        e_idx = wave.get('e')

        max_idx = len(ppg) - 1

        # 1. Required fields and bounds
        if ppg_peak is None or a_idx is None:
            return False
        if ppg_peak > max_idx or a_idx > max_idx:
            return False

        # 2. Pre-peak ordering: a < ppg_peak
        if a_idx >= ppg_peak:
            return False

        # b is optional but if present must satisfy: a < b < ppg_peak
        if b_idx is not None:
            if not (a_idx < b_idx < ppg_peak):
                return False

        # 3. Post-peak ordering: ppg_peak < c < d < e (each optional but ordered)
        post_peak_ordered = [idx for idx in [c_idx, d_idx, e_idx] if idx is not None]
        if any(idx <= ppg_peak for idx in post_peak_ordered):
            return False
        if post_peak_ordered != sorted(post_peak_ordered):
            return False

        # 4. APG polarity
        if a_idx < len(apg) and apg[a_idx] <= 0:
            return False
        if b_idx is not None and b_idx < len(apg) and apg[b_idx] >= 0:
            return False

        # 5. Crest time: a → ppg_peak must be physiologically plausible
        crest_time_ms = (ppg_peak - a_idx) * (1000.0 / fs)
        if not (20 <= crest_time_ms <= 400):
            return False

        # 6. b must be reasonably close to ppg_peak (within last 200ms of upstroke)
        if b_idx is not None:
            b_to_peak_ms = (ppg_peak - b_idx) * (1000.0 / fs)
            if b_to_peak_ms > 200:
                return False

        # 7. Dicrotic notch (e) within physiological window after systolic peak
        if e_idx is not None:
            e_delay_ms = (e_idx - ppg_peak) * (1000.0 / fs)
            if not (50 <= e_delay_ms <= 600):
                return False

        return True

    @staticmethod
    def has_ptt_requirements(wave: dict) -> bool:
        """
        Checks if a wave has the minimum fields needed for PTT computation:
          - ppg_peak  → PTT_sys
          - a         → PTT_a  (most important — used for BP calibration)
        Optional but scored:
          - e         → PTT_e
        Returns True if at least ppg_peak + a are present.
        """
        return wave.get('ppg_peak') is not None and wave.get('a') is not None


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
        delay_samples = int((pred_ptt_ms) * self.fs)
        return r_peak_idx + delay_samples

    def find_waves_with_belief(self, ppg_signal, r_peaks, bps):
        """
        Search for peaks ONLY within the physical belief window.
        """
        predictions = []
        # Search window +/- 40ms around the physical prediction
        window_size = int(40 * self.fs)

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
                    'error_ms': (actual_idx - t_pred_idx) * (1 / self.fs)
                })
        return predictions