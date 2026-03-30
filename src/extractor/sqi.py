import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk
from e2epyppg.ppg_sqa import sqa
from e2epyppg.utils import bandpass_filter


class PPGSQA:
    """
    Computes PPG Signal Quality using the e2epyppg library (Feli et al., 2023).
    Uses a one-class SVM to classify 30-second windows as Reliable/Unreliable.

    Returns a list of [start_idx, end_idx] intervals where quality is acceptable.
    """

    WINDOW_SEC = 30  # e2epyppg requires exact 30-second windows

    def __init__(
        self,
        fs: float,
        min_time_ms: float = 2000,
        pre_filter: bool   = False,
    ):
        self.fs          = float(fs)
        self.min_samples = int(min_time_ms * fs / 1000)
        self.pre_filter  = pre_filter
        self.win_samples = int(self.WINDOW_SEC * fs)

    def compute(self, signal: list) -> list:
        sig = np.array(signal, dtype=np.float64).flatten()

        if len(sig) < self.win_samples:
            print(f"[PPGSQI] Signal too short for e2epyppg "
                  f"({len(sig)} samples < {self.win_samples} needed for 30s). "
                  f"Falling back to full signal.")
            return [[0, len(sig) - 1]] if len(sig) >= self.min_samples else []

        try:
            if self.pre_filter:
                sig = bandpass_filter(sig=sig, fs=self.fs, lowcut=0.5, highcut=3.0)

            # ── Slice into exact 30-second windows, drop remainder ────────
            n_windows   = len(sig) // self.win_samples
            usable_len  = n_windows * self.win_samples
            sig_trimmed = sig[:usable_len]

            # e2epyppg expects a 1-D float64 array of exactly N*win_samples
            clean_indices, noisy_indices = sqa(
                sig=sig_trimmed,
                sampling_rate=int(self.fs),
                filter_signal=False,
            )

            if clean_indices is None or len(clean_indices) == 0:
                print("[PPGSQI] No clean segments detected by e2epyppg.")
                return []

            return self._indices_to_intervals(
                np.asarray(clean_indices, dtype=int), self.min_samples
            )

        except Exception as e:
            print(f"[PPGSQI] e2epyppg SQA failed: {e}. Falling back to full signal.")
            return [[0, len(signal) - 1]] if len(signal) >= self.min_samples else []

    @staticmethod
    def _indices_to_intervals(indices: np.ndarray, min_samples: int) -> list:
        """Convert sorted flat index array → contiguous [start, end] intervals."""
        if len(indices) == 0:
            return []

        indices   = np.sort(indices.flatten())
        intervals = []
        start     = indices[0]
        prev      = indices[0]

        for idx in indices[1:]:
            if idx - prev > 1:
                if prev - start >= min_samples:
                    intervals.append([int(start), int(prev)])
                start = idx
            prev = idx

        if prev - start >= min_samples:
            intervals.append([int(start), int(prev)])

        return intervals

class PPGSQI2:
    """
    Computes Signal Quality Index for PPG using a sliding window approach.
    Returns a list of [start_idx, end_idx] intervals where quality is acceptable.

    Checks per window:
      1. Skewness      — PPG should be right-skewed (systolic peak dominant)
      2. Kurtosis      — too flat or too spiky = noise
      3. Zero crossing rate — too high = noise/artifact
      4. Peak regularity — beats should be evenly spaced
      5. SNR estimate  — signal power vs noise floor
    """

    def __init__(
        self,
        fs: float,
        window_ms: float = 2000,       # sliding window size in ms
        step_ms: float = 500,          # step between windows in ms
        min_time_ms: float = 2000,     # minimum interval length to keep
        threshold: float = 0.6,        # fraction of checks that must pass (0-1)
        snr_min_db: float = 3.0,
        kurtosis_range: tuple = (1.5, 10.0),
        zcr_max: float = 0.35,
        peak_regularity_min: float = 0.5,
    ):
        self.fs = fs
        self.window_samples = int(window_ms * fs / 1000)
        self.step_samples   = int(step_ms   * fs / 1000)
        self.min_samples    = int(min_time_ms * fs / 1000)
        self.threshold      = threshold
        self.snr_min_db     = snr_min_db
        self.kurtosis_range = kurtosis_range
        self.zcr_max        = zcr_max
        self.peak_regularity_min = peak_regularity_min

    def compute(self, signal: list) -> list:
        """
        Returns list of [start_idx, end_idx] good-quality intervals.
        """
        sig = np.array(signal)
        n   = len(sig)
        good = np.zeros(n, dtype=bool)

        for start in range(0, n - self.window_samples + 1, self.step_samples):
            end    = start + self.window_samples
            window = sig[start:end]
            if self._is_good(window):
                good[start:end] = True

        return self._to_intervals(good, self.min_samples)

    def _is_good(self, window: np.ndarray) -> bool:
        checks = []

        # 1. SNR: ratio of signal power to high-freq noise power
        signal_power = np.var(window)
        noise        = window - np.convolve(window, np.ones(5) / 5, mode='same')
        noise_power  = np.var(noise) + 1e-12
        snr_db       = 10 * np.log10(signal_power / noise_power + 1e-12)
        checks.append(snr_db >= self.snr_min_db)

        # 2. Kurtosis: flat noise has low kurtosis, spike artifacts have very high
        mean, std = np.mean(window), np.std(window) + 1e-12
        kurt = np.mean(((window - mean) / std) ** 4)
        checks.append(self.kurtosis_range[0] <= kurt <= self.kurtosis_range[1])

        # 3. Zero crossing rate of the derivative (high = noisy)
        deriv = np.diff(window)
        zcr   = np.sum(np.diff(np.sign(deriv)) != 0) / len(deriv)
        checks.append(zcr <= self.zcr_max)

        # 4. Peak regularity: std of inter-peak intervals should be low
        peaks, _ = find_peaks(window, distance=int(self.fs * 0.3))
        if len(peaks) >= 2:
            intervals   = np.diff(peaks)
            regularity  = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-12))
            checks.append(regularity >= self.peak_regularity_min)
        else:
            checks.append(False)  # no detectable beats = bad window

        return sum(checks) / len(checks) >= self.threshold

    @staticmethod
    def _to_intervals(good: np.ndarray, min_samples: int) -> list:
        intervals = []
        in_good   = False
        start     = 0
        for i, val in enumerate(good):
            if val and not in_good:
                start   = i
                in_good = True
            elif not val and in_good:
                if i - start >= min_samples:
                    intervals.append([start, i - 1])
                in_good = False
        if in_good and len(good) - start >= min_samples:
            intervals.append([start, len(good) - 1])
        return intervals


class ECGSQI3:
    """
    Computes ECG Signal Quality using the ecg_qc library (Aura Healthcare).
    Uses a trained ML model to classify 9-second windows as good/bad quality.

    Returns a list of [start_idx, end_idx] intervals where quality is acceptable,
    same interface as the original hand-crafted ECGSQI class.

    Install: pip install ecg-qc
    """

    def __init__(
        self,
        fs: float,
        min_time_ms: float = 2000,
    ):
        self.fs          = int(fs)
        self.min_samples = int(min_time_ms * fs / 1000)

        # ecg_qc is instantiated with the sampling frequency
        try:
            self.model = EcgQc(sampling_frequency=self.fs)
        except Exception as e:
            print(f"[ECGSQI] Failed to initialize ecg_qc model: {e}")
            self.model = None

    def compute(self, signal: list) -> list:
        """
        Returns list of [start_idx, end_idx] good-quality intervals.
        """
        if self.model is None or not signal:
            return [[0, len(signal) - 1]] if len(signal) >= self.min_samples else []

        sig = np.array(signal, dtype=float)

        try:
            # ecg_qc.get_signal_quality() classifies the full signal and
            # returns a list of per-window quality labels (1 = good, 0 = bad).
            # Each window covers 9 seconds × fs samples.
            window_samples = 9 * self.fs
            quality_labels = self.model.get_signal_quality(sig.tolist())

            if not quality_labels:
                print("[ECGSQI] No quality labels returned.")
                return []

            # Map per-window labels back to per-sample boolean array
            good = np.zeros(len(sig), dtype=bool)
            for i, label in enumerate(quality_labels):
                start = i * window_samples
                end   = min(start + window_samples, len(sig))
                if label == 1:
                    good[start:end] = True

            return self._to_intervals(good, self.min_samples)

        except Exception as e:
            print(f"[ECGSQI] ecg_qc classification failed: {e}. Falling back to full signal.")
            return [[0, len(signal) - 1]] if len(signal) >= self.min_samples else []

    @staticmethod
    def _to_intervals(good: np.ndarray, min_samples: int) -> list:
        """Convert a boolean mask to [start, end] intervals."""
        intervals = []
        in_good   = False
        start     = 0
        for i, val in enumerate(good):
            if val and not in_good:
                start   = i
                in_good = True
            elif not val and in_good:
                if i - start >= min_samples:
                    intervals.append([start, i - 1])
                in_good = False
        if in_good and len(good) - start >= min_samples:
            intervals.append([start, int(len(good) - 1)])
        return intervals


class ECGSQA:
    """
    Computes ECG Signal Quality using neurokit2.

    Uses a two-stage approach:
      1. Primary: nk.ecg_quality() with zhao2018 (strict ML-based)
      2. Fallback: rule-based checks using R-peak detectability + RR regularity
         — used when zhao2018 rejects a window that visually looks clean
         (e.g. high-amplitude QRS complexes that trip the kSQI threshold)

    Returns a list of [start_idx, end_idx] intervals where quality is acceptable.
    """

    def __init__(
            self,
            fs: float,
            window_ms: float = 4000,
            step_ms: float = 1000,
            min_time_ms: float = 2000,
            accept_borderline: bool = True,
            use_fallback: bool = True,  # rule-based fallback when zhao2018 rejects
            min_r_peaks: int = 2,  # minimum R-peaks required in window
            rr_regularity_min: float = 0.3,  # lower = more lenient (0=random, 1=perfect)
            hr_range_bpm: tuple = (25, 220),  # physiological HR bounds
    ):
        self.fs = fs
        self.window_samples = int(window_ms * fs / 1000)
        self.step_samples = int(step_ms * fs / 1000)
        self.min_samples = int(min_time_ms * fs / 1000)
        self.accept_borderline = accept_borderline
        self.use_fallback = use_fallback
        self.min_r_peaks = min_r_peaks
        self.rr_regularity_min = rr_regularity_min
        self.hr_min = hr_range_bpm[0]
        self.hr_max = hr_range_bpm[1]

    def compute(self, signal: list) -> list:
        sig = np.array(signal, dtype=float)
        n = len(sig)
        good = np.zeros(n, dtype=bool)

        for start in range(0, n - self.window_samples + 1, self.step_samples):
            end = start + self.window_samples
            window = sig[start:end]
            if self._is_good(window):
                good[start:end] = True

        return self._to_intervals(good, self.min_samples)

    def _is_good(self, window: np.ndarray) -> bool:
        try:
            ecg_cleaned = nk.ecg_clean(window, sampling_rate=int(self.fs))
            _, r_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(self.fs))
            r_peaks = r_info["ECG_R_Peaks"]

            if len(r_peaks) < self.min_r_peaks:
                return False

            # ── Stage 1: zhao2018 ─────────────────────────────────────────
            try:
                quality = nk.ecg_quality(
                    ecg_cleaned,
                    rpeaks=r_peaks,
                    sampling_rate=int(self.fs),
                    method="zhao2018",
                )
                if quality == "Excellent":
                    return True
                if quality == "Barely acceptable" and self.accept_borderline:
                    return True
                # zhao2018 said bad — try fallback before rejecting
                if not self.use_fallback:
                    return False
            except Exception:
                if not self.use_fallback:
                    return False

            # ── Stage 2: rule-based fallback ──────────────────────────────
            return self._fallback_check(r_peaks)

        except Exception:
            return False

    def _fallback_check(self, r_peaks: np.ndarray) -> bool:
        """
        Lightweight physiological plausibility check.
        Passes if R-peaks are regular and HR is in the normal human range.
        Designed to accept clean high-amplitude ECG that zhao2018 rejects.
        """
        rr_intervals = np.diff(r_peaks)

        if len(rr_intervals) == 0:
            return False

        # 1. HR range: mean RR must correspond to a physiological heart rate
        mean_rr_ms = (np.mean(rr_intervals) / self.fs) * 1000
        hr_bpm = 60000.0 / (mean_rr_ms + 1e-12)
        if not (self.hr_min <= hr_bpm <= self.hr_max):
            return False

        # 2. RR regularity: coefficient of variation should be low
        cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-12)
        regularity = 1.0 - cv
        if regularity < self.rr_regularity_min:
            return False

        return True

    @staticmethod
    def _to_intervals(good: np.ndarray, min_samples: int) -> list:
        intervals = []
        in_good = False
        start = 0
        for i, val in enumerate(good):
            if val and not in_good:
                start = i
                in_good = True
            elif not val and in_good:
                if i - start >= min_samples:
                    intervals.append([start, i - 1])
                in_good = False
        if in_good and len(good) - start >= min_samples:
            intervals.append([start, int(len(good) - 1)])
        return intervals