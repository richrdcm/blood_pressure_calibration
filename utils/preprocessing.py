import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


class Preprocessor:
    @staticmethod
    def butter_bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4):
        """Standard Butterworth Bandpass Filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    @staticmethod
    def clean_signal(ppg_data: list, fs: float) -> np.ndarray:
        """
        Applies filtering and normalization.
        Standard range for BP estimation is 0.5Hz to 8.0Hz.
        """
        arr = np.array(ppg_data)

        # 1. Bandpass filter to remove baseline wander and high-freq noise
        filtered = Preprocessor.butter_bandpass(arr, 0.5, 8.0, fs)

        # 2. Normalize to [0, 1] for visualization and AI consistency
        # This removes the DC offset from the 160k MCS magnitude
        norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered) + 1e-8)

        return norm

    @staticmethod
    def process_ecg(ecg_data: list, fs: float) -> np.ndarray:
        """Filter ECG to highlight R-peaks."""
        arr = np.array(ecg_data)
        # 5-15Hz is standard for R-peak detection (Pan-Tompkins style)
        filtered = Preprocessor.butter_bandpass(arr, 5.0, 15.0, fs)
        return filtered

    @staticmethod
    def resample_signal(data: list, original_fs: int, target_fs: int) -> np.ndarray:
        """Upsamples or downsamples data to match a target frequency."""
        if original_fs == target_fs:
            return np.array(data)

        x_old = np.linspace(0, len(data), num=len(data))
        # Calculate how many samples we need for the target frequency
        num_samples_new = int(len(data) * target_fs / original_fs)
        x_new = np.linspace(0, len(data), num=num_samples_new)

        interpolation_func = interp1d(x_old, data, kind='cubic')
        return interpolation_func(x_new)