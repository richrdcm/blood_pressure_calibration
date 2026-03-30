import numpy as np
from neurokit2 import ecg_clean
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import resample
from schemas.bp_schema import BPSample
import pandas as pd
import heartpy as hp
import neurokit2 as nk
from src.extractor.sqi import ECGSQA, PPGSQA


class Preprocessor:
    def clean_signals(self, samples: list) -> list:
        """
        Resamples, filters, and normalizes the PPG signal.
        """
        cleaned = []
        for sample in samples:
            # 1. Resample both signals to target_fs
            ppg_time, clean_ppg = self.clean_ppg_signal(sample)
            ecg_time, clean_ecg = self.clean_ecg_signal(sample)

            # 2. Sync PPG to ECG clock FIRST
            df_ecg = pd.DataFrame({'time': ecg_time, 'ecg': clean_ecg})
            df_ppg = pd.DataFrame({'time': ppg_time, 'ppg': clean_ppg})

            synced_df = pd.merge_asof(
                df_ecg,
                df_ppg,
                on='time',
                direction='nearest',
                tolerance=50
            ).dropna()

            final_ts = synced_df['time'].to_numpy()
            final_ecg = synced_df['ecg'].to_numpy()
            final_ppg = synced_df['ppg'].to_numpy()

            # 3. SQI on the synced signals — indices now match final_ts/final_ppg/final_ecg
            ppg_sqi = PPGSQA(fs=sample.target_fs, min_time_ms=2000).compute(final_ppg) if len(final_ppg) > 0 else []
            ecg_sqi = ECGSQA(fs=sample.target_fs, min_time_ms=2000).compute(final_ecg) if len(final_ecg) > 0 else []
            joint_sqi = self.intersect_sqi_intervals(ppg_sqi, ecg_sqi, min_time_ms=2000, fs=sample.target_fs)

            print(f"[SQI] Patient {sample.patient_id}: "
                  f"PPG={len(ppg_sqi)} intervals, "
                  f"ECG={len(ecg_sqi)} intervals, "
                  f"Joint={len(joint_sqi)} intervals")

            # 4. Build cleaned sample
            clean_sample = BPSample(
                patient_id=str(sample.patient_id),
                ppg_timestamps=final_ts.tolist(),
                ecg_timestamps=final_ts.tolist(),
                ppg=final_ppg.tolist(),
                ecg=final_ecg.tolist(),
                bps=sample.bps,
                bpd=sample.bpd,
                target_fs=sample.target_fs,
                ppg_fs=sample.target_fs,
                ecg_fs=sample.target_fs,
                ppg_sqi=ppg_sqi,
                ecg_sqi=ecg_sqi,
                joint_sqi=joint_sqi,
            )
            cleaned.append(clean_sample)

        return cleaned

    def clean_ecg_signal(self, sample: BPSample) -> tuple:
        # 1. Resample ECG
        if sample.ecg:
            time, resampled = self.resample_signal(data=sample.ecg,
                                                   timestamps=sample.ecg_timestamps,
                                                   target_fs=sample.target_fs)

            ecg_cleaned = nk.ecg_clean(resampled, sampling_rate=sample.target_fs, method="neurokit")
        else:
            time = np.asarray([])
            ecg_cleaned = np.asarray([])
        return time, ecg_cleaned

    def clean_ppg_signal(self, sample) -> tuple:
        """
        Resamples, filters, and normalizes the PPG signal using NeuroKit2's optimized pipeline.
        """
        time, resampled = self.resample_signal(data=sample.ppg,
                                               timestamps=sample.ppg_timestamps,
                                               target_fs=sample.target_fs)
        # 2. Clean using NeuroKit2 (This safely handles the 0.5-8.0 Hz bandpass internally)
        cleaned_ppg = nk.ppg_clean(
            nk.as_vector(resampled),
            sampling_rate=sample.target_fs,
            method="elgendi"
        )

        # 3. Normalize to [0, 1] safely
        norm = nk.rescale(cleaned_ppg, to=[0, 1])

        return time, norm

    def clean_ppg_signal_cheby(self, sample) -> tuple:
        """
        Alternative PPG cleaning method using Chebyshev Type II filter
        and Moving Average smoothing. Interchangeable with clean_ppg_signal.
        """
        time, resampled = self.resample_signal(data=sample.ppg,
                                               timestamps=sample.ppg_timestamps,
                                               target_fs=sample.target_fs)

        fs = sample.ppg_fs

        # 1. Determine the Nyquist Limit
        nyquist = fs / 2.0

        # 2. Chebyshev Type II Bandpass Filter
        fL = 0.5
        # THE FIX: Cap fH to either 12.0 Hz OR just below the Nyquist limit, whichever is smaller.
        # We subtract 0.1 to leave a tiny mathematical safety margin.
        fH = min(12.0, nyquist - 0.1)

        order = 4
        b, a = signal.cheby2(order, rs=20, Wn=[fL, fH], btype='bandpass', fs=fs)
        ppg_cb2 = signal.filtfilt(b, a, resampled)

        # 3. Moving Average Smoothing (50ms window for the main PPG)
        if fs >= 75:
            win = round(fs * 50 / 1000)  # 50ms = 0.05 seconds
            B = np.ones(win) / win
            cleaned_ppg = signal.filtfilt(B, 1, ppg_cb2)
        else:
            cleaned_ppg = ppg_cb2

        # 4. Normalize to [0, 1] safely
        import neurokit2 as nk
        norm = nk.rescale(cleaned_ppg, to=[0, 1])

        return time, norm.tolist()

    @staticmethod
    def resample_signal(data: list, timestamps: list, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Resamples data to a target frequency based on the actual time bounds.
        """
        data_arr = np.array(data)
        time_arr = np.array(timestamps)

        # 1. Calculate actual duration in raw units
        start_time = time_arr[0]
        end_time = time_arr[-1]
        duration_raw = end_time - start_time

        # THE FIX: Convert duration to seconds if timestamps are in milliseconds.
        # If your 10-second clip shows up as ~10,000, we divide by 1000.
        duration_sec = duration_raw / 1000.0 if duration_raw > 1000 else duration_raw

        # 2. Compute exact number of points needed for the target_fs
        # Now 10 seconds * 125 Hz = 1250 samples (Correct!)
        num_samples_new = int(np.round(duration_sec * target_fs))

        # 3. Create the perfect target time grid
        new_timestamps = np.linspace(start_time, end_time, num_samples_new)

        # 4. Interpolate the data onto the new grid
        new_data = nk.signal_interpolate(
            x_values=time_arr,
            y_values=data_arr,
            x_new=new_timestamps,
            method="monotone_cubic"
        )

        return new_timestamps, new_data

    @staticmethod
    def intersect_sqi_intervals(
            ppg_sqi: list,
            ecg_sqi: list,
            min_time_ms: float = 2000,
            fs: float = 125,
    ) -> list:
        """
        Returns intervals where BOTH PPG and ECG quality are good.

        Parameters
        ----------
        ppg_sqi      : list of [start, end] index pairs from PPGSQI
        ecg_sqi      : list of [start, end] index pairs from ECGSQI
        min_time_ms  : minimum duration of a valid intersection in ms
        fs           : sampling frequency — used to convert min_time_ms to samples

        Returns
        -------
        List of [start, end] index pairs where both signals are usable.
        """
        min_samples = int(min_time_ms * fs / 1000)
        intersections = []

        for p_start, p_end in ppg_sqi:
            for e_start, e_end in ecg_sqi:
                # Overlap = max of starts, min of ends
                i_start = max(p_start, e_start)
                i_end = min(p_end, e_end)

                if i_end > i_start and (i_end - i_start) >= min_samples:
                    intersections.append([i_start, i_end])

        # Merge adjacent or overlapping intersections
        if not intersections:
            return []

        intersections.sort(key=lambda x: x[0])
        merged = [intersections[0]]
        for current in intersections[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 1:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)

        return merged


def remap_fiducial_indices(colleague_csv_path, raw_timestamps, clean_timestamps, old_ppg_wave):
    """
    Remaps indices from a low-frequency (25Hz) source to a high-frequency (135Hz)
    target using real-world timestamps as the synchronization bridge.

    :param df_colleague: DataFrame containing the 25Hz indices
    :param raw_timestamps: The original 25Hz timestamp array
    :param clean_timestamps: The resampled 135Hz timestamp array
    :return: A copy of the DataFrame with updated indices
    """
    import os
    if not os.path.exists(colleague_csv_path):
        print(f"Error: Colleague CSV not found at {colleague_csv_path}")
        return

    df_coll = pd.read_csv(colleague_csv_path)

    results = []
    # All potential fiducial points from colleague
    colleague_cols = ['sp', 'dn', 'dp', 'a', 'b', 'c', 'd', 'e', 'f']

    # Pre-calculate bounds for speed
    max_raw_idx = len(raw_timestamps) - 1

    for _, row in df_coll.iterrows():
        # Temporary storage for this specific heart beat
        beat_data = {}

        for col in colleague_cols:
            if col not in row or pd.isna(row[col]):
                beat_data[col if col != 'sp' else 'ppg_peak'] = None
                continue

            # 1. Get the 25Hz Index
            raw_idx = int(row[col])

            # 2. Synchronize via Timestamps
            if 0 <= raw_idx <= max_raw_idx:
                true_time = raw_timestamps[raw_idx]

                # Find the closest index in the 135Hz clean_timestamps
                # Using searchsorted for high-performance mapping
                clean_idx = np.searchsorted(clean_timestamps, true_time)

                # Refine to the absolute nearest neighbor
                if clean_idx > 0 and (clean_idx == len(clean_timestamps) or
                                      abs(true_time - clean_timestamps[clean_idx - 1]) < abs(
                            true_time - clean_timestamps[clean_idx])):
                    final_idx = clean_idx - 1
                else:
                    final_idx = clean_idx

                # Rename 'sp' (Systolic Peak) to 'ppg_peak' to match your dictionary key
                key = 'ppg_peak' if col == 'sp' else col
                beat_data[key] = int(final_idx)
            else:
                beat_data[col if col != 'sp' else 'ppg_peak'] = None

        # Add to results list if at least 'a' or 'ppg_peak' exists
        if beat_data.get('ppg_peak') is not None or beat_data.get('a') is not None:
            results.append(beat_data)

    new_ppg_wave = (old_ppg_wave[0], old_ppg_wave[1], results)

    return new_ppg_wave

