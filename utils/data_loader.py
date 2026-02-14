import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from schemas.bp_schema import BPSample
from config.mappings import DATASET_CONFIGS
from utils.preprocessing import Preprocessor


class DataLoader:
    @staticmethod
    def _align_uci_signals(ppg, ecg, fs):
        """
        Forces alignment based on physiological PTT constraints (100ms - 500ms).
        """
        try:
            # 1. Standardize inputs
            ppg = np.array(ppg)
            ecg = np.array(ecg)

            # 2. Extract ECG R-Peaks (The Trigger)
            ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
            _, r_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
            r_spikes = np.zeros(len(ecg))
            r_spikes[r_info["ECG_R_Peaks"]] = 1

            # 3. Extract PPG Onsets ('a' waves are the Arrival)
            # We use a sharp derivative to find the pulse start
            vpg = np.gradient(Preprocessor.clean_signal(ppg, fs))
            ppg_onsets, _ = signal.find_peaks(vpg, distance=int(fs * 0.5), prominence=np.std(vpg) * 0.5)
            p_spikes = np.zeros(len(ppg))
            p_spikes[ppg_onsets] = 1

            # 4. Cross-Correlate the Spikes
            # This finds the best 'global' shift to align all heartbeats
            corr = signal.correlate(p_spikes, r_spikes, mode='same')
            lags = signal.correlation_lags(len(p_spikes), len(r_spikes), mode='same')

            # 5. Define the "Physiological Search Window"
            # We want to find the shift that results in an R -> a delay of ~200ms
            # In UCI, the hardware lag can be +/- 1 second.
            # We search for the best correlation within +/- 1.5 seconds
            search_limit = int(1.5 * fs)
            mask = (lags > -search_limit) & (lags < search_limit)

            # Find the lag that maximizes peak alignment
            best_lag = lags[mask][np.argmax(corr[mask])]

            # 6. Apply the shift
            # This aligns the signals so index 'i' in both signals refers to the same moment
            aligned_ppg = np.roll(ppg, -best_lag)

            print(f"--- UCI SYNC FIX ---")
            print(f"Detected Hardware/Buffer Offset: {best_lag} samples ({(best_lag / fs) * 1000:.1f}ms)")
            print(f"Action: Shifting PPG by {-best_lag} samples to align with ECG.")

            return aligned_ppg.tolist()

        except Exception as e:
            print(f"Alignment Error: {e}")
            return ppg.tolist()

    @staticmethod
    def load_from_csv(file_path: str, dataset_type: str, max_duration_sec: float = None) -> list:
        config = DATASET_CONFIGS[dataset_type]
        raw_df = pd.read_csv(file_path)

        # Get standardized column names from config
        # We find which raw column maps to our internal 'timestamp', 'ppg', etc.
        inv_map = {v: k for k, v in config["columns"].items()}
        raw_ts_col = inv_map['timestamp']

        if max_duration_sec is not None:
            # We determine the end time based on the first timestamp
            start_time = raw_df[raw_ts_col].min()
            # Convert max_duration back to raw units if necessary
            # (e.g., if UCI is in seconds, 10s = 10 units. If MCS is in ms, 10s = 10,000 units)
            duration_in_raw_units = max_duration_sec / config["unit_conversion"]["timestamp"]
            end_time = start_time + duration_in_raw_units

            raw_df = raw_df[raw_df[raw_ts_col] <= end_time].copy()
            print(f"Testing Mode: Truncated {dataset_type} to {max_duration_sec}s ({len(raw_df)} rows)")

        raw_pid_col = inv_map['patient_id']
        raw_ppg_col = inv_map['ppg']
        raw_ecg_col = inv_map['ecg']
        raw_bps_col = inv_map['bps']
        raw_bpd_col = inv_map['bpd']

        # 1. Standardize Timestamps using the config conversion
        raw_df['standard_ts'] = (raw_df[raw_ts_col] - raw_df[raw_ts_col].min()) * config["unit_conversion"]["timestamp"]

        samples = []
        for pid, group in raw_df.groupby(raw_pid_col):

            if dataset_type == "mcs":
                # --- MCS LOGIC: Asynchronous Interleaved ---
                # Separate streams using flags
                ecg_stream = group[group['is_ecg'] == 1][['standard_ts', raw_ecg_col]].sort_values('standard_ts')
                ppg_stream = group[group['is_ppg'] == 1][['standard_ts', raw_ppg_col]].sort_values('standard_ts')

                # Align PPG to ECG clock
                synced_df = pd.merge_asof(
                    ecg_stream, ppg_stream,
                    left_on='standard_ts', right_on='standard_ts',
                    direction='nearest', tolerance=50
                ).dropna()

                final_ts = synced_df['standard_ts'].tolist()
                final_ppg = synced_df[raw_ppg_col].tolist()
                final_ecg = synced_df[raw_ecg_col].tolist()

            else:
                # --- UCI LOGIC: Synchronous Parallel ---
                group = group.sort_values('standard_ts')
                final_ts = group['standard_ts'].tolist()
                final_ecg = group[raw_ecg_col].tolist()

                # APPLY FIX: Align the PPG to the ECG before creating the sample
                raw_ppg = group[raw_ppg_col].values
                final_ppg = PulseDBStyleAligner.align(
                    raw_ppg,
                    np.array(final_ecg),
                    config["target_fs"]
                )

            # 2. Create Pydantic Sample (Invert PPG if it's the green channel)
            # Typically MCS (Green) needs inversion, UCI (IR/Red) does not.
            # You can add an 'invert' flag to your config/mappings.py for this.

            processed_ppg = np.negative(final_ppg).tolist() if dataset_type == "mcs" else final_ppg

            sample = BPSample(
                patient_id=str(pid),
                timestamp=final_ts,
                ppg=processed_ppg,
                ecg=final_ecg,
                bps=float(group[raw_bps_col].dropna().iloc[0]),
                bpd=float(group[raw_bpd_col].dropna().iloc[0]),
                fs=config["target_fs"]
            )
            samples.append(sample)

        return samples


class PulseDBStyleAligner:
    @staticmethod
    def align(ppg, ecg, fs):
        try:
            # 1. Detect ECG R-peaks
            ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
            _, r_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
            r_indices = r_peaks["ECG_R_Peaks"]

            # 2. Detect PPG Systolic Peaks
            ppg_cleaned = Preprocessor.clean_signal(ppg, fs)
            # Use neurokit's ppg_findpeaks for consistency with PulseDB style
            ppg_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=fs)
            p_indices = ppg_info["PPG_Peaks"]

            # 3. Create 'Rhythm Series' (Instantaneous Heart Rate)
            # We map the timing of each beat to a signal we can correlate
            def get_rhythm_signal(indices, length):
                sig = np.zeros(length)
                # We calculate the interval (diff) and 'smear' it
                # to create a continuous rhythm pattern
                intervals = np.diff(indices)
                for i in range(len(intervals)):
                    sig[indices[i]:indices[i + 1]] = intervals[i]
                return sig

            ecg_rhythm = get_rhythm_signal(r_indices, len(ecg))
            ppg_rhythm = get_rhythm_signal(p_indices, len(ppg))

            # 4. Cross-Correlate the Rhythms (This finds the 'Beat Lag')
            # This is robust against wave-shape differences
            correlation = signal.correlate(ppg_rhythm - np.mean(ppg_rhythm),
                                           ecg_rhythm - np.mean(ecg_rhythm),
                                           mode='same')
            lags = signal.correlation_lags(len(ppg_rhythm), len(ecg_rhythm), mode='same')

            # PulseDB typically searches within a few seconds window
            search_window = (lags > -int(2 * fs)) & (lags < int(2 * fs))
            beat_lag = lags[search_window][np.argmax(correlation[search_window])]

            # 5. Fine-tuning for PTT (The Physiological Offset)
            # After rhythm alignment, we nudge the signal so R is ~200ms before a-wave
            # We'll use the first derivative onset for precision
            aligned_ppg = np.roll(ppg, -beat_lag)

            # Calculate mean observed delay after rhythm match
            # If it's still 'wrong' (R after a), we nudge by a constant physiological offset
            test_vpg = np.gradient(Preprocessor.clean_signal(aligned_ppg, fs))
            # Find the first valid onset after the first R-peak
            first_r = r_indices[2]  # skip first two for stability
            search_area = test_vpg[first_r:first_r + int(fs * 0.6)]
            if len(search_area) > 0:
                onset_offset = np.argmax(search_area)
                # We want onset_offset to be around 0.2 * fs (200ms)
                # If it's not, we apply a final fine_shift
                fine_shift = onset_offset - int(0.2 * fs)
                aligned_ppg = np.roll(aligned_ppg, fine_shift)
                total_lag = beat_lag - fine_shift
            else:
                total_lag = beat_lag

            print(
                f"PulseDB Aligner: Total hardware lag corrected: {total_lag} samples ({(total_lag / fs) * 1000:.1f}ms)")
            return aligned_ppg.tolist()

        except Exception as e:
            print(f"PulseDB Alignment failed: {e}")
            return ppg