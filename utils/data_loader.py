import numpy as np
import pandas as pd
from schemas.bp_schema import BPSample
from config.mappings import DATASET_CONFIGS


class DataLoader:
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
                # Data is already aligned in rows
                group = group.sort_values('standard_ts')
                final_ts = group['standard_ts'].tolist()
                final_ppg = group[raw_ppg_col].tolist()
                final_ecg = group[raw_ecg_col].tolist()

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