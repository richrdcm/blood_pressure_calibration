import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from schemas.bp_schema import BPSample
from config.mappings import DATASET_CONFIGS
from utils.preprocessing import Preprocessor
import os


class DataLoader:

    @staticmethod
    def _append_sample(
            group: pd.DataFrame,
            pid: str,
            bps: float,
            bpd: float,
            samples: list,
            dataset_type: str,
            raw_ppg_col: str,
            raw_ecg_col: str,
            config: dict,
    ):
        """Extracts PPG/ECG from a group slice and appends a BPSample."""
        group = group.copy()

        # Re-zero timestamps so each extracted interval perfectly starts at 0
        group['standard_ts'] = group['standard_ts'] - group['standard_ts'].min()

        if dataset_type == "mcs":
            if raw_ecg_col in group.columns:
                ecg_mask = (group[raw_ecg_col] != 0) & (group[raw_ecg_col].notna())
                ecg_stream = group[ecg_mask].sort_values('standard_ts')
                ecg_timestamps = ecg_stream['standard_ts'].tolist()
                final_ecg = ecg_stream[raw_ecg_col].tolist()
            else:
                ecg_timestamps = []
                final_ecg = []

            ppg_mask = (group[raw_ppg_col] != 0) & (group[raw_ppg_col].notna())
            ppg_stream = group[ppg_mask].sort_values('standard_ts')
            ppg_timestamps = ppg_stream['standard_ts'].tolist()
            final_ppg = ppg_stream[raw_ppg_col].tolist()

        else:
            group = group.sort_values('standard_ts')
            ppg_timestamps = group['standard_ts'].tolist()
            ecg_timestamps = group['standard_ts'].tolist()
            final_ecg = group[raw_ecg_col].tolist()
            final_ppg = group[raw_ppg_col].tolist()

        if dataset_type == "mcs" and len(final_ppg) > 0:
            final_ppg = (-np.array(final_ppg) + np.max(final_ppg)).tolist()

        samples.append(BPSample(
            patient_id=str(pid),
            ppg_timestamps=ppg_timestamps,
            ecg_timestamps=ecg_timestamps if ecg_timestamps else None,
            ppg=final_ppg,
            ecg=final_ecg if final_ecg else None,
            bps=bps,
            bpd=bpd,
            target_fs=config["target_fs"],
            ppg_fs=config["channels"]["ppg"]["fs"],
            ecg_fs=config["channels"]["ecg"]["fs"],
        ))

    @staticmethod
    def _parse_bp_reference(bp_ref_path: str) -> pd.DataFrame:
        """Parses a BP reference CSV."""
        try:
            try:
                df = pd.read_csv(bp_ref_path, sep=';')
            except Exception:
                df = pd.read_csv(bp_ref_path, sep=',')

            df.columns = [c.strip() for c in df.columns]
            cols = list(df.columns)

            if len(cols) < 4:
                print(f"[BP Ref] ERROR: expected 4 columns, got {len(cols)}")
                return pd.DataFrame()

            date_col, time_col, bps_col, bpd_col = cols[0], cols[1], cols[2], cols[3]

            df['datetime'] = pd.to_datetime(
                df[date_col].astype(str).str.strip() + ' ' +
                df[time_col].astype(str).str.strip(),
                dayfirst=True,
                errors='coerce',
            )
            df['bps'] = pd.to_numeric(df[bps_col], errors='coerce')
            df['bpd'] = pd.to_numeric(df[bpd_col], errors='coerce')

            df = df[['datetime', 'bps', 'bpd']].dropna().sort_values('datetime').reset_index(drop=True)
            print(f"[BP Ref] Loaded {len(df)} BP reference readings from '{bp_ref_path}'")
            return df

        except Exception as e:
            print(f"[BP Ref] Failed to parse BP reference file: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_from_csv(
            file_path: str,
            dataset_type: str,
            index_from: int = None,
            index_to: int = None,
            each_file_is_own_patient: bool = True,
            bp_ref_path: str = None,
            min_duration_msec: float = 60000,
            max_duration_msec: float = None,
    ) -> list:

        # ── Build file list ───────────────────────────────────────────────────
        if index_from is not None and index_to is not None:
            paths = [
                f"{file_path}_{i}.csv"
                for i in range(index_from, index_to + 1)
                if os.path.isfile(f"{file_path}_{i}.csv")
            ]
            if not paths:
                print("[load_from_csv] No files found in the given range.")
                return []
        else:
            paths = [file_path]

        # ── Load BP reference (MCS only) ──────────────────────────────────────
        bp_ref_df = DataLoader._parse_bp_reference(bp_ref_path) if bp_ref_path else pd.DataFrame()

        config = DATASET_CONFIGS[dataset_type]
        inv_map = {v: k for k, v in config["columns"].items()}
        raw_ts_col = inv_map['timestamp']
        raw_pid_col = inv_map['patient_id']
        unit_conv = config["unit_conversion"]["timestamp"]
        prefix_stem = os.path.splitext(os.path.basename(file_path))[0]

        # ── Load and concatenate all files ────────────────────────────────────
        frames = []
        for p in paths:
            df = pd.read_csv(p)
            file_stem = os.path.splitext(os.path.basename(p))[0]

            df = df.drop_duplicates(subset=[raw_ts_col], keep='first')

            if raw_pid_col not in df.columns:
                df[raw_pid_col] = file_stem if each_file_is_own_patient else prefix_stem
            elif not each_file_is_own_patient:
                df[raw_pid_col] = prefix_stem

            # Absolute datetime for BP alignment
            raw_ts_sec = df[raw_ts_col] / 1000.0 if dataset_type == "mcs" else df[raw_ts_col]
            df['abs_datetime'] = pd.to_datetime(raw_ts_sec, unit='s')
            df['_source_file'] = file_stem  # track which file each row came from

            # Time cap (only when no BP reference involved)
            if bp_ref_df.empty and max_duration_msec is not None:
                start_time = df[raw_ts_col].min()
                duration_in_raw_units = max_duration_msec * unit_conv
                df = df[df[raw_ts_col] <= (start_time + duration_in_raw_units)].copy()
                print(f"[load_from_csv] {file_stem}: truncated to {max_duration_msec}ms")

            # Standardize relative timestamps
            last_ts = frames[-1]['standard_ts'].max() if (not each_file_is_own_patient and frames) else 0.0
            df['standard_ts'] = ((df[raw_ts_col] - df[raw_ts_col].min()) * unit_conv) + last_ts
            df = df.drop_duplicates(subset=['standard_ts'], keep='first')

            frames.append(df)

        raw_df = pd.concat(frames, ignore_index=True).sort_values('abs_datetime')

        raw_ppg_col = inv_map['ppg']
        raw_ecg_col = inv_map['ecg']
        raw_bps_col = inv_map['bps']
        raw_bpd_col = inv_map['bpd']

        samples = []

        for pid, group in raw_df.groupby(raw_pid_col):
            group_sorted = group.sort_values('abs_datetime')

            # ── UCI: BP embedded in columns, split on BP change points ──────
            if dataset_type == "uci":
                # For UCI, pid comes directly from the patient_id column in the CSV
                # so it already correctly identifies the patient across multiple files.
                # We prefix with the source file stem only when the pid is ambiguous
                # (i.e. when the CSV had no patient_id column and we used the filename).
                source_files = group_sorted['_source_file'].unique()
                if len(source_files) > 1:
                    # Patient spans multiple files — use pid as-is (already unique)
                    effective_pid = str(pid)
                else:
                    # Single file — prefix with file stem to avoid collisions
                    # e.g. pid=1 from Part_1.csv → "Part_1_patient_1"
                    effective_pid = f"{source_files[0]}_{pid}"

                samples += DataLoader._split_uci_by_bp(
                    group=group_sorted,
                    pid=effective_pid,
                    raw_ppg_col=raw_ppg_col,
                    raw_ecg_col=raw_ecg_col,
                    raw_bps_col=raw_bps_col,
                    raw_bpd_col=raw_bpd_col,
                    config=config,
                    max_duration_msec=max_duration_msec,
                    min_duration_msec=min_duration_msec,
                )

            # ── MCS: BP from external reference file ──────────────────────
            elif not bp_ref_df.empty:
                window_ms = max_duration_msec if max_duration_msec else 60000
                window_td = pd.Timedelta(milliseconds=window_ms)

                bp_in_window = bp_ref_df[
                    bp_ref_df['datetime'] >= group_sorted['abs_datetime'].min()
                    ].reset_index(drop=True)

                boundaries = [group_sorted['abs_datetime'].min()] + bp_in_window['datetime'].tolist()

                for i, bp_row in bp_in_window.iterrows():
                    bp_dt = bp_row['datetime']
                    bps = float(bp_row['bps'])
                    bpd = float(bp_row['bpd'])

                    broad_start = boundaries[i]
                    broad_end = bp_dt
                    mask = (group_sorted['abs_datetime'] >= broad_start) & \
                           (group_sorted['abs_datetime'] <= broad_end)
                    broad_group = group_sorted[mask]

                    if len(broad_group) < 10:
                        print(f"[BP Ref] Patient {pid} interval {i}: insufficient data — skipping.")
                        continue

                    actual_last_dt = broad_group['abs_datetime'].max()
                    target_start_dt = actual_last_dt - window_td
                    final_group = broad_group[broad_group['abs_datetime'] >= target_start_dt]

                    if len(final_group) < 10:
                        print(f"[BP Ref] Patient {pid}_bp{i + 1}: insufficient data in window — skipping.")
                        continue

                    time_gap = (bp_dt - actual_last_dt).total_seconds()
                    print(f"[load_from_csv] Patient {pid}_bp{i + 1}: SBP={bps}, DBP={bpd}, "
                          f"gap={time_gap:.1f}s, rows={len(final_group)}")

                    DataLoader._append_sample(
                        group=final_group, pid=f"{pid}_bp{i + 1}",
                        bps=bps, bpd=bpd, samples=samples,
                        dataset_type=dataset_type,
                        raw_ppg_col=raw_ppg_col, raw_ecg_col=raw_ecg_col,
                        config=config,
                    )

            # ── MCS: No BP reference — single sample ──────────────────────
            else:
                bps_series = group[raw_bps_col].dropna() if raw_bps_col in group.columns else pd.Series(dtype=float)
                bpd_series = group[raw_bpd_col].dropna() if raw_bpd_col in group.columns else pd.Series(dtype=float)
                bps = float(bps_series.iloc[0]) if not bps_series.empty else None
                bpd = float(bpd_series.iloc[0]) if not bpd_series.empty else None
                DataLoader._append_sample(
                    group_sorted, str(pid), bps, bpd, samples,
                    dataset_type, raw_ppg_col, raw_ecg_col, config,
                )

        print(f"[load_from_csv] Successfully generated {len(samples)} sample(s).")
        return samples

    @staticmethod
    def _split_uci_by_bp(
            group: pd.DataFrame,
            pid: str,
            raw_ppg_col: str,
            raw_ecg_col: str,
            raw_bps_col: str,
            raw_bpd_col: str,
            config: dict,
            max_duration_msec: float = None,
            min_duration_msec: float = 60000,  # ← minimum 60 seconds per segment
    ) -> list:
        """
        Splits a UCI patient group into one BPSample per unique BP measurement.
        Only segments with at least min_duration_msec of signal are kept.
        """
        samples = []
        fs = config["target_fs"]
        min_samples = int(min_duration_msec * fs / 1000)

        # ── Check BP columns exist ────────────────────────────────────────────
        has_bps = raw_bps_col in group.columns and group[raw_bps_col].notna().any()
        has_bpd = raw_bpd_col in group.columns and group[raw_bpd_col].notna().any()

        if not has_bps or not has_bpd:
            print(f"[UCI] Patient {pid}: no BP columns — creating single sample.")
            DataLoader._append_sample(
                group, pid, None, None, samples,
                "uci", raw_ppg_col, raw_ecg_col, config,
            )
            return samples

        # ── Detect BP change points ───────────────────────────────────────────
        bps_vals = group[raw_bps_col].ffill()
        bpd_vals = group[raw_bpd_col].ffill()
        bp_changed = (
                (bps_vals != bps_vals.shift(1)) |
                (bpd_vals != bpd_vals.shift(1))
        )
        bp_changed.iloc[0] = True
        segment_ids = bp_changed.cumsum()
        n_segments = segment_ids.nunique()

        print(f"[UCI] Patient {pid}: {n_segments} BP segment(s) detected "
              f"across {len(group)} rows.")

        skipped_short = 0
        skipped_no_bp = 0
        skipped_trim = 0

        for seg_id, seg_group in group.groupby(segment_ids):
            seg_group = seg_group.copy().sort_values('standard_ts')

            # ── BP value for this segment ─────────────────────────────────────
            bps_series = seg_group[raw_bps_col].dropna()
            bpd_series = seg_group[raw_bpd_col].dropna()

            if bps_series.empty or bpd_series.empty:
                skipped_no_bp += 1
                continue

            bps = float(bps_series.iloc[0])
            bpd = float(bpd_series.iloc[0])

            # ── Check raw segment duration BEFORE any trimming ────────────────
            ts_min_raw = seg_group['standard_ts'].min()
            ts_max_raw = seg_group['standard_ts'].max()
            raw_duration_ms = ts_max_raw - ts_min_raw

            if raw_duration_ms < min_duration_msec:
                skipped_short += 1
                print(f"[UCI] Patient {pid} segment {seg_id} (SBP={bps}, DBP={bpd}): "
                      f"only {raw_duration_ms / 1000:.1f}s available "
                      f"(need {min_duration_msec / 1000:.0f}s) — skipping.")
                continue

            # ── Apply ECG/PPG alignment ───────────────────────────────────────
            ecg_arr = seg_group[raw_ecg_col].values if raw_ecg_col in seg_group.columns else None
            ppg_arr = seg_group[raw_ppg_col].values

            if ecg_arr is not None and len(ecg_arr) > 0:
                aligned_ppg = PulseDBStyleAligner.align(
                    ppg_arr, ecg_arr, config["target_fs"]
                )
                seg_group[raw_ppg_col] = aligned_ppg

            # ── Trim to last N ms (closest to BP measurement moment) ──────────
            # After trimming we re-check the duration guarantee
            if max_duration_msec is not None:
                ts_max = seg_group['standard_ts'].max()
                ts_cutoff = ts_max - max_duration_msec
                seg_group = seg_group[seg_group['standard_ts'] >= ts_cutoff]

                # Re-check after trim
                trimmed_duration_ms = (
                        seg_group['standard_ts'].max() -
                        seg_group['standard_ts'].min()
                )
                if trimmed_duration_ms < min_duration_msec:
                    skipped_trim += 1
                    print(f"[UCI] Patient {pid} segment {seg_id}: "
                          f"only {trimmed_duration_ms / 1000:.1f}s after trim "
                          f"(need {min_duration_msec / 1000:.0f}s) — skipping.")
                    continue

            # ── Final row count sanity check ──────────────────────────────────
            # At target_fs, 60s = 60*fs rows minimum
            if len(seg_group) < min_samples:
                skipped_short += 1
                print(f"[UCI] Patient {pid} segment {seg_id}: "
                      f"{len(seg_group)} rows < {min_samples} required "
                      f"({min_duration_msec / 1000:.0f}s × {fs}Hz) — skipping.")
                continue

            duration_s = (seg_group['standard_ts'].max() - seg_group['standard_ts'].min()) / 1000
            print(f"[UCI] Patient {pid} segment {seg_id}: "
                  f"SBP={bps}, DBP={bpd}, "
                  f"duration={duration_s:.1f}s, rows={len(seg_group)} ✓")

            DataLoader._append_sample(
                group=seg_group,
                pid=f"{pid}_seg{seg_id}",
                bps=bps, bpd=bpd,
                samples=samples,
                dataset_type="uci",
                raw_ppg_col=raw_ppg_col,
                raw_ecg_col=raw_ecg_col,
                config=config,
            )

        print(f"[UCI] Patient {pid}: {len(samples)} valid segment(s) produced. "
              f"Skipped: {skipped_short} too short, "
              f"{skipped_no_bp} no BP, "
              f"{skipped_trim} insufficient after trim.")

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
            ppg_cleaned = Preprocessor.clean_ppg_signal(ppg, fs)
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
            test_vpg = np.gradient(Preprocessor.clean_ppg_signal(aligned_ppg, fs))
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

