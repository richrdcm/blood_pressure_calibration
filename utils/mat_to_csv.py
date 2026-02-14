import h5py
import pandas as pd
import numpy as np
import os


def convert_uci_v73_mat_to_csv(mat_path, output_path):
    """
    Converts a UCI Part_X.mat (v7.3 HDF5) file into a consolidated CSV.
    Renames columns to: patient_id, timestamp, ppg, ecg, bps, bpd.
    """
    print(f"Loading {mat_path} via h5py...")

    try:
        with h5py.File(mat_path, 'r') as f:
            # 1. Identify the main key (usually 'p')
            main_key = 'p'
            if main_key not in f.keys():
                main_key = [k for k in f.keys() if not k.startswith('#')][0]

            print(f"Found main key: {main_key}")
            refs = f[main_key]

            # Flatten references
            ref_list = refs[0] if refs.shape[0] == 1 else refs[:, 0]
            total_patients = len(ref_list)
            print(f"Total patients found in file: {total_patients}")

            all_patients_data = []

            for i, ref in enumerate(ref_list):
                try:
                    # 2. Dereference
                    patient_data = f[ref]
                    mat = np.array(patient_data)

                    # 3. Shape Handling
                    if mat.shape[0] != 3 and mat.shape[1] == 3:
                        mat = mat.T

                    if mat.shape[0] != 3:
                        print(f"Skipping Patient {i}: Unexpected shape {mat.shape}")
                        continue

                    # Extract Raw Signals
                    # Row 0: PPG, Row 1: ABP, Row 2: ECG (Lead II)
                    raw_ppg = mat[0, :]
                    raw_abp = mat[1, :]
                    raw_ecg = mat[2, :]

                    # 4. Extract BP Targets (Sys/Dia)
                    # We calculate the scaler Systolic/Diastolic for this segment
                    val_sys = np.max(raw_abp)
                    val_dia = np.min(raw_abp)

                    # 5. Create Timestamps
                    timestamps = np.arange(len(raw_ppg)) * (1.0 / 125.0)

                    # 6. Build ID
                    filename = os.path.basename(mat_path).split('.')[0]
                    patient_id = f"{filename}_{i:04d}"

                    # 7. Create DataFrame with Requested Column Names
                    # Mappings:
                    # "subject_id" -> "patient_id"
                    # "time"       -> "timestamp"
                    # "PPG"        -> "ppg"
                    # "II"         -> "ecg"
                    # "sys"        -> "bps"
                    # "dia"        -> "bpd"

                    df = pd.DataFrame({
                        'patient_id': patient_id,
                        'timestamp': timestamps,
                        'ppg': raw_ppg,
                        'ecg': raw_ecg,
                        'bps': val_sys,  # Constant value for this segment
                        'bpd': val_dia  # Constant value for this segment
                    })

                    all_patients_data.append(df)

                except Exception as e:
                    print(f"Error reading Patient {i}: {e}")
                    continue

                if i % 100 == 0:
                    print(f"Processed {i}/{total_patients} patients...")

            # 8. Concatenate and Save
            print("Concatenating all patients into one CSV...")
            if not all_patients_data:
                print("No data found!")
                return

            final_df = pd.concat(all_patients_data, ignore_index=True)

            print(f"Saving to {output_path}...")
            final_df.to_csv(output_path, index=False)
            print("Done.")

    except OSError:
        print(f"Could not open {mat_path}. Is it a valid .mat file?")

# Usage
convert_uci_v73_mat_to_csv('datasets/raw/uci/Part_1.mat', 'datasets/raw/uci/Part_1.csv')