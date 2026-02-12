import h5py
import pandas as pd
import numpy as np
import os


def convert_uci_v73_mat_to_csv(mat_path, output_path):
    """
    Converts a UCI Part_X.mat (v7.3 HDF5) file into a consolidated CSV.
    """
    print(f"Loading {mat_path} via h5py...")

    with h5py.File(mat_path, 'r') as f:
        # 1. Identify the main key (usually 'p' or 'cell')
        # In HDF5 format, the variables are keys in the root
        main_key = [k for k in f.keys() if k != '#refs#'][0]
        data_refs = f[main_key][0]  # Array of references to the actual matrices

        all_patients_data = []
        total_patients = len(data_refs)

        for i, ref in enumerate(data_refs):
            # 2. Dereference the patient matrix
            # In HDF5, f[ref] accesses the specific patient's 3xN matrix
            patient_matrix = np.array(f[ref])

            # Note: h5py reads matrices as (Samples, Channels) or (3, Samples)
            # UCI standard: row 0=PPG, row 1=ABP, row 2=ECG
            if patient_matrix.shape[0] != 3:
                # If transposed, fix it:
                if patient_matrix.shape[1] == 3:
                    patient_matrix = patient_matrix.T
                else:
                    continue

            ppg = patient_matrix[0]
            abp = patient_matrix[1]
            ecg = patient_matrix[2]

            # 3. Label Extraction
            bps = np.max(abp)
            bpd = np.min(abp)

            # 4. Create local timestamps (125Hz)
            timestamps = np.arange(len(ppg)) * (1.0 / 125.0)

            # Build Subject ID
            filename = os.path.basename(mat_path).split('.')[0]
            temp_df = pd.DataFrame({
                'subject_id': f"UCI_{filename}_{i:04d}",
                'time': timestamps,
                'PPG': ppg,
                'II': ecg,
                'sys': bps,
                'dia': bpd
            })

            all_patients_data.append(temp_df)

            if i % 100 == 0:
                print(f"Processed {i}/{total_patients} patients...")

    # 5. Concatenate and Save
    print("Saving consolidated CSV...")
    final_df = pd.concat(all_patients_data, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"Success! {output_path} is ready.")

# Usage:
convert_uci_v73_mat_to_csv('datasets/raw/uci/Part_1.mat', 'datasets/raw/uci/Part_1.csv')