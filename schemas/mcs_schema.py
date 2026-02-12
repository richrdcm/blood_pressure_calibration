import pandas as pd
from schemas.bp_schema import BPSample

def map_mcs_to_internal(df: pd.DataFrame, fs: int = 125) -> BPSample:
    """Maps MCS Wearable headers to internal BPSample."""
    return BPSample(
        patient_id=str(df['patient_id'].iloc[0]),
        timestamp=df['timestamp'].tolist(),
        ppg=df['green'].tolist(),
        ecg=df['ecg'].tolist() if 'ecg' in df.columns else None,
        bps=float(df['bps'].iloc[0]),
        bpd=float(df['bpd'].iloc[0]),
        fs=fs
    )