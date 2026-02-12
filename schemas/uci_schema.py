import pandas as pd
from schemas.bp_schema import BPSample

def map_uci_to_internal(df: pd.DataFrame, patient_id: str, fs: int = 125) -> BPSample:
    """Maps UCI/MIMIC headers to internal BPSample."""
    return BPSample(
        patient_id=str(df.get('subject_id', [patient_id])[0]),
        timestamp=df['time'].tolist(),
        ppg=df['PPG'].tolist(),
        ecg=df['II'].tolist() if 'II' in df.columns else None,
        bps=float(df['sys'].iloc[0]),
        bpd=float(df['dia'].iloc[0]),
        fs=fs
    )