from pydantic import BaseModel, Field
from typing import List, Optional

class BPSample(BaseModel):
    """Internal standardized format for a single measurement window."""

    # Identity
    patient_id: str

    # PPG signal
    ppg_timestamps: List[float]
    ppg: List[float]
    ppg_fs: int

    # ECG signal (optional — not all devices record ECG)
    ecg_timestamps: Optional[List[float]] = None
    ecg: Optional[List[float]] = None
    ecg_fs: int

    # Blood pressure ground truth (optional — not always available)
    bps: Optional[float] = None  # Systolic
    bpd: Optional[float] = None  # Diastolic

    # Sampling
    target_fs: int  # Unified frequency after resampling

    # Signal Quality Intervals — list of [start_idx, end_idx] pairs
    # Indices refer to the SYNCED, RESAMPLED signal arrays (post clean_signals)
    ppg_sqi:   Optional[List[List[int]]] = None  # PPG-only good intervals
    ecg_sqi:   Optional[List[List[int]]] = None  # ECG-only good intervals
    joint_sqi: Optional[List[List[int]]] = None  # intervals where BOTH are good