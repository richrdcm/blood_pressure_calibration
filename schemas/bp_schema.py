from pydantic import BaseModel, Field
from typing import List, Optional

class BPSample(BaseModel):
    """Internal standardized format for a single measurement window."""
    patient_id: str
    ppg_timestamps: List[float]  # Standardized to milliseconds or seconds
    ppg: List[float]
    ecg_timestamps: Optional[List[float]] = None  # Standardized to milliseconds or seconds
    ecg: Optional[List[float]] = None
    bps: float  # Systolic
    bpd: float  # Diastolic
    target_fs: int     # Sampling frequency
    ppg_fs: int  # Raw PPG frequency
    ecg_fs: int  # Raw ECG frequency