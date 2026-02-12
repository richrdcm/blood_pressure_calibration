from pydantic import BaseModel, Field
from typing import List, Optional

class BPSample(BaseModel):
    """Internal standardized format for a single measurement window."""
    patient_id: str
    timestamp: List[float]  # Standardized to milliseconds or seconds
    ppg: List[float]
    ecg: Optional[List[float]] = None
    bps: float  # Systolic
    bpd: float  # Diastolic
    fs: int     # Sampling frequency