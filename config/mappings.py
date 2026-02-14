"""
config/mappings.py
Source-to-Internal mapping configurations for the BP Estimation Library.
"""

# The internal standardized keys used by the Pydantic BPSample schema
INTERNAL_KEYS = {
    "ppg": "ppg",
    "ecg": "ecg",
    "timestamp": "timestamp",
    "bps": "bps",
    "bpd": "bpd",
    "patient_id": "patient_id"
}

# Mapping for the UCI (MIMIC-II) Dataset
# Note: UCI often labels PPG as 'pleth' or 'PPG' and ECG as 'II' or 'V'
UCI_MAPPING = {
    "dataset_name": "uci",
    "columns": {
        "ppg": "ppg", "ecg": "ecg", "timestamp": "timestamp",
        "bps": "bps", "bpd": "bpd", "patient_id": "patient_id"
    },
    "channels": {
        "ppg": {"fs": 125},
        "ecg": {"fs": 125}
    },
    "target_fs": 125, # The frequency the library will use internally
    "unit_conversion": {"timestamp": 1000.0} # seconds to ms
}

MCS_MAPPING = {
    "dataset_name": "mcs",
    "columns": {
        "green": "ppg", "ecg": "ecg", "timestamp": "timestamp",
        "bps": "bps", "bpd": "bpd", "patient_id": "patient_id"
    },
    "channels": {
        "ppg": {"fs": 20},   # <--- Change this if the watch updates
        "ecg": {"fs": 125}   # <--- Change this if the watch updates
    },
    "target_fs": 125,
    "unit_conversion": {"timestamp": 1.0} # already ms
}

DATASET_CONFIGS = {"uci": UCI_MAPPING, "mcs": MCS_MAPPING}