"""
src/extractor/sqi.py

Signal Quality Index (SQI) scoring per beat.

Each beat receives a composite score in [0, 100] based on:
  1. Morphology completeness  – are all APG fiducials (a, b, c, d, e) present?
  2. PTT physiological range  – PTT_a in [80, 450] ms (normal HR 40–180 bpm)?
  3. PPG amplitude            – systolic amplitude above noise floor?
  4. APG polarity             – 'a' wave positive, 'b' wave negative (as expected)?
  5. Reflection time sanity   – diastolic reflection lag in [50, 400] ms?

The score is a weighted average of the individual sub-scores (each 0–1).
Weights are tunable via SQIScorer constructor kwargs.
"""

import numpy as np
from typing import Optional


# Default physiological thresholds
_PTT_A_MIN_MS = 80.0
_PTT_A_MAX_MS = 450.0
_REFLECTION_MIN_MS = 50.0
_REFLECTION_MAX_MS = 400.0


class SQIScorer:
    """
    Computes a per-beat Signal Quality Index.

    Parameters
    ----------
    w_morphology : float
        Weight for APG fiducial completeness (default 0.30).
    w_ptt_range : float
        Weight for PTT_a physiological plausibility (default 0.30).
    w_amplitude : float
        Weight for systolic PPG amplitude above noise (default 0.20).
    w_apg_polarity : float
        Weight for APG a/b polarity check (default 0.10).
    w_reflection : float
        Weight for reflection time sanity (default 0.10).
    """

    def __init__(
        self,
        w_morphology: float = 0.30,
        w_ptt_range: float = 0.30,
        w_amplitude: float = 0.20,
        w_apg_polarity: float = 0.10,
        w_reflection: float = 0.10,
    ):
        total = w_morphology + w_ptt_range + w_amplitude + w_apg_polarity + w_reflection
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"SQI weights must sum to 1.0, got {total:.4f}")
        self.weights = {
            "morphology": w_morphology,
            "ptt_range": w_ptt_range,
            "amplitude": w_amplitude,
            "apg_polarity": w_apg_polarity,
            "reflection": w_reflection,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_beats(
        self,
        ppg_signal: list,
        apg_signal: list,
        waves: list,
        time_features: list,
        fs: int,
    ) -> list:
        """
        Score every beat and return an enriched list of dicts.

        Parameters
        ----------
        ppg_signal  : cleaned PPG array
        apg_signal  : APG (second derivative of PPG)
        waves       : output of MorphologyExtractor.extract_sample_waves()[2]
        time_features : output of TimeExtractor.compute_heartbeat_times()
        fs          : sampling frequency (Hz)

        Returns
        -------
        List of dicts, one per beat, each containing:
          - all original time_feature keys
          - 'sqi_score'      : composite score 0–100
          - 'sqi_components' : dict of individual sub-scores
        """
        ppg_arr = np.array(ppg_signal)
        apg_arr = np.array(apg_signal)

        # Build a lookup: a_idx -> wave dict
        wave_by_a = {w["a"]: w for w in waves if w.get("a") is not None}

        # Global amplitude stats (used for relative amplitude check)
        ppg_p2p = float(np.ptp(ppg_arr)) if len(ppg_arr) > 0 else 1.0

        scored = []
        for tf in time_features:
            a_idx = tf.get("a_idx")
            wave = wave_by_a.get(a_idx)

            components = {
                "morphology":   self._score_morphology(wave),
                "ptt_range":    self._score_ptt_range(tf.get("ptt_a"), fs),
                "amplitude":    self._score_amplitude(ppg_arr, wave, ppg_p2p),
                "apg_polarity": self._score_apg_polarity(apg_arr, wave),
                "reflection":   self._score_reflection(tf.get("reflection_time"), fs),
            }

            composite = sum(
                self.weights[k] * v for k, v in components.items()
            )
            sqi = round(composite * 100, 1)

            scored.append({
                **tf,
                "sqi_score": sqi,
                "sqi_components": {k: round(v * 100, 1) for k, v in components.items()},
            })

        return scored

    # ------------------------------------------------------------------
    # Sub-scorers (each returns a float in [0, 1])
    # ------------------------------------------------------------------

    @staticmethod
    def _score_morphology(wave: Optional[dict]) -> float:
        """Fraction of expected APG fiducials that are present."""
        if wave is None:
            return 0.0
        keys = ["a", "b", "c", "d", "e", "ppg_peak"]
        present = sum(1 for k in keys if wave.get(k) is not None)
        return present / len(keys)

    @staticmethod
    def _score_ptt_range(ptt_a_ms: Optional[float], fs: int) -> float:
        """
        PTT_a is already in seconds in the time_feature dict (computed as
        timestamp difference). Convert to ms for the threshold comparison.
        Returns 1.0 if inside physiological range, decays linearly outside.
        """
        if ptt_a_ms is None:
            return 0.0
        # time_features stores PTT in the same units as timestamps (ms)
        ptt = float(ptt_a_ms)
        if _PTT_A_MIN_MS <= ptt <= _PTT_A_MAX_MS:
            return 1.0
        # Linear penalty outside range (0 at 2× the boundary distance)
        if ptt < _PTT_A_MIN_MS:
            margin = _PTT_A_MIN_MS
            return max(0.0, 1.0 - ((_PTT_A_MIN_MS - ptt) / margin))
        else:
            margin = _PTT_A_MAX_MS
            return max(0.0, 1.0 - ((ptt - _PTT_A_MAX_MS) / margin))

    @staticmethod
    def _score_amplitude(ppg_arr: np.ndarray, wave: Optional[dict], ppg_p2p: float) -> float:
        """Systolic peak amplitude relative to global p2p range."""
        if wave is None or ppg_arr is None or ppg_p2p < 1e-9:
            return 0.0
        sys_idx = wave.get("ppg_peak")
        if sys_idx is None or sys_idx >= len(ppg_arr):
            return 0.0
        sys_amp = float(ppg_arr[sys_idx]) - float(np.min(ppg_arr))
        ratio = sys_amp / ppg_p2p
        # Expect systolic peak to be ≥ 20 % of p2p; full score at ≥ 50 %
        return float(np.clip((ratio - 0.20) / 0.30, 0.0, 1.0))

    @staticmethod
    def _score_apg_polarity(apg_arr: np.ndarray, wave: Optional[dict]) -> float:
        """Check that APG 'a' > 0 and 'b' < 0 (canonical waveform polarity)."""
        if wave is None or apg_arr is None:
            return 0.0
        score = 0.0
        a_idx = wave.get("a")
        b_idx = wave.get("b")
        if a_idx is not None and a_idx < len(apg_arr):
            score += 0.5 if apg_arr[a_idx] > 0 else 0.0
        if b_idx is not None and b_idx < len(apg_arr):
            score += 0.5 if apg_arr[b_idx] < 0 else 0.0
        return score

    @staticmethod
    def _score_reflection(reflection_time_ms: Optional[float], fs: int) -> float:
        """Reflection time (dp − sys) in physiological range."""
        if reflection_time_ms is None:
            return 0.0
        rt = float(reflection_time_ms)
        if _REFLECTION_MIN_MS <= rt <= _REFLECTION_MAX_MS:
            return 1.0
        if rt < _REFLECTION_MIN_MS:
            return max(0.0, rt / _REFLECTION_MIN_MS)
        else:
            excess = rt - _REFLECTION_MAX_MS
            return max(0.0, 1.0 - (excess / _REFLECTION_MAX_MS))

