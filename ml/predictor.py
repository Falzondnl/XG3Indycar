"""
IndycarPredictor — Production inference engine.
Loads trained ensemble + calibrator + feature extractor.
Handles per-race win probability prediction with Harville podium estimation.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from config import HARVILLE_TOP_N, R0_DIR
from ml.features import FEATURES, IndycarFeatureExtractor
from ml.ensemble import IndycarEnsemble
from ml.calibrator import BetaCalibrator

logger = logging.getLogger(__name__)

ENSEMBLE_PKL = "ensemble.pkl"
CALIBRATOR_PKL = "calibrator.pkl"
EXTRACTOR_PKL = "extractor.pkl"


def _harville_podium(win_probs: np.ndarray, top_n: int = 3) -> np.ndarray:
    """
    Harville model for P(driver finishes in top-N).
    Exact calculation for n <= 50, simplified approximation for larger fields.
    """
    n = len(win_probs)
    if n == 0:
        return np.array([])

    win_probs = np.clip(win_probs, 1e-9, 1 - 1e-9)
    probs = win_probs / win_probs.sum()
    top_n = min(top_n, n - 1)
    podium_probs = np.zeros(n)

    if n <= 50 and top_n == 3:
        # Exact Harville for top-3
        for i in range(n):
            p_win = probs[i]

            # P(i finishes 2nd)
            p_2nd = 0.0
            for j in range(n):
                if j == i:
                    continue
                rest_sum = 1.0 - probs[j]
                if rest_sum < 1e-9:
                    continue
                p_2nd += probs[j] * (probs[i] / rest_sum)

            # P(i finishes 3rd)
            p_3rd = 0.0
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    rest_jk = 1.0 - probs[j] - probs[k]
                    if rest_jk < 1e-9:
                        continue
                    rest_j = 1.0 - probs[j]
                    if rest_j < 1e-9:
                        continue
                    p_3rd += (
                        probs[j]
                        * (probs[k] / rest_j)
                        * (probs[i] / rest_jk)
                    )

            podium_probs[i] = p_win + p_2nd + p_3rd
    else:
        # Simplified approximation for large fields
        for i in range(n):
            remaining_sum = 1.0
            p_not_top = 1.0
            for _ in range(top_n):
                p_slot = probs[i] / max(remaining_sum, 1e-9)
                p_not_top *= (1.0 - p_slot)
                remaining_sum -= probs[i]
                if remaining_sum < 1e-9:
                    break
            podium_probs[i] = 1.0 - p_not_top

    return np.clip(podium_probs, 0.0, 1.0)


class IndycarPredictor:
    """
    Production predictor for IndyCar race outcomes.
    Loads ensemble, calibrator, and feature extractor from model directory.
    """

    def __init__(self) -> None:
        self.ensemble: IndycarEnsemble | None = None
        self.calibrator: BetaCalibrator | None = None
        self.extractor: IndycarFeatureExtractor | None = None
        self._model_dir: str = R0_DIR
        self._loaded = False

    def load(self, model_dir: str = R0_DIR) -> "IndycarPredictor":
        """Load all model artefacts from model_dir."""
        self._model_dir = model_dir

        ensemble_path = os.path.join(model_dir, ENSEMBLE_PKL)
        calibrator_path = os.path.join(model_dir, CALIBRATOR_PKL)
        extractor_path = os.path.join(model_dir, EXTRACTOR_PKL)

        if not os.path.exists(ensemble_path):
            raise FileNotFoundError(f"Ensemble not found: {ensemble_path}")
        if not os.path.exists(calibrator_path):
            raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")
        if not os.path.exists(extractor_path):
            raise FileNotFoundError(f"Extractor not found: {extractor_path}")

        self.ensemble = IndycarEnsemble.load(ensemble_path)
        self.calibrator = BetaCalibrator.load(calibrator_path)
        self.extractor = IndycarFeatureExtractor.load(extractor_path)
        self._loaded = True
        logger.info(
            "IndycarPredictor loaded from %s (drivers=%d, teams=%d)",
            model_dir,
            self.extractor.driver_count,
            self.extractor.team_count,
        )
        return self

    def predict_race(
        self,
        drivers: list[dict[str, Any]],
        event_name: str,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Predict win + podium probabilities for a race field.

        drivers: list of dicts with keys:
            - driver_id: int (IndyCar driver ID; 0 if unknown)
            - driver_name: str
            - team_name: str (optional)
        event_name: str  (e.g. "Indianapolis 500", "Detroit Grand Prix")
        year: int        (defaults to current year if None)

        Returns list sorted by win_prob desc:
            [{driver_name, driver_id, team_name, win_prob, podium_prob}]
        """
        if not self._loaded:
            raise RuntimeError("Predictor not loaded — call load() first")
        if not drivers:
            raise ValueError("drivers list cannot be empty")

        import datetime
        if year is None:
            year = datetime.datetime.utcnow().year

        # Build feature rows
        feature_dicts = self.extractor.get_features_for_race(
            drivers=drivers,
            event_name=event_name,
            year=year,
        )
        if not feature_dicts:
            raise RuntimeError("Feature extractor returned empty result")

        X = pd.DataFrame(
            [{k: v for k, v in fd.items() if k in FEATURES} for fd in feature_dicts]
        )

        # Ensemble predict
        raw_probs = self.ensemble.predict_proba(X)

        # Calibrate
        cal_probs = self.calibrator.calibrate(raw_probs)

        # Normalise within race to sum=1
        total = cal_probs.sum()
        if total < 1e-9:
            cal_probs = np.ones(len(cal_probs)) / len(cal_probs)
        else:
            cal_probs = cal_probs / total

        # Harville podium (top 3)
        podium_probs = _harville_podium(cal_probs, top_n=3)

        results = []
        for i, fd in enumerate(feature_dicts):
            results.append({
                "driver_name": fd.get("driver_name", "Unknown"),
                "driver_id": fd.get("driver_id", 0),
                "team_name": fd.get("team_name", "Unknown"),
                "win_prob": float(cal_probs[i]),
                "podium_prob": float(podium_probs[i]),
            })

        results.sort(key=lambda x: x["win_prob"], reverse=True)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def driver_count(self) -> int:
        if self.extractor:
            return self.extractor.driver_count
        return 0

    @property
    def team_count(self) -> int:
        if self.extractor:
            return self.extractor.team_count
        return 0
