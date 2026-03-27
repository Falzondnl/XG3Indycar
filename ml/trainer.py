"""
IndyCar Model Trainer
=====================
Temporal split: train=1996-2015, val=2016-2019, test=2020-2026
Saves ensemble.pkl, calibrator.pkl, extractor.pkl to models/r0/
Reports: AUC, Brier on win prediction (test set, race-normalised).
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import (
    INDYCAR_CSV,
    R0_DIR,
    TRAIN_YEARS_MAX,
    VAL_YEARS_MIN,
    VAL_YEARS_MAX,
    TEST_YEARS_MIN,
)
from ml.features import FEATURES, IndycarFeatureExtractor
from ml.ensemble import IndycarEnsemble
from ml.calibrator import BetaCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class IndycarTrainer:
    """Full training pipeline for IndyCar ML models."""

    def train(
        self,
        csv_path: str = INDYCAR_CSV,
        out_dir: str = R0_DIR,
    ) -> dict:
        """
        Run full training pipeline.
        Returns metrics dict: auc, brier, n_train, n_val, n_test.
        """
        t_start = time.time()
        os.makedirs(out_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # 1. Feature extraction (chronological, no leakage)
        # ----------------------------------------------------------------
        logger.info("=== STEP 1: Feature extraction ===")
        extractor = IndycarFeatureExtractor()
        dataset = extractor.build_dataset(csv_path)

        if dataset.empty:
            raise RuntimeError("Feature extraction returned empty dataset — check CSV path")

        logger.info(
            "Dataset: %d rows, %d races, %d positive (win rate=%.4f)",
            len(dataset),
            dataset["race_id"].nunique(),
            int(dataset["target"].sum()),
            dataset["target"].mean(),
        )

        # ----------------------------------------------------------------
        # 2. Temporal split
        # ----------------------------------------------------------------
        logger.info("=== STEP 2: Temporal train/val/test split ===")
        train_df = dataset[dataset["year"] <= TRAIN_YEARS_MAX].copy()
        val_df = dataset[
            (dataset["year"] >= VAL_YEARS_MIN) & (dataset["year"] <= VAL_YEARS_MAX)
        ].copy()
        test_df = dataset[dataset["year"] >= TEST_YEARS_MIN].copy()

        logger.info(
            "Split sizes — train=%d (<=2015), val=%d (2016-2019), test=%d (>=2020)",
            len(train_df), len(val_df), len(test_df),
        )
        logger.info(
            "Target rates — train=%.4f val=%.4f test=%.4f",
            train_df["target"].mean(),
            val_df["target"].mean(),
            test_df["target"].mean(),
        )

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise RuntimeError("One or more temporal splits are empty — check year filtering")

        X_train = train_df[FEATURES]
        y_train = train_df["target"]
        groups_train = train_df["race_id"]

        X_val = val_df[FEATURES]
        y_val = val_df["target"]

        X_test = test_df[FEATURES]
        y_test = test_df["target"]

        # ----------------------------------------------------------------
        # 3. Ensemble training
        # ----------------------------------------------------------------
        logger.info("=== STEP 3: Ensemble training (CatBoost + LightGBM + XGBoost) ===")
        ensemble = IndycarEnsemble()
        ensemble.fit(X_train, y_train, groups_train, X_val, y_val)

        # ----------------------------------------------------------------
        # 4. Calibration on validation set
        # ----------------------------------------------------------------
        logger.info("=== STEP 4: Calibration on validation set ===")
        val_raw = ensemble.predict_proba(X_val)
        calibrator = BetaCalibrator()
        calibrator.fit(val_raw, y_val.values)

        # ----------------------------------------------------------------
        # 5. Evaluation on test set
        # ----------------------------------------------------------------
        logger.info("=== STEP 5: Test set evaluation ===")
        test_raw = ensemble.predict_proba(X_test)
        test_cal = calibrator.calibrate(test_raw)

        auc_raw = roc_auc_score(y_test, test_cal)
        brier_raw = brier_score_loss(y_test, test_cal)

        # Race-normalised probabilities (sum to 1 within each race)
        test_copy = test_df.copy()
        test_copy["cal_prob"] = test_cal
        race_totals = test_copy.groupby("race_id")["cal_prob"].transform("sum")
        test_copy["norm_prob"] = test_copy["cal_prob"] / race_totals.clip(lower=1e-9)

        auc_norm = roc_auc_score(y_test, test_copy["norm_prob"])
        brier_norm = brier_score_loss(y_test, test_copy["norm_prob"])

        logger.info(
            "=== RESULTS === Raw: AUC=%.4f Brier=%.4f | Normalised: AUC=%.4f Brier=%.4f",
            auc_raw, brier_raw, auc_norm, brier_norm,
        )

        # ----------------------------------------------------------------
        # 6. Save artefacts
        # ----------------------------------------------------------------
        logger.info("=== STEP 6: Saving artefacts to %s ===", out_dir)
        ensemble.save(os.path.join(out_dir, "ensemble.pkl"))
        calibrator.save(os.path.join(out_dir, "calibrator.pkl"))
        extractor.save(os.path.join(out_dir, "extractor.pkl"))

        elapsed = time.time() - t_start
        logger.info("Training complete in %.1fs", elapsed)

        metrics = {
            "auc_raw": round(auc_raw, 4),
            "brier_raw": round(brier_raw, 4),
            "auc_normalised": round(auc_norm, 4),
            "brier_normalised": round(brier_norm, 4),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_drivers": extractor.driver_count,
            "n_teams": extractor.team_count,
            "elapsed_seconds": round(elapsed, 1),
        }
        logger.info("Metrics: %s", metrics)
        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train IndyCar ML models")
    parser.add_argument("--csv", default=INDYCAR_CSV, help="Path to IndyCar session CSV")
    parser.add_argument("--out", default=R0_DIR, help="Output directory for model artefacts")
    args = parser.parse_args()

    trainer = IndycarTrainer()
    metrics = trainer.train(csv_path=args.csv, out_dir=args.out)
    print("\n=== FINAL METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
