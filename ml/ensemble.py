"""
IndyCar 3-Model Stacking Ensemble
CatBoost + LightGBM + XGBoost → LogisticRegression meta-learner
GroupKFold on race_id to prevent data leakage within a race.
"""
from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ml.features import FEATURES

logger = logging.getLogger(__name__)

CB_PARAMS: dict[str, Any] = {
    "iterations": 400,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "class_weights": {0: 1.0, 1: 15.0},  # Approximate balance for ~6% win rate
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 40,
}

LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "scale_pos_weight": 15,  # Compensate for ~6% positive rate
    "random_state": 42,
    "verbosity": 0,
}

N_SPLITS = 5


class IndycarEnsemble:
    """
    3-model stacking ensemble for IndyCar race win probability.
    Base models: CatBoost, LightGBM, XGBoost
    Meta-learner: LogisticRegression (trained on GroupKFold OOF predictions)
    """

    def __init__(self) -> None:
        self.cb_model: Any = None
        self.lgb_model: Any = None
        self.xgb_model: Any = None
        self.meta: LogisticRegression | None = None
        self._feature_names = FEATURES

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "IndycarEnsemble":
        """
        Fit base models with validation set for early stopping.
        Then fit meta-learner using GroupKFold OOF predictions.
        """
        from catboost import CatBoostClassifier, Pool
        import lightgbm as lgb
        import xgboost as xgb

        logger.info(
            "Fitting IndyCar ensemble: train=%d val=%d features=%d positive_rate=%.4f",
            len(X_train), len(X_val), len(self._feature_names), y_train.mean(),
        )

        # ---- CatBoost ----
        logger.info("Training CatBoost ...")
        cb = CatBoostClassifier(**CB_PARAMS)
        train_pool = Pool(X_train[FEATURES], label=y_train)
        val_pool = Pool(X_val[FEATURES], label=y_val)
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        self.cb_model = cb
        logger.info("CatBoost done. Best iteration: %d", cb.get_best_iteration())

        # ---- LightGBM ----
        logger.info("Training LightGBM ...")
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(
            X_train[FEATURES], y_train,
            eval_set=[(X_val[FEATURES], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        self.lgb_model = lgb_model
        logger.info("LightGBM done.")

        # ---- XGBoost (3.x API) ----
        logger.info("Training XGBoost ...")
        xgb_model = xgb.XGBClassifier(
            **XGB_PARAMS,
            callbacks=[xgb.callback.EarlyStopping(rounds=40, save_best=True)],
        )
        xgb_model.fit(
            X_train[FEATURES], y_train,
            eval_set=[(X_val[FEATURES], y_val)],
            verbose=False,
        )
        self.xgb_model = xgb_model
        logger.info("XGBoost done.")

        # ---- Meta-learner via GroupKFold OOF ----
        logger.info("Building GroupKFold OOF predictions for meta-learner ...")
        gkf = GroupKFold(n_splits=N_SPLITS)
        oof_cb = np.zeros(len(X_train))
        oof_lgb = np.zeros(len(X_train))
        oof_xgb = np.zeros(len(X_train))

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(X_train, y_train, groups=groups_train)
        ):
            Xf_tr = X_train.iloc[train_idx][FEATURES]
            yf_tr = y_train.iloc[train_idx]
            Xf_val = X_train.iloc[val_idx][FEATURES]

            # CatBoost OOF fold
            cb_f = CatBoostClassifier(**CB_PARAMS)
            cb_f.fit(Pool(Xf_tr, label=yf_tr), verbose=0)
            oof_cb[val_idx] = cb_f.predict_proba(Xf_val)[:, 1]

            # LightGBM OOF fold
            lgb_f = lgb.LGBMClassifier(**LGB_PARAMS)
            lgb_f.fit(Xf_tr, yf_tr, callbacks=[lgb.log_evaluation(-1)])
            oof_lgb[val_idx] = lgb_f.predict_proba(Xf_val)[:, 1]

            # XGBoost OOF fold
            xgb_f = xgb.XGBClassifier(**XGB_PARAMS)
            xgb_f.fit(Xf_tr, yf_tr, verbose=False)
            oof_xgb[val_idx] = xgb_f.predict_proba(Xf_val)[:, 1]

            logger.info("  OOF Fold %d/%d done", fold + 1, N_SPLITS)

        meta_X = np.column_stack([oof_cb, oof_lgb, oof_xgb])
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta.fit(meta_X, y_train.values)
        logger.info("Meta-learner fitted. Coefs: %s", self.meta.coef_)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(win) for each row as 1D array."""
        if self.meta is None:
            raise RuntimeError("Ensemble not fitted — call fit() first")
        Xf = X[FEATURES] if isinstance(X, pd.DataFrame) else X

        cb_prob = self.cb_model.predict_proba(Xf)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(Xf)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(Xf)[:, 1]

        meta_X = np.column_stack([cb_prob, lgb_prob, xgb_prob])
        return self.meta.predict_proba(meta_X)[:, 1]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("IndycarEnsemble saved to %s", path)

    @staticmethod
    def load(path: str) -> "IndycarEnsemble":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("IndycarEnsemble loaded from %s", path)
        return obj
