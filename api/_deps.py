"""
FastAPI dependency injection for IndyCar MS.
Provides predictor and feed instances from app.state.
"""
from __future__ import annotations

from fastapi import HTTPException, Request, status

from ml.predictor import IndycarPredictor
from feeds.optic_odds import OpticOddsFeed


def get_predictor(request: Request) -> IndycarPredictor:
    predictor: IndycarPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IndyCar predictor not loaded — service initialising or models not trained",
        )
    return predictor


def get_optic_feed(request: Request) -> OpticOddsFeed:
    feed: OpticOddsFeed | None = getattr(request.app.state, "optic_feed", None)
    if feed is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Optic Odds feed not initialised",
        )
    return feed
