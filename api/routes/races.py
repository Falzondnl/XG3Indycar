"""
Race prediction endpoints for IndyCar MS.
GET  /api/v1/indycar/races/upcoming  — fetch upcoming races from Optic Odds
POST /api/v1/indycar/races/predict   — run ML predictor on supplied driver field
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from api._deps import get_optic_feed, get_predictor
from feeds.optic_odds import OpticOddsFeed
from ml.predictor import IndycarPredictor

logger = logging.getLogger(__name__)
router = APIRouter()


class DriverInput(BaseModel):
    driver_id: int = Field(default=0, description="IndyCar driver ID (0 if unknown)")
    driver_name: str = Field(..., min_length=1, description="Driver full name")
    team_name: str = Field(default="Unknown", description="Team name")


class PredictRaceRequest(BaseModel):
    drivers: list[DriverInput] = Field(
        ..., min_length=2, description="Race field (minimum 2 drivers)"
    )
    event_name: str = Field(..., min_length=2, description="Event name (e.g. 'Indianapolis 500')")
    year: int | None = Field(default=None, ge=1996, le=2050, description="Season year (defaults to current)")


class PredictRaceResponse(BaseModel):
    event_name: str
    year: int
    field_size: int
    predictions: list[dict[str, Any]]


@router.get("/upcoming")
async def get_upcoming_races(
    limit: int = 20,
    feed: OpticOddsFeed = Depends(get_optic_feed),
) -> dict[str, Any]:
    """Fetch upcoming IndyCar races from Optic Odds."""
    try:
        races = await feed.get_upcoming_races(limit=min(limit, 100))
        return {
            "count": len(races),
            "races": races,
        }
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Failed to fetch upcoming races: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Optic Odds API error: {exc}",
        )


@router.post("/predict", response_model=PredictRaceResponse)
async def predict_race(
    body: PredictRaceRequest,
    predictor: IndycarPredictor = Depends(get_predictor),
) -> PredictRaceResponse:
    """
    Predict win + podium probabilities for a supplied IndyCar race field.
    Drivers with unknown driver_id default to ELO 1500 and zero career stats.
    """
    import datetime
    year = body.year if body.year is not None else datetime.datetime.utcnow().year
    drivers_input = [d.model_dump() for d in body.drivers]

    try:
        predictions = predictor.predict_race(
            drivers=drivers_input,
            event_name=body.event_name,
            year=year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    return PredictRaceResponse(
        event_name=body.event_name,
        year=year,
        field_size=len(predictions),
        predictions=predictions,
    )
