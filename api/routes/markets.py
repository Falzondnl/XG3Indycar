"""
Market pricing endpoints for IndyCar MS.
POST /api/v1/indycar/races/price  — predict + price all markets for a race
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api._deps import get_predictor
from ml.predictor import IndycarPredictor
from pricing.markets import build_all_markets
from config import WIN_MARGIN, PODIUM_MARGIN, H2H_MARGIN

logger = logging.getLogger(__name__)
router = APIRouter()


class DriverInput(BaseModel):
    driver_id: int = Field(default=0)
    driver_name: str = Field(..., min_length=1)
    team_name: str = Field(default="Unknown")


class PriceRaceRequest(BaseModel):
    drivers: list[DriverInput] = Field(..., min_length=2)
    event_name: str = Field(..., min_length=2)
    year: int | None = Field(default=None, ge=1996, le=2050)
    win_margin: float = Field(default=WIN_MARGIN, ge=0.0, le=0.5)
    podium_margin: float = Field(default=PODIUM_MARGIN, ge=0.0, le=0.5)
    h2h_margin: float = Field(default=H2H_MARGIN, ge=0.0, le=0.5)


@router.post("/price")
async def price_race(
    body: PriceRaceRequest,
    predictor: IndycarPredictor = Depends(get_predictor),
) -> dict[str, Any]:
    """
    Run the full prediction + pricing pipeline for an IndyCar race.
    Returns race_winner, podium_finisher, and h2h markets.
    Automatically detects Indianapolis 500 and adds special market notes.
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

    try:
        market_bundle = build_all_markets(
            drivers=predictions,
            event_name=body.event_name,
            win_margin=body.win_margin,
            podium_margin=body.podium_margin,
            h2h_margin=body.h2h_margin,
        )
    except Exception as exc:
        logger.error("Market building error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Market pricing failed: {exc}",
        )

    return {
        "success": True,
        "event_name": body.event_name,
        "year": year,
        "field_size": len(predictions),
        "top_predictions": predictions[:5],
        "markets": market_bundle["markets"],
        "is_indy500": market_bundle.get("is_indy500", False),
    }
