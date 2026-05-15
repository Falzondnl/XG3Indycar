"""
Market pricing endpoints for IndyCar MS.
POST /api/v1/indycar/races/price  — predict + price all markets for a race
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
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

    LOCK-INDYCAR-TIER-2-CASCADE-001: when predictor unavailable, attempts
    OpticOdds Pinnacle scrape before refusing.
    LOCK-INDYCAR-TIER-3-REFUSE-503-001: structured 503 when both Tier 1 + 2 fail.
    LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001: prediction_source on every response.
    """
    import datetime
    year = body.year if body.year is not None else datetime.datetime.utcnow().year

    predictor_available = predictor is not None and predictor.is_loaded

    if not predictor_available:
        # LOCK-INDYCAR-TIER-2-CASCADE-001: attempt OpticOdds Pinnacle scrape.
        from feeds.optic_odds import OpticOddsFeed as _OpticFeed
        _optic = _OpticFeed()
        pinnacle_markets = None
        if _optic.is_available():
            try:
                pinnacle_markets = await _optic.get_race_odds_devigged(
                    event_name=body.event_name,
                    year=year,
                )
            except Exception as _t2_exc:
                logger.warning(
                    "indycar_tier2_optic_failed event=%s year=%d error=%s "
                    "LOCK-INDYCAR-TIER-2-CASCADE-001",
                    body.event_name, year, _t2_exc,
                )

        if pinnacle_markets is not None:
            logger.info(
                "indycar_tier2_market_scrape event=%s year=%d "
                "LOCK-INDYCAR-TIER-2-CASCADE-001 LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001",
                body.event_name, year,
            )
            return {
                "success": True,
                "event_name": body.event_name,
                "year": year,
                "field_size": len(body.drivers),
                "markets": pinnacle_markets,
                "prediction_source": "market_scrape",
                "model_available": False,
                "tier": 2,
            }

        # LOCK-INDYCAR-TIER-3-REFUSE-503-001
        _cid = str(uuid.uuid4())
        logger.error(
            "indycar_fixture_unpriced event=%s year=%d correlation_id=%s "
            "LOCK-INDYCAR-TIER-3-REFUSE-503-001",
            body.event_name, year, _cid,
        )
        return JSONResponse(
            status_code=503,
            content={
                "code": "FIXTURE_UNPRICED",
                "reason": "no_model_no_market_data",
                "message": (
                    f"IndyCar predictor not loaded and Optic Odds Pinnacle scrape "
                    f"unavailable for {body.event_name} {year}. "
                    f"Cannot price: silent fallback to uniform odds is forbidden. "
                    f"LOCK-INDYCAR-TIER-3-REFUSE-503-001"
                ),
                "correlation_id": _cid,
                "retry_after": 30,
                "event_name": body.event_name,
                "year": year,
            },
        )

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

    # LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001: prediction_source MUST be present.
    return {
        "success": True,
        "event_name": body.event_name,
        "year": year,
        "field_size": len(predictions),
        "top_predictions": predictions[:5],
        "markets": market_bundle["markets"],
        "is_indy500": market_bundle.get("is_indy500", False),
        "prediction_source": "model",
    }
