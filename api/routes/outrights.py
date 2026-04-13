"""
IndyCar Season Championship Outright Market
=============================================

Produces IndyCar series championship winner prices derived from the live
ELO ratings in the FeatureExtractor (app.state.predictor.extractor).

Endpoint
--------
GET /api/v1/indycar/outrights/championship?season=2026&track_type=overall

Returns a ranked list of drivers with fair win probability and decimal odds
at the configured championship margin (10%).

Methodology
-----------
Uses Harville softmax on overall ELO (elo_overall dict) to produce
championship win probabilities.  Driver names are resolved from the
extractor's career statistics if available; otherwise driver_id is used.

This mirrors the F1 WDC endpoint pattern:
  GET /api/v1/formula1/outrights/wdc
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_CHAMPIONSHIP_MARGIN: float = float(os.getenv("CHAMPIONSHIP_MARGIN", "0.10"))
_TOP_N_CHAMPIONSHIP: int = int(os.getenv("CHAMPIONSHIP_TOP_N", "30"))


def _harville_softmax_probs(elo_ratings: list[tuple[Any, float]], temperature: float = 400.0) -> list[tuple[Any, float]]:
    """Convert ELO ratings to championship win probabilities via Harville softmax."""
    if not elo_ratings:
        return []
    max_elo = max(elo for _, elo in elo_ratings)
    exp_vals = [(key, math.exp((elo - max_elo) / temperature)) for key, elo in elo_ratings]
    total = sum(v for _, v in exp_vals)
    if total == 0:
        n = len(exp_vals)
        return [(key, 1.0 / n) for key, _ in exp_vals]
    probs = [(key, v / total) for key, v in exp_vals]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs


def _probability_to_decimal(prob: float, margin: float) -> float:
    """Convert fair probability to decimal odds with margin."""
    if prob <= 0:
        return 999.99
    return max(1.01, round(1.0 / (prob * (1.0 + margin)), 2))


@router.get("/championship")
async def get_championship_outrights(
    season: int = 2026,
    track_type: str = "overall",
    top_n: int = _TOP_N_CHAMPIONSHIP,
    request: Request = None,
) -> JSONResponse:
    """
    IndyCar series championship outright market.

    Win probabilities are derived from driver ELO ratings via Harville softmax.

    Args:
        season:     Championship season year.  Default: 2026.
        track_type: ELO variant — "overall", "oval", "road", "street".
                    Default: "overall" (recommended for season outrights).
        top_n:      Number of drivers in the market.  Default: 30.

    Returns:
        {
          "market": "indycar_championship",
          "season_year": 2026,
          "track_type": "overall",
          "entries": [
            {"rank": 1, "driver_id": 4931, "name": "Josef Newgarden",
             "probability": 0.18, "price": 4.72},
            ...
          ],
          "margin": 0.10
        }
    """
    top_n = max(5, min(top_n, 50))
    _VALID_TRACK_TYPES = {"overall", "oval", "road", "street"}
    if track_type not in _VALID_TRACK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"track_type must be one of {sorted(_VALID_TRACK_TYPES)}",
        )

    predictor = getattr(request.app.state, "predictor", None) if request else None
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="IndyCar predictor not loaded — models may still be initialising",
        )

    extractor = predictor.extractor
    if extractor is None:
        raise HTTPException(status_code=503, detail="Feature extractor not available")

    # Fetch ELO data by track type
    track_type_map = {"overall": None, "oval": 0, "road": 1, "street": 2}
    tt_int = track_type_map[track_type]

    try:
        if tt_int is None:
            raw: dict[int, float] = extractor.elo_overall
        else:
            raw = {
                did: elo
                for (did, tt), elo in extractor.elo_track_type.items()
                if tt == tt_int
            }
    except AttributeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ELO data unavailable: {exc}",
        ) from exc

    if not raw:
        raise HTTPException(
            status_code=503,
            detail=f"No ELO data for track_type={track_type}",
        )

    # Sort by ELO descending, take top_n
    sorted_drivers = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Resolve driver names from extractor career stats if available
    driver_names: dict[int, str] = {}
    try:
        if hasattr(extractor, "driver_career") and extractor.driver_career:
            for did, stats in extractor.driver_career.items():
                name = stats.get("driver_name") or stats.get("name")
                if name:
                    driver_names[int(did)] = name
        elif hasattr(extractor, "driver_names") and extractor.driver_names:
            driver_names = {int(k): str(v) for k, v in extractor.driver_names.items()}
    except Exception as exc:
        logger.warning("championship: driver name lookup failed: %s", exc)

    # Compute championship win probabilities
    probs = _harville_softmax_probs(sorted_drivers)

    entries: list[dict[str, Any]] = []
    for rank, (driver_id, prob) in enumerate(probs, start=1):
        name = driver_names.get(int(driver_id), f"Driver #{driver_id}")
        entries.append({
            "rank": rank,
            "driver_id": driver_id,
            "name": name,
            "probability": round(prob, 6),
            "price": _probability_to_decimal(prob, _CHAMPIONSHIP_MARGIN),
        })

    logger.info(
        "indycar_championship season=%d track_type=%s entries=%d top=%s",
        season, track_type, len(entries),
        entries[0]["name"] if entries else "none",
    )

    return JSONResponse({
        "market": "indycar_championship",
        "season_year": season,
        "track_type": track_type,
        "entries": entries,
        "margin": _CHAMPIONSHIP_MARGIN,
        "total_probability": round(sum(e["probability"] for e in entries), 6),
    })
