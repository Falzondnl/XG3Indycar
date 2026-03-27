"""
Admin endpoints for IndyCar MS.
GET  /api/v1/indycar/admin/status        — model status, ELO counts, pkl files
GET  /api/v1/indycar/admin/elo-ratings   — top ELO rated drivers
POST /api/v1/indycar/admin/train         — trigger model training
"""
from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from config import MODEL_DIR, R0_DIR, SERVICE_NAME, SERVICE_VERSION

router = APIRouter()
_START_TIME = time.time()

# Training lock — prevent concurrent training runs
_training_running = False


def _list_pkl_files(directory: str) -> list[str]:
    """List .pkl files in directory, return [] if not found."""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".pkl")]


@router.get("/status")
async def admin_status(request: Request) -> JSONResponse:
    """Returns current service status including model artefacts and ELO counts."""
    predictor = getattr(request.app.state, "predictor", None)
    optic_feed = getattr(request.app.state, "optic_feed", None)

    driver_count = 0
    team_count = 0
    track_elo_count = 0
    history_count = 0
    model_loaded = False

    if predictor is not None and predictor.is_loaded:
        model_loaded = True
        driver_count = predictor.driver_count
        team_count = predictor.team_count
        if predictor.extractor is not None:
            track_elo_count = len(predictor.extractor.elo_track_type)
            history_count = sum(
                len(v) for v in predictor.extractor.driver_history.values()
            )

    r0_files = _list_pkl_files(R0_DIR)

    return JSONResponse({
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "model_loaded": model_loaded,
        "driver_count": driver_count,
        "team_count": team_count,
        "track_elo_count": track_elo_count,
        "history_record_count": history_count,
        "optic_feed_active": optic_feed is not None,
        "model_files": {
            "r0": r0_files,
        },
        "r0_dir": R0_DIR,
    })


@router.get("/elo-ratings")
async def get_elo_ratings(
    request: Request,
    top_n: int = 30,
    track_type: str = "overall",
) -> JSONResponse:
    """
    Return top-N drivers by ELO rating.
    track_type: 'overall', 'oval', 'road', 'street'
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Predictor not loaded"},
        )

    extractor = predictor.extractor
    track_type_map = {"overall": None, "oval": 0, "road": 1, "street": 2}

    if track_type not in track_type_map:
        return JSONResponse(
            status_code=422,
            content={"error": f"Unknown track_type '{track_type}'. Use: overall, oval, road, street"},
        )

    tt_int = track_type_map[track_type]

    if tt_int is None:
        # Overall ELO
        ratings = [
            {"driver_id": did, "elo": round(elo, 2)}
            for did, elo in extractor.elo_overall.items()
        ]
    else:
        # Track-type ELO
        ratings = [
            {"driver_id": did, "track_type": track_type, "elo": round(elo, 2)}
            for (did, tt), elo in extractor.elo_track_type.items()
            if tt == tt_int
        ]

    ratings.sort(key=lambda x: x["elo"], reverse=True)
    ratings = ratings[:top_n]

    return JSONResponse({
        "track_type": track_type,
        "top_n": top_n,
        "count": len(ratings),
        "ratings": ratings,
    })


def _run_training_background() -> None:
    """Background training task."""
    global _training_running
    try:
        from ml.trainer import IndycarTrainer
        trainer = IndycarTrainer()
        metrics = trainer.train()
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Background training complete: %s", metrics)
    except Exception as exc:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Background training failed: %s", exc, exc_info=True)
    finally:
        _training_running = False


@router.post("/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Trigger model training in the background.
    Returns immediately — training runs asynchronously.
    Check /admin/status for model_loaded=true when complete.
    """
    global _training_running
    if _training_running:
        return JSONResponse(
            status_code=409,
            content={"status": "already_running", "message": "Training is already in progress"},
        )

    _training_running = True
    background_tasks.add_task(_run_training_background)

    return JSONResponse({
        "status": "started",
        "message": "Training started in background. Poll /admin/status to check completion.",
    })
