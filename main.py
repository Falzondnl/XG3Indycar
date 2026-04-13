"""
XG3 IndyCar Microservice — Entry Point
FastAPI application with lifespan model loading.
Port: 8033

Data: IndyCar session results (11,251 race rows, 1996-2026)
Markets: Race Winner (12% margin), Podium Top-3 (10%), H2H (5%)
Model: 3-model stacking ensemble (CatBoost + LightGBM + XGBoost)
       + BetaCalibrator + Harville podium estimation
ELO: Per-driver overall + per-track-type (oval/road/street) + team ELO
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DEBUG, PORT, SERVICE_NAME, SERVICE_VERSION, R0_DIR
from ml.predictor import IndycarPredictor
from feeds.optic_odds import OpticOddsFeed
from api.routes import health, races, markets, admin
from api.routes.outrights import router as outrights_router

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: load predictor models and initialise feeds.
    Shutdown: log teardown.
    """
    logger.info("Starting %s v%s (port=%d)", SERVICE_NAME, SERVICE_VERSION, PORT)

    # Initialise Optic Odds feed
    app.state.optic_feed = OpticOddsFeed()
    logger.info("Optic Odds feed initialised (sport_id=motorsports, league_id=indycar)")

    # Load predictor
    predictor = IndycarPredictor()
    try:
        predictor.load(R0_DIR)
        logger.info(
            "IndyCar predictor loaded: %d drivers tracked, %d teams",
            predictor.driver_count,
            predictor.team_count,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "Model files not found (%s) — predictor in unloaded state. "
            "Run: python -c \"from ml.trainer import IndycarTrainer; t=IndycarTrainer(); t.train()\"",
            exc,
        )
    except Exception as exc:
        logger.error("Predictor load failed: %s", exc, exc_info=True)

    app.state.predictor = predictor
    logger.info("%s startup complete", SERVICE_NAME)
    yield

    logger.info("%s shutting down", SERVICE_NAME)


def create_app() -> FastAPI:
    app = FastAPI(
        title="XG3 IndyCar Microservice",
        description=(
            "IndyCar race win probability prediction and market pricing. "
            "Covers all IndyCar Series events including the Indianapolis 500. "
            "3-model ensemble (CatBoost + LightGBM + XGBoost) with ELO ratings "
            "per track type (oval / road / street), Harville podium estimation, "
            "and H2H markets."
        ),
        version=SERVICE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health routes (no prefix — standard pattern for Enterprise proxy)
    app.include_router(health.router, tags=["health"])

    # Domain routes
    app.include_router(
        races.router,
        prefix="/api/v1/indycar/races",
        tags=["races"],
    )
    app.include_router(
        markets.router,
        prefix="/api/v1/indycar/races",
        tags=["markets"],
    )
    app.include_router(
        admin.router,
        prefix="/api/v1/indycar/admin",
        tags=["admin"],
    )
    app.include_router(
        outrights_router,
        prefix="/api/v1/indycar/outrights",
        tags=["outrights"],
    )

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "docs": "/docs",
            "health": "/health",
        })

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
