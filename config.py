"""
XG3 IndyCar Microservice — Configuration
IndyCar race win probability prediction and pricing.
"""
from __future__ import annotations

import os

SERVICE_NAME = "xg3-indycar"
SERVICE_VERSION = "1.0.0"
PORT = int(os.getenv("PORT", "8033"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Data paths
INDYCAR_CSV = os.getenv(
    "INDYCAR_CSV",
    "D:/codex/Data/motorsports/tier1/curated/indycar/session_details_flat.csv",
)
DRIVER_YEAR_CSV = os.getenv(
    "DRIVER_YEAR_CSV",
    "D:/codex/Data/motorsports/tier1/curated/indycar/driver_year_details_flat.csv",
)
YEAR_POINTS_CSV = os.getenv(
    "YEAR_POINTS_CSV",
    "D:/codex/Data/motorsports/tier1/curated/indycar/year_point_breakdown.csv",
)

# Model directories
MODEL_DIR = os.getenv("MODEL_DIR", "models")
R0_DIR = f"{MODEL_DIR}/r0"

# Pricing margins
WIN_MARGIN = float(os.getenv("WIN_MARGIN", "0.12"))       # Race winner = 12%
PODIUM_MARGIN = float(os.getenv("PODIUM_MARGIN", "0.10"))  # Top 3 podium = 10%
H2H_MARGIN = float(os.getenv("H2H_MARGIN", "0.05"))        # H2H = 5%

# Optic Odds
OPTIC_ODDS_API_KEY = os.getenv("OPTIC_ODDS_API_KEY", "")
OPTIC_ODDS_BASE_URL = "https://api.opticodds.com/api/v3"
OPTIC_ODDS_SPORT_ID = "motorsports"
OPTIC_ODDS_LEAGUE_ID = "indycar"

# ELO parameters
ELO_K = int(os.getenv("ELO_K", "32"))
ELO_DEFAULT = 1500.0
HARVILLE_TOP_N = 30

# Feature settings
MIN_RACE_SIZE = 5        # Minimum finishers to include a race in training
MAX_DISPLAY_DRIVERS = 30  # Cap race winner market display

# Temporal split years
TRAIN_YEARS_MAX = 2015
VAL_YEARS_MIN = 2016
VAL_YEARS_MAX = 2019
TEST_YEARS_MIN = 2020
