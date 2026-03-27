"""
IndyCar Market Builder
======================
Produces race-winner, podium Top-3, Indianapolis 500 special, and H2H markets.
All margins applied via Shin power-method normalisation.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from config import WIN_MARGIN, PODIUM_MARGIN, H2H_MARGIN

logger = logging.getLogger(__name__)

MIN_SELECTIONS = 2
MIN_PROB_CLIP = 0.001     # Floor per selection (0.1%)
MAX_DISPLAY_DRIVERS = 30  # Cap for race winner market display

# Indianapolis 500 / Indy 500 detection
_INDY500_KEYWORDS = ["indianapolis 500", "indy 500", "indy500"]


def _is_indy500(event_name: str) -> bool:
    name = event_name.lower()
    return any(kw in name for kw in _INDY500_KEYWORDS)


def _apply_margin_shin(probs: list[float], margin: float) -> list[float]:
    """
    Shin margin: scale normalised probabilities to target overround = 1 + margin.
    Preserves relative probability ratios.
    """
    if not probs:
        return []
    arr = np.array(probs, dtype=float)
    arr = np.clip(arr, MIN_PROB_CLIP, 1.0)
    s = arr.sum()
    if s < 1e-9:
        arr = np.ones(len(arr)) / len(arr)
    else:
        arr = arr / s
    target_sum = 1.0 + margin
    arr = arr * target_sum
    return arr.tolist()


def _prob_to_decimal_odds(p: float) -> float:
    """Convert margined probability to decimal odds. Minimum 1.001."""
    p = max(p, MIN_PROB_CLIP)
    return round(max(1.0 / p, 1.001), 3)


def _format_selection(name: str, prob: float, margined_prob: float) -> dict[str, Any]:
    return {
        "name": name,
        "probability": round(float(prob), 6),
        "margined_probability": round(float(margined_prob), 6),
        "decimal_odds": _prob_to_decimal_odds(margined_prob),
    }


def build_race_winner_market(
    drivers: list[dict[str, Any]],
    event_name: str = "",
    margin: float = WIN_MARGIN,
) -> dict[str, Any]:
    """
    Race winner market: one selection per driver.
    Top MAX_DISPLAY_DRIVERS drivers shown by win probability.
    """
    if len(drivers) < MIN_SELECTIONS:
        raise ValueError(f"Need at least {MIN_SELECTIONS} drivers for a market")

    sorted_drivers = sorted(drivers, key=lambda x: x["win_prob"], reverse=True)
    display_drivers = sorted_drivers[:MAX_DISPLAY_DRIVERS]

    raw_probs = [d["win_prob"] for d in display_drivers]
    margined_probs = _apply_margin_shin(raw_probs, margin)
    overround = sum(margined_probs)

    selections = []
    for drv, mp in zip(display_drivers, margined_probs):
        selections.append(
            _format_selection(
                name=f"{drv['driver_name']} ({drv['team_name']})",
                prob=drv["win_prob"],
                margined_prob=mp,
            )
        )

    market: dict[str, Any] = {
        "market_type": "race_winner",
        "event_name": event_name,
        "margin": round(margin, 4),
        "overround": round(overround, 4),
        "selection_count": len(selections),
        "total_field_size": len(drivers),
        "selections": selections,
    }

    # Special flag for Indianapolis 500
    if _is_indy500(event_name):
        market["special_event"] = "Indianapolis 500"
        market["notes"] = "Prestige oval — elevated variance; track history highly predictive"

    return market


def build_podium_market(
    drivers: list[dict[str, Any]],
    margin: float = PODIUM_MARGIN,
) -> dict[str, Any]:
    """
    Top-3 podium Yes/No binary market for top-10 most likely podium drivers.
    Harville podium_prob used from predictor.
    """
    if len(drivers) < MIN_SELECTIONS:
        raise ValueError(f"Need at least {MIN_SELECTIONS} drivers for podium market")

    sorted_drivers = sorted(drivers, key=lambda x: x["podium_prob"], reverse=True)
    top_drivers = sorted_drivers[:10]

    selections = []
    for drv in top_drivers:
        p_yes = float(drv["podium_prob"])
        p_yes = max(p_yes, MIN_PROB_CLIP)
        p_no = max(1.0 - p_yes, MIN_PROB_CLIP)

        yes_mp, no_mp = _apply_margin_shin([p_yes, p_no], margin)

        selections.append({
            "driver_name": drv["driver_name"],
            "team_name": drv["team_name"],
            "podium_yes": {
                "probability": round(p_yes, 6),
                "margined_probability": round(yes_mp, 6),
                "decimal_odds": _prob_to_decimal_odds(yes_mp),
            },
            "podium_no": {
                "probability": round(p_no, 6),
                "margined_probability": round(no_mp, 6),
                "decimal_odds": _prob_to_decimal_odds(no_mp),
            },
        })

    return {
        "market_type": "podium_finisher",
        "margin": round(margin, 4),
        "selections": selections,
        "description": "Will this driver finish in the top 3?",
    }


def build_h2h_markets(
    drivers: list[dict[str, Any]],
    margin: float = H2H_MARGIN,
    top_n_drivers: int = 8,
) -> list[dict[str, Any]]:
    """
    Head-to-head markets for top-N drivers by win probability.
    Produces all pairwise H2H matchups among the top-N.
    P(A beats B) = win_prob_A / (win_prob_A + win_prob_B)  [Harville-style]
    """
    if len(drivers) < 2:
        return []

    sorted_drivers = sorted(drivers, key=lambda x: x["win_prob"], reverse=True)
    top_drivers = sorted_drivers[:top_n_drivers]

    markets = []
    for i in range(len(top_drivers)):
        for j in range(i + 1, len(top_drivers)):
            d_a = top_drivers[i]
            d_b = top_drivers[j]

            p_a = float(d_a["win_prob"])
            p_b = float(d_b["win_prob"])
            total = p_a + p_b
            if total < 1e-9:
                continue

            p_a_h2h = p_a / total
            p_b_h2h = p_b / total

            a_mp, b_mp = _apply_margin_shin([p_a_h2h, p_b_h2h], margin)

            markets.append({
                "market_type": "h2h",
                "driver_a": {
                    "name": d_a["driver_name"],
                    "team": d_a["team_name"],
                    "win_prob": round(p_a_h2h, 6),
                    "margined_probability": round(a_mp, 6),
                    "decimal_odds": _prob_to_decimal_odds(a_mp),
                },
                "driver_b": {
                    "name": d_b["driver_name"],
                    "team": d_b["team_name"],
                    "win_prob": round(p_b_h2h, 6),
                    "margined_probability": round(b_mp, 6),
                    "decimal_odds": _prob_to_decimal_odds(b_mp),
                },
                "margin": round(margin, 4),
            })

    return markets


def build_all_markets(
    drivers: list[dict[str, Any]],
    event_name: str = "",
    win_margin: float = WIN_MARGIN,
    podium_margin: float = PODIUM_MARGIN,
    h2h_margin: float = H2H_MARGIN,
) -> dict[str, Any]:
    """
    Build all available markets for a race.
    Returns {race_winner, podium_finisher, h2h, [indy500 special if applicable]}.
    """
    if not drivers:
        raise ValueError("drivers list cannot be empty")

    markets: dict[str, Any] = {}
    errors: dict[str, str] = {}

    try:
        markets["race_winner"] = build_race_winner_market(drivers, event_name, win_margin)
    except Exception as exc:
        logger.error("Failed to build race_winner market: %s", exc)
        errors["race_winner"] = str(exc)

    try:
        markets["podium_finisher"] = build_podium_market(drivers, podium_margin)
    except Exception as exc:
        logger.error("Failed to build podium market: %s", exc)
        errors["podium_finisher"] = str(exc)

    try:
        h2h_list = build_h2h_markets(drivers, h2h_margin)
        markets["h2h"] = {
            "market_type": "h2h_collection",
            "matchup_count": len(h2h_list),
            "matchups": h2h_list,
        }
    except Exception as exc:
        logger.error("Failed to build H2H markets: %s", exc)
        errors["h2h"] = str(exc)

    result: dict[str, Any] = {
        "event_name": event_name,
        "field_size": len(drivers),
        "is_indy500": _is_indy500(event_name),
        "markets": markets,
    }
    if errors:
        result["errors"] = errors

    return result
