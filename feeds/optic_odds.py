"""
Optic Odds feed integration for IndyCar.
sport_id: "motorsports"
league_id: "indycar"
Fetches upcoming races and competitor lists.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import OPTIC_ODDS_API_KEY, OPTIC_ODDS_BASE_URL, OPTIC_ODDS_SPORT_ID, OPTIC_ODDS_LEAGUE_ID

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
_UPCOMING_LIMIT = 50


class OpticOddsFeed:
    """
    Async client for Optic Odds API v3.
    Fetches IndyCar fixtures and available markets.
    """

    def __init__(self) -> None:
        self._api_key = OPTIC_ODDS_API_KEY
        self._base_url = OPTIC_ODDS_BASE_URL
        self._sport_id = OPTIC_ODDS_SPORT_ID
        self._league_id = OPTIC_ODDS_LEAGUE_ID

    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self._api_key}

    async def get_upcoming_races(self, limit: int = _UPCOMING_LIMIT) -> list[dict[str, Any]]:
        """
        Fetch upcoming IndyCar races from Optic Odds.
        Returns list of race dicts.
        Raises RuntimeError if API key not configured.
        """
        if not self._api_key:
            raise RuntimeError(
                "OPTIC_ODDS_API_KEY not configured — cannot fetch upcoming races"
            )

        url = f"{self._base_url}/fixtures"
        params = {
            "sport_id": self._sport_id,
            "league_id": self._league_id,
            "status": "upcoming",
            "limit": limit,
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        fixtures = data.get("data", [])
        logger.info("Fetched %d upcoming IndyCar fixtures from Optic Odds", len(fixtures))
        return self._normalise_fixtures(fixtures)

    async def get_race_competitors(self, fixture_id: str) -> list[dict[str, Any]]:
        """
        Fetch competitor list for a specific IndyCar fixture.
        Returns list of {driver_id, driver_name, team_name} dicts.
        """
        if not self._api_key:
            raise RuntimeError("OPTIC_ODDS_API_KEY not configured")

        url = f"{self._base_url}/fixtures/{fixture_id}/participants"
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        participants = data.get("data", [])
        return [
            {
                "driver_id": p.get("id", 0),
                "driver_name": p.get("name", "Unknown"),
                "team_name": p.get("team", {}).get("name", "Unknown") if isinstance(p.get("team"), dict) else "Unknown",
            }
            for p in participants
        ]

    async def get_available_odds(self, fixture_id: str) -> dict[str, Any]:
        """Fetch available market odds for a fixture."""
        if not self._api_key:
            raise RuntimeError("OPTIC_ODDS_API_KEY not configured")

        url = f"{self._base_url}/odds"
        params = {"fixture_id": fixture_id}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return data.get("data", {})

    def _normalise_fixtures(self, fixtures: list[dict]) -> list[dict[str, Any]]:
        result = []
        for f in fixtures:
            league = f.get("league", {})
            result.append({
                "fixture_id": f.get("id", ""),
                "race_name": f.get("name", ""),
                "league": league.get("name", "") if isinstance(league, dict) else str(league),
                "start_date": f.get("start_date", ""),
                "status": f.get("status", "upcoming"),
                "sport_id": self._sport_id,
                "league_id": self._league_id,
            })
        return result
