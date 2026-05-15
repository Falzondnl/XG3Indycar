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

        url = f"{self._base_url}/fixtures/active"
        params = {
            "sport": self._sport_id,
            "league": self._league_id,
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

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def get_race_odds_devigged(
        self,
        event_name: str,
        year: int,
        bookmaker: str = "pinnacle",
    ) -> "dict[str, Any] | None":
        """
        LOCK-INDYCAR-TIER-2-CASCADE-001: Tier 2 fallback pricing.

        1. Discover the fixture ID by searching /fixtures/active for event_name.
        2. Fetch Pinnacle outright winner odds via Optic Odds.
        3. Devig via ratio method → fair implied probabilities.
        4. Return structured market dict.

        Returns None when fixture not found, odds unavailable, or < 2 runners.
        Never raises — callers treat None as Tier 2 unavailable → Tier 3 refuse.
        """
        if not self._api_key:
            logger.debug(
                "indycar_tier2_skipped event=%s no_api_key "
                "LOCK-INDYCAR-TIER-2-CASCADE-001",
                event_name,
            )
            return None

        # Step 1: Discover fixture by name match.
        fixture_id: "str | None" = None
        try:
            url = f"{self._base_url}/fixtures/active"
            params = {
                "sport": self._sport_id,
                "league": self._league_id,
                "search": event_name,
            }
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
            for fixture in data.get("data", []):
                fname = (fixture.get("name") or "").lower()
                if event_name.lower() in fname:
                    fixture_id = fixture.get("id")
                    break
        except Exception as exc:
            logger.warning(
                "indycar_tier2_discovery_failed event=%s error=%s "
                "LOCK-INDYCAR-TIER-2-CASCADE-001",
                event_name, exc,
            )
            return None

        if not fixture_id:
            logger.info(
                "indycar_tier2_fixture_not_found event=%s year=%d "
                "LOCK-INDYCAR-TIER-2-CASCADE-001",
                event_name, year,
            )
            return None

        # Step 2: Fetch Pinnacle outright winner odds.
        try:
            url = f"{self._base_url}/fixtures/odds"
            params = {
                "fixture_id": fixture_id,
                "market": "outright_winner",
                "sportsbook": bookmaker,
            }
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=self._headers())
                resp.raise_for_status()
                odds_data = resp.json()
        except Exception as exc:
            logger.warning(
                "indycar_tier2_odds_fetch_failed fixture_id=%s error=%s "
                "LOCK-INDYCAR-TIER-2-CASCADE-001",
                fixture_id, exc,
            )
            return None

        # Step 3: Parse runners and devig.
        raw_entries: "list[tuple[str, float]]" = []
        for entry in odds_data.get("data", []):
            for odd in entry.get("odds", []):
                market_id = odd.get("market_id", "")
                if market_id not in ("outright_winner", "winner", "race_winner"):
                    continue
                american = odd.get("price")
                name = odd.get("name") or odd.get("participant_name")
                if not name or american is None:
                    continue
                try:
                    a = float(american)
                    dec = (1 + a / 100) if a > 0 else (1 + 100 / abs(a))
                    if dec > 1.0:
                        raw_entries.append((str(name), 1.0 / dec))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue

        if len(raw_entries) < 2:
            logger.info(
                "indycar_tier2_insufficient_runners fixture_id=%s runners=%d "
                "LOCK-INDYCAR-TIER-2-CASCADE-001",
                fixture_id, len(raw_entries),
            )
            return None

        total_implied = sum(p for _, p in raw_entries)
        if total_implied < 1e-9:
            return None

        fair_probs = [(name, p / total_implied) for name, p in raw_entries]
        fair_probs.sort(key=lambda x: x[1], reverse=True)

        selections = [
            {
                "driver_name": name,
                "fair_prob": round(fair_p, 6),
                "decimal_odds": round(1.0 / fair_p, 3) if fair_p > 0 else 999.99,
                "source": bookmaker,
            }
            for name, fair_p in fair_probs
        ]

        logger.info(
            "indycar_tier2_market_scrape_ok fixture_id=%s runners=%d "
            "bookmaker=%s overround=%.4f LOCK-INDYCAR-TIER-2-CASCADE-001",
            fixture_id, len(selections), bookmaker, total_implied,
        )

        return {
            "race_winner": {
                "market_type": "race_winner",
                "source": bookmaker,
                "overround": round(total_implied, 4),
                "selections": selections,
            }
        }

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
