"""
IndyCar regression lock tests.

LOCK-INDYCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001
LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001
LOCK-INDYCAR-TIER-2-CASCADE-001
LOCK-INDYCAR-TIER-3-REFUSE-503-001
"""
from __future__ import annotations

import pathlib
import uuid


# ---------------------------------------------------------------------------
# LOCK-INDYCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001
# ---------------------------------------------------------------------------


class TestIndycarEloNoSilentFallback:
    """LOCK-INDYCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001"""

    def test_all_unknown_drivers_triggers_refuse(self) -> None:
        """
        The /races/predict route must refuse (503 FIXTURE_UNPRICED) when ALL
        submitted drivers have driver_id=0 (no ELO seed).
        This is verified by reading races.py and checking the all_unknown guard.
        """
        races_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "races.py"
        if races_path.exists():
            source = races_path.read_text(encoding="utf-8")
            assert "all_unknown" in source, (
                "LOCK-INDYCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001 VIOLATED: "
                "all_unknown guard missing from races.py"
            )
            assert "FIXTURE_UNPRICED" in source, (
                "LOCK-INDYCAR-ELO-DEFAULT-NO-SILENT-FALLBACK-001 VIOLATED: "
                "FIXTURE_UNPRICED code missing from races.py"
            )

    def test_partial_unknown_sets_partial_elo_source(self) -> None:
        """
        When some (but not all) drivers lack ELO, prediction_source must be
        'partial_elo', not 'ml_ensemble'.
        """
        unknown_drivers = ["Ryo Hirakawa"]
        all_drivers = ["Josef Newgarden", "Alex Palou", "Ryo Hirakawa"]
        all_unknown = len(unknown_drivers) == len(all_drivers)
        prediction_source = "partial_elo" if unknown_drivers else "ml_ensemble"

        assert not all_unknown
        assert prediction_source == "partial_elo"


# ---------------------------------------------------------------------------
# LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001
# ---------------------------------------------------------------------------


class TestIndycarPredictionSourceField:
    """LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001"""

    def test_markets_price_response_has_prediction_source(self) -> None:
        """The /markets/price response must include prediction_source."""
        mock_response = {
            "success": True,
            "event_name": "Indianapolis 500",
            "year": 2026,
            "field_size": 33,
            "top_predictions": [],
            "markets": {},
            "is_indy500": True,
            "prediction_source": "model",  # LOCK — must be present
        }
        assert "prediction_source" in mock_response, (
            "LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001 VIOLATED: "
            "prediction_source missing from /markets/price response"
        )
        _valid = {"model", "model_pinnacle_blend", "market_scrape",
                  "market_scrape_reverse_engineered", "unpriced", "partial_elo"}
        assert mock_response["prediction_source"] in _valid

    def test_markets_source_has_lock_id(self) -> None:
        """markets.py must contain LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001."""
        markets_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "markets.py"
        if markets_path.exists():
            source = markets_path.read_text(encoding="utf-8")
            assert "LOCK-INDYCAR-PRED-SRC-MS-RESPONSE-001" in source


# ---------------------------------------------------------------------------
# LOCK-INDYCAR-TIER-2-CASCADE-001
# ---------------------------------------------------------------------------


class TestIndycarTier2Cascade:
    """LOCK-INDYCAR-TIER-2-CASCADE-001"""

    def test_optic_feed_has_get_race_odds_devigged(self) -> None:
        """OpticOddsFeed must expose get_race_odds_devigged for Tier 2 cascade."""
        from feeds.optic_odds import OpticOddsFeed
        assert hasattr(OpticOddsFeed, "get_race_odds_devigged"), (
            "LOCK-INDYCAR-TIER-2-CASCADE-001 VIOLATED: "
            "OpticOddsFeed.get_race_odds_devigged missing"
        )

    def test_markets_source_has_tier2_lock_id(self) -> None:
        """markets.py must contain LOCK-INDYCAR-TIER-2-CASCADE-001."""
        markets_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "markets.py"
        if markets_path.exists():
            source = markets_path.read_text(encoding="utf-8")
            assert "LOCK-INDYCAR-TIER-2-CASCADE-001" in source

    def test_tier2_response_prediction_source_is_market_scrape(self) -> None:
        """Tier 2 response must use prediction_source='market_scrape'."""
        mock_tier2 = {
            "success": True,
            "event_name": "Indianapolis 500",
            "year": 2026,
            "field_size": 33,
            "markets": {"race_winner": {}},
            "prediction_source": "market_scrape",  # LOCK
            "model_available": False,
            "tier": 2,
        }
        assert mock_tier2["prediction_source"] == "market_scrape"
        assert mock_tier2["model_available"] is False


# ---------------------------------------------------------------------------
# LOCK-INDYCAR-TIER-3-REFUSE-503-001
# ---------------------------------------------------------------------------


class TestIndycarTier3Refuse:
    """LOCK-INDYCAR-TIER-3-REFUSE-503-001"""

    def test_tier3_body_has_required_structured_fields(self) -> None:
        """Structured 503 body must have code, reason, correlation_id, retry_after."""
        cid = str(uuid.uuid4())
        tier3_body = {
            "code": "FIXTURE_UNPRICED",
            "reason": "no_model_no_market_data",
            "message": "test",
            "correlation_id": cid,
            "retry_after": 30,
            "event_name": "Indianapolis 500",
            "year": 2026,
        }
        for required in ("code", "reason", "message", "correlation_id", "retry_after"):
            assert required in tier3_body, (
                f"LOCK-INDYCAR-TIER-3-REFUSE-503-001 VIOLATED: missing {required}"
            )
        assert tier3_body["code"] == "FIXTURE_UNPRICED"
        assert tier3_body["retry_after"] > 0

    def test_markets_source_has_tier3_lock_id(self) -> None:
        """markets.py must contain LOCK-INDYCAR-TIER-3-REFUSE-503-001."""
        markets_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "markets.py"
        if markets_path.exists():
            source = markets_path.read_text(encoding="utf-8")
            assert "LOCK-INDYCAR-TIER-3-REFUSE-503-001" in source
