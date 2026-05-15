"""
LOCK-INDYCAR-TIER-2B-REVERSE-ENGINEER-001
===========================================

Regression tests for IndyCar Tier 2B PL reverse-engineer.
"""
from __future__ import annotations

import numpy as np
import pytest

from pricing.tier2b_reverse_engineer import (
    IndyCarTier2BReverseEngineer,
    devig_outright_market,
    plackett_luce_inverse,
    _gumbel_max_top_k,
    get_tier2b_engineer,
)


def test_pl_inverse_round_trip():
    fair = {"palou": 0.30, "newgarden": 0.25, "oward": 0.20, "dixon": 0.15, "rossi": 0.10}
    ids, skills, recon = plackett_luce_inverse(fair)
    for did, p in zip(ids, recon):
        assert abs(p - fair[did]) < 1e-12


def test_top_k_sums_to_k():
    skills = np.array([7.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5], dtype=np.float64)
    for k in [1, 3, 5]:
        marg = _gumbel_max_top_k(skills, k)
        assert abs(marg.sum() - k) < 1e-9


def test_engineer_indy500():
    odds = {
        "palou": 4.0, "newgarden": 5.5, "oward": 7.0, "ericsson": 9.0,
        "dixon": 11.0, "power": 12.0, "rossi": 17.0, "herta": 19.0,
        "mclaughlin": 21.0, "lundqvist": 26.0, "kirkwood": 31.0,
    }
    eng = IndyCarTier2BReverseEngineer()
    r = eng.reverse_engineer("indy500_2026", odds, track_type="oval")
    assert r.solver_converged
    assert r.prediction_source == "market_scrape_reverse_engineered"
    assert r.track_type == "oval"
    assert abs(r.podium_probs().sum() - 3.0) < 1e-9
    assert abs(r.top_5_probs().sum() - 5.0) < 1e-9


def test_thin_field_raises():
    eng = IndyCarTier2BReverseEngineer()
    with pytest.raises(ValueError):
        eng.reverse_engineer("thin", {"a": 2.0, "b": 3.0})


def test_singleton():
    assert get_tier2b_engineer() is get_tier2b_engineer()
