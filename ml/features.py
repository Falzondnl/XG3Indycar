"""
IndyCar Feature Extractor
=========================
Processes race results chronologically (1996-2026).
All features use strict temporal integrity: features at time T use ONLY results before T.

Features extracted per driver per race (16 total):
  elo_overall          — global ELO across all IndyCar races (K=32)
  elo_track_type       — ELO per track type: oval / road / street
  team_elo             — team average ELO (rolling)
  rolling_win_rate_3   — win rate (last 3 races overall)
  rolling_win_rate_5   — win rate (last 5 races overall)
  rolling_win_rate_10  — win rate (last 10 races overall)
  rolling_avg_finish_5 — avg finish position (last 5 races overall)
  rolling_laps_led_pct_5 — avg fraction of laps led (last 5 races overall)
  rolling_avg_speed_5  — avg speed (last 5 races overall)
  rolling_avg_start_5  — avg starting position (last 5 races overall)
  career_wins          — total career wins before this race
  career_races         — total career race starts before this race
  venue_avg_finish     — avg finish at this event_name historically
  venue_race_count     — number of times raced at this venue before
  year                 — season year (captures era changes)
  track_type_encoded   — 0=oval, 1=road, 2=street
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from config import ELO_DEFAULT, ELO_K, INDYCAR_CSV, MIN_RACE_SIZE

logger = logging.getLogger(__name__)


# Module-level picklable defaultdict factories (lambdas cannot be pickled)
def _elo_default() -> float:
    return ELO_DEFAULT


def _list_default() -> list:
    return []


def _career_default() -> list:
    return [0, 0]  # [wins, total_races]


FEATURES = [
    "elo_overall",
    "elo_track_type",
    "team_elo",
    "rolling_win_rate_3",
    "rolling_win_rate_5",
    "rolling_win_rate_10",
    "rolling_avg_finish_5",
    "rolling_laps_led_pct_5",
    "rolling_avg_speed_5",
    "rolling_avg_start_5",
    "career_wins",
    "career_races",
    "venue_avg_finish",
    "venue_race_count",
    "year",
    "track_type_encoded",
]

# Track type mapping based on event_name keywords
_OVAL_KEYWORDS = [
    "indianapolis", "500", "iowa", "texas", "pocono", "michigan",
    "richmond", "homestead", "chicagoland", "gateway", "oval",
    "auto club", "kentucky", "fontana", "phoenix",
]
_ROAD_KEYWORDS = [
    "road", "circuit", "gp", "grand prix", "midohio", "mid-ohio",
    "laguna", "watkins", "sonoma", "elkhart", "barber", "portland",
    "the glen", "glen", "rahal", "raceway park",
]
# Anything not matched → street circuit

TRACK_TYPE_OVAL = 0
TRACK_TYPE_ROAD = 1
TRACK_TYPE_STREET = 2


def classify_track_type(event_name: str) -> int:
    """Classify event track type: 0=oval, 1=road, 2=street."""
    name = event_name.lower()
    for kw in _OVAL_KEYWORDS:
        if kw in name:
            return TRACK_TYPE_OVAL
    for kw in _ROAD_KEYWORDS:
        if kw in name:
            return TRACK_TYPE_ROAD
    return TRACK_TYPE_STREET


def _elo_update(
    rating_a: float, rating_b: float, score_a: float, k: float
) -> tuple[float, float]:
    """Standard ELO update. score_a=1.0 means A beat B."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    delta = k * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


class IndycarFeatureExtractor:
    """
    Stateful feature extractor for IndyCar races.
    Call build_dataset() once for training.
    Call warm() at service startup to rebuild state to current day.
    Use get_features_for_race() for live inference.
    """

    def __init__(self) -> None:
        # Overall ELO per driver_id
        self.elo_overall: dict[int, float] = defaultdict(_elo_default)
        # Per-track-type ELO: (driver_id, track_type_int) -> float
        self.elo_track_type: dict[tuple[int, int], float] = defaultdict(_elo_default)
        # Team ELO: team_name -> float
        self.team_elo: dict[str, float] = defaultdict(_elo_default)
        # Per-driver race history: driver_id -> list of result dicts
        # Each dict: {year, finish, start, laps_led, laps_total, speed_avg, event_name, track_type}
        self.driver_history: dict[int, list[dict[str, Any]]] = defaultdict(_list_default)
        # Career stats: driver_id -> [wins, total_races]
        self.career_stats: dict[int, list[int]] = defaultdict(_career_default)
        # Venue history: (driver_id, event_name_norm) -> list of finish positions
        self.venue_history: dict[tuple[int, str], list[int]] = defaultdict(_list_default)
        # Fitted flag
        self._fitted = False

    # ------------------------------------------------------------------
    # DATASET CONSTRUCTION (training only)
    # ------------------------------------------------------------------

    def build_dataset(self, csv_path: str = INDYCAR_CSV) -> pd.DataFrame:
        """
        Read IndyCar CSV, iterate chronologically, extract pre-race features.
        Returns DataFrame with FEATURES + ['race_id', 'year', 'target'].
        Strict temporal integrity: features at time T use ONLY results before T.
        """
        logger.info("Loading IndyCar CSV from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d total rows", len(df))

        # Filter to race sessions only
        df = df[df["session_type"] == "R"].copy()
        logger.info("Race rows: %d", len(df))

        # Extract year from session_date (session_year is always NaN)
        df["date_parsed"] = pd.to_datetime(
            df["session_date"], format="%A, %B %d, %Y", errors="coerce"
        )
        df["year"] = df["date_parsed"].dt.year
        df = df.dropna(subset=["date_parsed", "year"]).copy()
        df["year"] = df["year"].astype(int)

        # Normalise key columns
        df["DriversID"] = pd.to_numeric(df["DriversID"], errors="coerce").fillna(0).astype(int)
        df["PositionFinish"] = pd.to_numeric(df["PositionFinish"], errors="coerce")
        df["PositionStart"] = pd.to_numeric(df["PositionStart"], errors="coerce")
        df["LapsLed"] = pd.to_numeric(df["LapsLed"], errors="coerce").fillna(0)
        df["LapsComplete"] = pd.to_numeric(df["LapsComplete"], errors="coerce").fillna(0)
        df["SpeedAvg"] = pd.to_numeric(df["SpeedAvg"], errors="coerce").fillna(0.0)

        # Drop rows without a valid finish position or driver
        df = df.dropna(subset=["PositionFinish"]).copy()
        df = df[df["DriversID"] > 0].copy()
        df["PositionFinish"] = df["PositionFinish"].astype(int)

        # Track type per event
        df["track_type"] = df["event_name"].fillna("").apply(classify_track_type)

        # Laps led fraction per driver per race
        # We compute this after grouping, so keep raw columns for now

        # Normalise event name for venue history
        df["event_norm"] = df["event_name"].fillna("").str.lower().str.strip()

        # Build a unique race_id from events_session_id
        df["race_id"] = df["events_session_id"].astype(str)

        # Sort chronologically
        df = df.sort_values(["date_parsed", "race_id", "PositionFinish"]).reset_index(drop=True)

        # Group by race for feature extraction
        race_order = (
            df.groupby("race_id", sort=False)["date_parsed"]
            .first()
            .sort_values()
        )
        race_df_map = {rid: grp for rid, grp in df.groupby("race_id")}

        logger.info("Building features over %d races ...", len(race_order))
        records: list[dict[str, Any]] = []

        for race_id, race_date in race_order.items():
            grp = race_df_map[race_id].copy()

            # Skip small fields
            if len(grp) < MIN_RACE_SIZE:
                continue

            year = int(grp["year"].iloc[0])
            track_type = int(grp["track_type"].iloc[0])
            event_norm = str(grp["event_norm"].iloc[0])

            # Compute total laps for this race (max laps completed by winner)
            winner_laps = grp.loc[grp["PositionFinish"] == 1, "LapsComplete"]
            total_laps = float(winner_laps.iloc[0]) if len(winner_laps) > 0 else float(grp["LapsComplete"].max())
            total_laps = max(total_laps, 1.0)

            # --- Extract PRE-RACE features for each driver ---
            race_records: list[dict[str, Any]] = []
            for _, row in grp.iterrows():
                driver_id = int(row["DriversID"])
                team_name = str(row["TeamName"]) if pd.notna(row.get("TeamName")) else "Unknown"

                feats = self._extract_features(
                    driver_id=driver_id,
                    team_name=team_name,
                    track_type=track_type,
                    event_norm=event_norm,
                    year=year,
                )
                feats["race_id"] = race_id
                feats["year"] = year
                feats["target"] = 1 if int(row["PositionFinish"]) == 1 else 0
                race_records.append(feats)

            records.extend(race_records)

            # --- UPDATE state AFTER extracting features for this race ---
            self._update_state_from_race(grp, track_type, event_norm, total_laps)

        dataset = pd.DataFrame(records)
        if not dataset.empty:
            logger.info(
                "Dataset built: %d rows, %d races, target_rate=%.4f",
                len(dataset),
                dataset["race_id"].nunique(),
                dataset["target"].mean(),
            )
        self._fitted = True
        return dataset

    def _extract_features(
        self,
        driver_id: int,
        team_name: str,
        track_type: int,
        event_norm: str,
        year: int,
    ) -> dict[str, Any]:
        """Extract pre-race features for a single driver. Reads from current state only."""
        hist = self.driver_history[driver_id]
        career = self.career_stats[driver_id]  # [wins, total_races]

        # ELO ratings (before this race)
        elo_ov = float(self.elo_overall[driver_id])
        elo_tt = float(self.elo_track_type[(driver_id, track_type)])
        t_elo = float(self.team_elo[team_name])

        # Rolling stats from full race history
        finishes = [h["finish"] for h in hist]
        starts = [h["start"] for h in hist if h["start"] > 0]
        speeds = [h["speed_avg"] for h in hist if h["speed_avg"] > 0]
        laps_led_pcts = [h["laps_led_pct"] for h in hist]
        wins = [1 if h["finish"] == 1 else 0 for h in hist]

        rolling_win_3 = float(np.mean(wins[-3:])) if wins else 0.0
        rolling_win_5 = float(np.mean(wins[-5:])) if wins else 0.0
        rolling_win_10 = float(np.mean(wins[-10:])) if wins else 0.0
        rolling_avg_finish_5 = float(np.mean(finishes[-5:])) if finishes else 20.0
        rolling_laps_led_pct_5 = float(np.mean(laps_led_pcts[-5:])) if laps_led_pcts else 0.0
        rolling_avg_speed_5 = float(np.mean(speeds[-5:])) if speeds else 0.0
        rolling_avg_start_5 = float(np.mean(starts[-5:])) if starts else 10.0

        career_wins = float(career[0])
        career_races = float(career[1])

        # Venue history
        venue_key = (driver_id, event_norm)
        venue_finishes = self.venue_history[venue_key]
        venue_avg_finish = float(np.mean(venue_finishes)) if venue_finishes else 15.0
        venue_race_count = float(len(venue_finishes))

        return {
            "elo_overall": elo_ov,
            "elo_track_type": elo_tt,
            "team_elo": t_elo,
            "rolling_win_rate_3": rolling_win_3,
            "rolling_win_rate_5": rolling_win_5,
            "rolling_win_rate_10": rolling_win_10,
            "rolling_avg_finish_5": rolling_avg_finish_5,
            "rolling_laps_led_pct_5": rolling_laps_led_pct_5,
            "rolling_avg_speed_5": rolling_avg_speed_5,
            "rolling_avg_start_5": rolling_avg_start_5,
            "career_wins": career_wins,
            "career_races": career_races,
            "venue_avg_finish": venue_avg_finish,
            "venue_race_count": venue_race_count,
            "year": float(year),
            "track_type_encoded": float(track_type),
        }

    def _update_state_from_race(
        self,
        grp: pd.DataFrame,
        track_type: int,
        event_norm: str,
        total_laps: float,
    ) -> None:
        """Update ELO and history AFTER extracting features for this race."""
        # Sort by finish position for pairwise ELO
        ranked = grp.sort_values("PositionFinish").reset_index(drop=True)
        n = len(ranked)

        if n < 2:
            return

        # Collect driver ids and teams
        driver_ids = [int(r["DriversID"]) for _, r in ranked.iterrows()]
        team_names = [
            str(r["TeamName"]) if pd.notna(r.get("TeamName")) else "Unknown"
            for _, r in ranked.iterrows()
        ]

        # Pairwise ELO: every pair where i finished ahead of j
        k_scaled = ELO_K / max(n, 1)  # scale K by field size
        for i in range(n):
            for j in range(i + 1, n):
                a_id, b_id = driver_ids[i], driver_ids[j]

                # Overall ELO
                new_a, new_b = _elo_update(
                    self.elo_overall[a_id], self.elo_overall[b_id], 1.0, k_scaled
                )
                self.elo_overall[a_id] = new_a
                self.elo_overall[b_id] = new_b

                # Track-type ELO
                new_a_tt, new_b_tt = _elo_update(
                    self.elo_track_type[(a_id, track_type)],
                    self.elo_track_type[(b_id, track_type)],
                    1.0,
                    k_scaled,
                )
                self.elo_track_type[(a_id, track_type)] = new_a_tt
                self.elo_track_type[(b_id, track_type)] = new_b_tt

        # Team ELO: winner's team beats all others (team-level 1vAll update)
        if n > 0:
            winner_team = team_names[0]
            for j in range(1, n):
                loser_team = team_names[j]
                new_w, new_l = _elo_update(
                    self.team_elo[winner_team], self.team_elo[loser_team], 1.0, k_scaled
                )
                self.team_elo[winner_team] = new_w
                self.team_elo[loser_team] = new_l

        # Update driver history and career stats
        for _, row in ranked.iterrows():
            driver_id = int(row["DriversID"])
            finish = int(row["PositionFinish"])
            start = int(row["PositionStart"]) if pd.notna(row.get("PositionStart")) and row["PositionStart"] > 0 else 0
            laps_led = float(row["LapsLed"]) if pd.notna(row.get("LapsLed")) else 0.0
            laps_led_pct = laps_led / total_laps
            speed_avg = float(row["SpeedAvg"]) if pd.notna(row.get("SpeedAvg")) else 0.0

            self.driver_history[driver_id].append({
                "finish": finish,
                "start": start,
                "laps_led_pct": laps_led_pct,
                "speed_avg": speed_avg,
                "event_norm": event_norm,
                "track_type": track_type,
            })

            # Career stats
            self.career_stats[driver_id][1] += 1  # total races
            if finish == 1:
                self.career_stats[driver_id][0] += 1  # wins

            # Venue history
            venue_key = (driver_id, event_norm)
            self.venue_history[venue_key].append(finish)

    # ------------------------------------------------------------------
    # WARM: rebuild state from CSV up to today (service startup)
    # ------------------------------------------------------------------

    def warm(self, csv_path: str = INDYCAR_CSV) -> None:
        """
        Rebuild internal ELO and history state from full CSV history.
        Same logic as build_dataset() but discards feature rows — only updates state.
        """
        logger.info("Warming IndyCar feature extractor from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["session_type"] == "R"].copy()
        df["date_parsed"] = pd.to_datetime(
            df["session_date"], format="%A, %B %d, %Y", errors="coerce"
        )
        df["year"] = df["date_parsed"].dt.year
        df = df.dropna(subset=["date_parsed", "year"]).copy()
        df["year"] = df["year"].astype(int)

        df["DriversID"] = pd.to_numeric(df["DriversID"], errors="coerce").fillna(0).astype(int)
        df["PositionFinish"] = pd.to_numeric(df["PositionFinish"], errors="coerce")
        df["PositionStart"] = pd.to_numeric(df["PositionStart"], errors="coerce")
        df["LapsLed"] = pd.to_numeric(df["LapsLed"], errors="coerce").fillna(0)
        df["LapsComplete"] = pd.to_numeric(df["LapsComplete"], errors="coerce").fillna(0)
        df["SpeedAvg"] = pd.to_numeric(df["SpeedAvg"], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["PositionFinish"]).copy()
        df = df[df["DriversID"] > 0].copy()
        df["PositionFinish"] = df["PositionFinish"].astype(int)
        df["track_type"] = df["event_name"].fillna("").apply(classify_track_type)
        df["event_norm"] = df["event_name"].fillna("").str.lower().str.strip()
        df["race_id"] = df["events_session_id"].astype(str)
        df = df.sort_values(["date_parsed", "race_id", "PositionFinish"]).reset_index(drop=True)

        race_order = (
            df.groupby("race_id", sort=False)["date_parsed"]
            .first()
            .sort_values()
        )
        race_df_map = {rid: grp for rid, grp in df.groupby("race_id")}

        for race_id, _ in race_order.items():
            grp = race_df_map[race_id].copy()
            if len(grp) < MIN_RACE_SIZE:
                continue
            track_type = int(grp["track_type"].iloc[0])
            event_norm = str(grp["event_norm"].iloc[0])
            winner_laps = grp.loc[grp["PositionFinish"] == 1, "LapsComplete"]
            total_laps = float(winner_laps.iloc[0]) if len(winner_laps) > 0 else float(grp["LapsComplete"].max())
            total_laps = max(total_laps, 1.0)
            self._update_state_from_race(grp, track_type, event_norm, total_laps)

        self._fitted = True
        logger.info(
            "Warm complete: %d drivers, %d track-type ELOs, %d teams",
            len(self.elo_overall),
            len(self.elo_track_type),
            len(self.team_elo),
        )

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def get_features_for_race(
        self,
        drivers: list[dict[str, Any]],
        event_name: str,
        year: int,
    ) -> list[dict[str, Any]]:
        """
        For live inference: return feature dict per driver using current state.

        drivers: list of dicts, each with keys:
            - driver_id: int   (IndyCar driver ID, 0 if unknown)
            - driver_name: str
            - team_name: str   (optional, defaults to "Unknown")
        event_name: str  (e.g. "Indianapolis 500")
        year: int        (current season year)

        Returns list of dicts with FEATURES + driver_name, driver_id.
        """
        if not self._fitted:
            raise RuntimeError("Extractor not fitted — call warm() or build_dataset() first")

        track_type = classify_track_type(event_name)
        event_norm = event_name.lower().strip()

        result = []
        for drv in drivers:
            driver_id = int(drv.get("driver_id", 0))
            driver_name = str(drv.get("driver_name", "Unknown"))
            team_name = str(drv.get("team_name", "Unknown"))

            feats = self._extract_features(
                driver_id=driver_id,
                team_name=team_name,
                track_type=track_type,
                event_norm=event_norm,
                year=year,
            )
            feats["driver_name"] = driver_name
            feats["driver_id"] = driver_id
            feats["team_name"] = team_name
            result.append(feats)

        return result

    # ------------------------------------------------------------------
    # SERIALISATION
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("IndycarFeatureExtractor saved to %s", path)

    @staticmethod
    def load(path: str) -> "IndycarFeatureExtractor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("IndycarFeatureExtractor loaded from %s", path)
        return obj

    @property
    def driver_count(self) -> int:
        return len(self.elo_overall)

    @property
    def team_count(self) -> int:
        return len(self.team_elo)
