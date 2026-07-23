#!/usr/bin/env python3
"""Run one auditable XGBoost totals prediction on a held-out NBA game."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from nba_api.stats.endpoints import leaguegamefinder

SEASONS = ("2023-24", "2024-25")
FEATURES = [
    "rolling_ortg",
    "rolling_drtg_OPP",
    "rolling_pace",
    "rolling_pace_OPP",
]


def fetch_history() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in SEASONS:
        endpoint = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
            timeout=60,
        )
        frames.append(endpoint.get_data_frames()[0])

    games = pd.concat(frames, ignore_index=True)
    games = games[games["SEASON_ID"].astype(str).str.startswith("2")].copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], utc=True)
    games = games.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])

    games["POSS"] = games["FGA"] + 0.44 * games["FTA"] - games["OREB"] + games["TOV"]
    games = games[games["POSS"] > 0].copy()
    games["Pace"] = games["POSS"]
    games["ORtg"] = (games["PTS"] / games["POSS"]) * 100
    games["PTS_ALLOWED"] = games["PTS"] - games["PLUS_MINUS"]
    games["DRtg"] = (games["PTS_ALLOWED"] / games["POSS"]) * 100

    grouped = games.groupby("TEAM_ID", group_keys=False)
    games["rolling_ortg"] = grouped["ORtg"].transform(
        lambda values: values.shift(1).rolling(10, min_periods=10).mean()
    )
    games["rolling_drtg"] = grouped["DRtg"].transform(
        lambda values: values.shift(1).rolling(10, min_periods=10).mean()
    )
    games["rolling_pace"] = grouped["Pace"].transform(
        lambda values: values.shift(1).rolling(10, min_periods=10).mean()
    )
    return games.dropna(subset=["rolling_ortg", "rolling_drtg", "rolling_pace"])


def build_matchups(games: pd.DataFrame) -> pd.DataFrame:
    matchups = pd.merge(games, games, on="GAME_ID", suffixes=("", "_OPP"))
    matchups = matchups[matchups["TEAM_ID"] != matchups["TEAM_ID_OPP"]].copy()
    matchups = matchups[
        np.isclose(matchups["GAME_DATE"].astype("int64"), matchups["GAME_DATE_OPP"].astype("int64"))
    ]
    return matchups.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"])


def run_one_prediction() -> dict[str, object]:
    games = fetch_history()
    matchups = build_matchups(games)
    if matchups.empty:
        raise RuntimeError("No matchup rows available after feature engineering")

    game_order = (
        matchups[["GAME_ID", "GAME_DATE"]]
        .drop_duplicates()
        .sort_values(["GAME_DATE", "GAME_ID"])
    )
    heldout_game_id = str(game_order.iloc[-1]["GAME_ID"])
    heldout_date = pd.Timestamp(game_order.iloc[-1]["GAME_DATE"])

    train = matchups[matchups["GAME_ID"].astype(str) != heldout_game_id].copy()
    train = train[train["GAME_DATE"] < heldout_date]
    heldout = matchups[matchups["GAME_ID"].astype(str) == heldout_game_id].copy()

    if len(heldout) != 2:
        raise RuntimeError(f"Expected two held-out team rows, received {len(heldout)}")
    if len(train) < 100:
        raise RuntimeError(f"Insufficient training rows: {len(train)}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[FEATURES], train["PTS"])
    predictions = model.predict(heldout[FEATURES])

    team_predictions: list[dict[str, object]] = []
    for (_, row), predicted_points in zip(heldout.iterrows(), predictions, strict=True):
        team_predictions.append(
            {
                "team": str(row["TEAM_NAME"]),
                "opponent": str(row["TEAM_NAME_OPP"]),
                "matchup": str(row.get("MATCHUP", "")),
                "predicted_points": round(float(predicted_points), 3),
                "actual_points": int(row["PTS"]),
            }
        )

    predicted_total = float(np.sum(predictions))
    actual_total = int(heldout["PTS"].sum())
    feature_cutoff_at = pd.Timestamp(train["GAME_DATE"].max()).isoformat()
    generated_at = datetime.now(timezone.utc).isoformat()
    commit_sha = os.getenv("GITHUB_SHA", "local")[:12]

    return {
        "event_id": f"NBA-{heldout_game_id}",
        "event_date": heldout_date.isoformat(),
        "prediction_type": "full_game_total_points_point_estimate",
        "model_name": "totals-edge-engine-xgbregressor",
        "model_version": f"xgb-500-depth4-{commit_sha}",
        "probability_calibrated": None,
        "calibration_status": "uncalibrated",
        "prediction_generated_at": generated_at,
        "feature_cutoff_at": feature_cutoff_at,
        "brier_score": None,
        "validation_sample_size": int(len(train)),
        "features": FEATURES,
        "predicted_total": round(predicted_total, 3),
        "actual_total": actual_total,
        "absolute_error": round(abs(predicted_total - actual_total), 3),
        "teams": team_predictions,
        "note": (
            "This is a historical holdout point forecast. It is not a calibrated over/under "
            "probability and must not use validated Kelly sizing."
        ),
    }


if __name__ == "__main__":
    result = run_one_prediction()
    print("DOCTORE_XGBOOST_PREDICTION=" + json.dumps(result, separators=(",", ":")))
