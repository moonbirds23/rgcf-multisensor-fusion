"""Prepare seed-aggregated AV2 1000 values for static figure authoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results" / "av2_1000"
    raw = pd.read_csv(results / "scenario_level.csv")
    metadata = pd.read_csv(root / "data" / "manifests" / "av2_1000_v2" / "test_200.csv")
    scenario = (
        raw.groupby(["method", "model_seed", "scenario_id"], as_index=False)["position_rmse"].mean()
        .groupby(["method", "scenario_id"], as_index=False)["position_rmse"].mean()
        .merge(metadata[["scenario_id", "motion_type", "city_name"]], on="scenario_id", validate="many_to_one")
    )
    paired = scenario[scenario["method"].isin(["full_pefnet", "covariance_intersection"])]
    paired = paired.pivot(index=["scenario_id", "motion_type", "city_name"], columns="method", values="position_rmse").reset_index()
    paired = paired.rename(columns={"full_pefnet": "full", "covariance_intersection": "ci"})
    selected = ["full_pefnet", "covariance_intersection", "pefnet_no_external_evidence", "posterior_only", "t1"]
    motion = pd.read_csv(results / "by_motion.csv")
    city = pd.read_csv(results / "by_city.csv")
    payload = {
        "scatter": paired.round(4).to_dict("records"),
        "motion": motion[motion["method"].isin(selected)].round(4).to_dict("records"),
        "city": city[city["method"].isin(["full_pefnet", "covariance_intersection"])] .round(4).to_dict("records"),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
