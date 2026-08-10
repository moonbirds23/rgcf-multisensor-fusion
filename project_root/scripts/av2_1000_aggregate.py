"""Scenario-level seed aggregation and paired bootstrap confidence intervals."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from tools.av2_1000.common import atomic_json
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=5000);args=p.parse_args()
 with args.input.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
 groups=defaultdict(list)
 for r in rows: groups[(r["method"],r["model_seed"],r["scenario_id"])].append(float(r["position_rmse"]))
 first=[{"method":k[0],"model_seed":k[1],"scenario_id":k[2],"position_rmse":float(np.mean(v))} for k,v in groups.items()]
 second=defaultdict(list)
 for r in first: second[(r["method"],r["scenario_id"])].append(r["position_rmse"])
 final=[{"method":k[0],"scenario_id":k[1],"position_rmse":float(np.mean(v)),"model_seed_std":float(np.std(v))} for k,v in second.items()]
 args.output.mkdir(parents=True,exist_ok=True)
 for name,data,fields in (("scenario_level_after_measurement_seed_average.csv",first,["method","model_seed","scenario_id","position_rmse"]),("scenario_level_after_model_seed_average.csv",final,["method","scenario_id","position_rmse","model_seed_std"])):
  with (args.output/name).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 by=defaultdict(list)
 for r in final:by[r["method"]].append(r["position_rmse"])
 overall=[{"method":k,"position_rmse_mean":float(np.mean(v)),"position_rmse_std":float(np.std(v)),"scenarios":len(v)}for k,v in sorted(by.items())]
 with (args.output/"overall.csv").open("w",newline="",encoding="utf-8")as f:w=csv.DictWriter(f,fieldnames=list(overall[0]) if overall else ["method"]);w.writeheader();w.writerows(overall)
 paired=[]; lookup=defaultdict(dict)
 for r in final: lookup[r["method"]][r["scenario_id"]]=r["position_rmse"]
 if "full_pefnet" in lookup:
  rng=np.random.default_rng(20260711)
  for baseline in ("t1","covariance_intersection","ci_eu","centralized_multisensor_ekf","posterior_only","pefnet_no_external_evidence"):
   if baseline not in lookup: continue
   ids=sorted(set(lookup["full_pefnet"]).intersection(lookup[baseline])); diff=np.asarray([lookup["full_pefnet"][i]-lookup[baseline][i] for i in ids])
   boots=np.asarray([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(args.bootstrap)])
   paired.append({"comparison":"full_pefnet - "+baseline,"scenarios":len(ids),"mean_difference":float(diff.mean()),"ci95_low":float(np.percentile(boots,2.5)),"ci95_high":float(np.percentile(boots,97.5))})
 if paired:
  with (args.output/"bootstrap_confidence_intervals.csv").open("w",newline="",encoding="utf-8")as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 atomic_json(args.output/"aggregation_manifest.json",{"measurement_seed_aggregated_first":True,"scenario_count_by_method":{k:len(v)for k,v in by.items()}});(args.output/"_SUCCESS").write_text("ok\n")
if __name__=="__main__":main()
