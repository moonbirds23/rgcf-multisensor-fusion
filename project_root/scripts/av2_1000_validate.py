"""Fail-fast structural validation for formal AV2 artifacts."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from tools.av2_1000.common import formal_paths,read_csv
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("stage",choices=("manifest","truth","sim","features","all"));args=p.parse_args();x=formal_paths(args.root)
 if args.stage in ("manifest","all"):
  rows=[]
  for name,n,source in (("train_700.csv",700,"train"),("val_100.csv",100,"train"),("test_200.csv",200,"val")):
   group=read_csv(x["manifests"]/name);assert len(group)==n,(name,len(group));assert all(r["official_split"]==source for r in group);rows+=group
  assert len({r["scenario_id"]for r in rows})==1000,"manifest overlap"
 if args.stage in ("truth","all"): assert len(list(x["truth"].rglob("*.npz")))==1000,"truth cache count"
 if args.stage in ("sim","all"): assert len(list(x["sim"].rglob("seed_*.npz")))==1400,"nominal sim cache count"
 if args.stage in ("features","all"):
  for split in ("train","validation","test"):assert (x["features"]/split/"_SUCCESS").exists() or list((x["features"]/split).glob("shard_*")),"missing feature shards: "+split
 print("AV2 1000 validation: PASS")
if __name__=="__main__":main()
