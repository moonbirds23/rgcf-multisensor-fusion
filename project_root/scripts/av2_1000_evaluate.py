"""Evaluate fixed test scenarios without treating seeds as independent scenes."""
from __future__ import annotations
import argparse,csv,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from configs.av2_nominal_1000_v1 import NOMINAL_PROTOCOL as V31_PROTOCOL
from configs.av2_nominal_1000_v1 import WARMUP_SECONDS as V31_WARMUP_SECONDS
from configs.av2_nominal_1000_v32 import NOMINAL_PROTOCOL as V32_PROTOCOL
from configs.av2_nominal_1000_v32 import WARMUP_SECONDS as V32_WARMUP_SECONDS
from data.av2.feature_builder import build_av2_feature_arrays
from models.gnn_fusion import MeasurementEvaluatedRGCFA0Directional
from simulation.supplementary_baselines import fuse_av2_ci_tracks,run_centralized_multisensor_ekf,run_ci_eu_sequence
from tools.av2_1000.cache import load_outputs
from tools.av2_1000.common import formal_paths,require_cuda,write_csv

def _model(): return MeasurementEvaluatedRGCFA0Directional(post_in_dim=9,meas_in_dim=18,evidence_in_dim=16,pair_dim=8,hidden_dim=64,meas_hidden_dim=64,output_fusion_mode="info_diag")
def _protocol(name):
 if name=="v31":return V31_PROTOCOL,V31_WARMUP_SECONDS,"AV2_NOMINAL_1000_V3.1"
 if name=="v32":return V32_PROTOCOL,V32_WARMUP_SECONDS,"AV2_NOMINAL_1000_V3.2"
 raise ValueError(name)
def _metric(pred,target,mask):
 e=np.asarray(pred)[mask]-target[mask]; pos=np.linalg.norm(e[:,:2],axis=1);vel=np.linalg.norm(e[:,2:],axis=1)
 return float(np.sqrt(np.mean(pos**2))),float(np.sqrt(np.mean(vel**2))),float(np.percentile(pos,95)),int(mask.sum())
def _baseline(outputs,target,mask,protocol):
 x=np.asarray(outputs.posterior_mean);p=np.asarray(outputs.posterior_covariance_reported);valid=np.asarray(outputs.posterior_available)
 out={"t1":x[:,0],"t2":x[:,1],"t3":x[:,2]};mean=np.zeros_like(target);cw=np.zeros_like(target);ci=np.zeros_like(target)
 for t in range(len(target)):
  ids=np.flatnonzero(valid[t]);
  if not len(ids): continue
  mean[t]=x[t,ids].mean(0); weights=1/np.maximum(np.trace(p[t,ids],axis1=1,axis2=2),1e-8);weights/=weights.sum();cw[t]=(x[t,ids]*weights[:,None]).sum(0)
  state,cov,_=fuse_av2_ci_tracks(x[t],p[t],valid[t],n_grid=31);ci[t]=state
 ci_eu=run_ci_eu_sequence(outputs,protocol,ci_grid_points=31);cm_ekf=run_centralized_multisensor_ekf(outputs,protocol)
 out.update({"mean_fusion":mean,"covariance_weighted":cw,"covariance_intersection":ci,"ci_eu":ci_eu.xhat,"centralized_multisensor_ekf":cm_ekf.xhat});return out
def _predict(model,feature,method,device):
 import torch
 post=torch.from_numpy(feature.post_feat).float().to(device);mask=torch.from_numpy(feature.post_mask).float().to(device);meas=torch.from_numpy(feature.meas_feat).float().to(device);ev=torch.from_numpy(feature.evidence_feat).float().to(device);em=torch.from_numpy(feature.evidence_mask).float().to(device);pair=torch.from_numpy(feature.mp_pair_feat).float().to(device)
 if method=="posterior_only":meas,ev,em,pair=torch.zeros_like(meas),torch.zeros_like(ev),torch.zeros_like(em),torch.zeros_like(pair)
 elif method=="pefnet_no_external_evidence":ev,em,pair=torch.zeros_like(ev),torch.zeros_like(em),torch.cat((pair[:,:,:3],torch.zeros_like(pair[:,:,3:])),2)
 with torch.inference_mode():return model(post_feat=post,mask=mask,meas_feat=meas,evidence_feat=ev,evidence_mask=em,mp_pair_feat=pair).pred.cpu().numpy()
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--artifact-key",default="av2_1000");p.add_argument("--protocol",choices=("v31","v32"),default="v31");p.add_argument("--checkpoint",action="append",default=[],help="method=path; repeat for every learned model checkpoint");p.add_argument("--output",type=Path,default=None);args=p.parse_args()
 protocol,warmup_seconds,protocol_name=_protocol(args.protocol);device=require_cuda();paths=formal_paths(args.root,artifact_key=args.artifact_key);out=args.output or paths["results"];out.mkdir(parents=True,exist_ok=True);models=[]
 import torch
 for entry in args.checkpoint:
  method,path=entry.split("=",1);payload=torch.load(path,map_location=device);metadata=payload.get("metadata",{});checkpoint_protocol=metadata.get("protocol_name");
  if checkpoint_protocol not in (None,protocol_name):raise RuntimeError(f"checkpoint protocol mismatch: {checkpoint_protocol} != {protocol_name}")
  model=_model().to(device);model.load_state_dict(payload["model_state_dict"]);model.eval();models.append((method,int(metadata.get("model_seed",-1)),model))
 rows=[]
 for sim in sorted((paths["sim"] / "test").rglob("seed_*.npz")):
  outputs,target=load_outputs(sim);feature=build_av2_feature_arrays(outputs,target,warmup_seconds=warmup_seconds,protocol=protocol);mask=np.asarray(feature.eval_mask,dtype=bool)&np.all(outputs.posterior_available,axis=1)
  if not mask.any():raise RuntimeError("no common test posterior steps: "+str(sim))
  with np.load(sim,allow_pickle=False)as raw:
   sid=str(raw["scenario_id"]);seed=int(raw["measurement_seed"]);cached_protocol=str(raw["protocol_name"])
  if cached_protocol!=protocol_name:raise RuntimeError(f"cache protocol mismatch: {cached_protocol} != {protocol_name}: {sim}")
  for method,pred in _baseline(outputs,target,mask,protocol).items():
   a,b,c,n=_metric(pred,target,mask);rows.append({"method":method,"model_seed":-1,"scenario_id":sid,"measurement_seed":seed,"num_eval_steps":n,"position_rmse":a,"velocity_rmse":b,"position_p95":c})
  for method,model_seed,model in models:
   started=time.perf_counter();pred=_predict(model,feature,method,device);elapsed=time.perf_counter()-started;a,b,c,n=_metric(pred,target,mask);rows.append({"method":method,"model_seed":model_seed,"scenario_id":sid,"measurement_seed":seed,"num_eval_steps":n,"position_rmse":a,"velocity_rmse":b,"position_p95":c,"inference_time_ms_per_step":elapsed*1000/n})
 fields=sorted({k for r in rows for k in r});write_csv(out/"scenario_level.csv",rows,fields);(out/"_SUCCESS").write_text("ok\n")
if __name__=="__main__":main()
