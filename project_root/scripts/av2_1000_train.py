"""Strict-CUDA training for full PEFNet and evidence ablations."""
from __future__ import annotations
import argparse, csv, json, os, random, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.av2_shard_dataset import Av2TimeStepShardDataset
from models.gnn_fusion import MeasurementEvaluatedRGCFA0Directional
from tools.av2_1000.common import atomic_json, formal_paths, require_cuda, sha256_file

def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _model(): return MeasurementEvaluatedRGCFA0Directional(post_in_dim=9, meas_in_dim=18, evidence_in_dim=16, pair_dim=8, hidden_dim=64, meas_hidden_dim=64, output_fusion_mode="info_diag")
def _adapt(batch, method):
    import torch
    post, mask, meas, evidence, evidence_mask, pair, target = [batch[x] for x in ("post_feat","mask","meas_feat","evidence_feat","evidence_mask","mp_pair_feat","target")]
    if method == "posterior_only": meas, evidence, evidence_mask, pair = torch.zeros_like(meas), torch.zeros_like(evidence), torch.zeros_like(evidence_mask), torch.zeros_like(pair)
    elif method == "pefnet_no_external_evidence": evidence, evidence_mask, pair = torch.zeros_like(evidence), torch.zeros_like(evidence_mask), torch.cat((pair[:, :, :3], torch.zeros_like(pair[:, :, 3:])), 2)
    return post, mask, meas, evidence, evidence_mask, pair, target
def _epoch(model, loader, device, method, optimizer=None):
    import torch
    train = optimizer is not None; model.train(train); total = n = 0.0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k,v in batch.items()}; post, mask, meas, evidence, evidence_mask, pair, target = _adapt(batch, method)
        with torch.set_grad_enabled(train):
            pred = model(post_feat=post, mask=mask, meas_feat=meas, evidence_feat=evidence, evidence_mask=evidence_mask, mp_pair_feat=pair).pred
            loss = ((pred[:,:2]-target[:,:2])**2).mean() + .2*((pred[:,2:]-target[:,2:])**2).mean()
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            if train: optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += float(loss.detach()) * len(target); n += len(target)
    return total/max(n,1)
def _save(path, payload):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True); fd, temp = tempfile.mkstemp(dir=str(path.parent), suffix=".pt"); os.close(fd)
    try: torch.save(payload, temp); os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--method",choices=("posterior_only","pefnet_no_external_evidence","full_pefnet"),required=True); p.add_argument("--model-seed",type=int,required=True); p.add_argument("--epochs",type=int,default=30); p.add_argument("--patience",type=int,default=5); p.add_argument("--batch-size",type=int,default=512); args=p.parse_args()
    import torch
    device=require_cuda(); paths=formal_paths(args.root); train=Av2TimeStepShardDataset(paths["features"] / "train"); valid=Av2TimeStepShardDataset(paths["features"] / "validation")
    if not len(train) or not len(valid): raise RuntimeError("missing formal train/validation feature rows")
    _seed(args.model_seed); model=_model().to(device); assert next(model.parameters()).is_cuda; opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loader=lambda ds,shuffle: torch.utils.data.DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=2,pin_memory=True)
    run=paths["runs"] / args.method / ("seed_"+str(args.model_seed)); history=[]; best=float("inf"); stale=0
    for epoch in range(1,args.epochs+1):
        tr=_epoch(model,loader(train,True),device,args.method,opt); va=_epoch(model,loader(valid,False),device,args.method); history.append({"epoch":epoch,"train_loss":tr,"validation_loss":va})
        metadata={"method":args.method,"model_seed":args.model_seed,"epoch":epoch,"best_validation_loss":min(best,va),"train_manifest_sha256":sha256_file(paths["manifests"] / "train_700.csv"),"validation_manifest_sha256":sha256_file(paths["manifests"] / "val_100.csv"),"torch_version":torch.__version__,"cuda_version":torch.version.cuda}
        _save(run / "last.pt", {"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"metadata":metadata})
        if va < best: best=va; stale=0; _save(run / "best.pt", {"model_state_dict":model.state_dict(),"metadata":metadata})
        else: stale+=1
        if stale>=args.patience: break
    run.mkdir(parents=True,exist_ok=True)
    with (run/"history.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=["epoch","train_loss","validation_loss"]);w.writeheader();w.writerows(history)
    atomic_json(run/"run_manifest.json", {"method":args.method,"model_seed":args.model_seed,"best_validation_loss":best,"strict_cuda":True,"epochs_completed":len(history)}); (run/"_SUCCESS").write_text("ok\n")
if __name__=="__main__": main()
