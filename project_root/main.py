from __future__ import annotations

import argparse
from typing import Any, Dict

from core.config_loader import load_experiment_bundle
from core.result_manager import ResultManager
from core.types import RunRequest
from experiments.generate_dataset import run_generate_dataset_experiment
from experiments.simulate import run_simulate_experiment
from experiments.train import run_train_experiment
from training.repeat_runner import repeat_training_evaluation


def _safe_dict(obj: Any) -> Dict:
    return obj if isinstance(obj, dict) else {}


def build_arg_parser():
    p = argparse.ArgumentParser(description="NF-DKF / RGCF unified experiment entry")
    p.add_argument("--mode", default="simulate", choices=["simulate", "generate_dataset", "train", "repeat", "fault_test", "generalization_test", "plot_weights"])
    p.add_argument("--preset", default="clean_baseline")
    p.add_argument("--train_preset", default=None)
    p.add_argument("--test_preset", default=None)
    p.add_argument("--fault_mode", default=None, choices=[None, "clean", "drop", "pollution", "both"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--repeat_runs", type=int, default=5)
    p.add_argument("--dataset_store_root", default="dataset_store")
    p.add_argument("--dataset_note", default="")
    p.add_argument("--dataset_id", default=None)
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--use_latest_matching_dataset", action="store_true")
    p.add_argument("--experiment_name", default=None)
    return p


def _req(args) -> RunRequest:
    return RunRequest(args.mode, args.preset, args.device, args.epochs, args.lr, args.batch_size, args.hidden_dim, args.repeat_runs, args.experiment_name)


def _brief(bundle):
    print(f"Preset      : {bundle.identity.preset_name}")
    print(f"Experiment  : {bundle.identity.experiment_name}")
    print(f"Scene       : {bundle.identity.scene_name}")
    print(f"Model       : {bundle.identity.model_name}")
    print(f"Fault mode  : {bundle.identity.fault_mode}")
    print(f"Device      : {bundle.base.runtime.device}")


def main():
    args = build_arg_parser().parse_args()
    bundle = load_experiment_bundle(_req(args))
    _brief(bundle)

    if args.mode == "simulate":
        res = run_simulate_experiment(bundle)
        rm = ResultManager(bundle, mode="simulate", experiment_name_override=bundle.identity.experiment_name)
        rm.save_simulate_result(sim=res.sim, baseline_metrics=res.baseline_metrics, fault_logs=res.fault_logs)
        print(f"Results saved to: {rm.run_dir}")
        return

    if args.mode == "generate_dataset":
        res = run_generate_dataset_experiment(bundle, dataset_store_root=args.dataset_store_root, dataset_note=args.dataset_note)
        print(f"dataset_store_dir: {res.dataset_store_dir}")
        print(res.counts)
        return

    if args.mode == "train":
        res = run_train_experiment(bundle, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, dataset_store_root=args.dataset_store_root, dataset_id=args.dataset_id, dataset_dir=args.dataset_dir, use_latest_matching_dataset=args.use_latest_matching_dataset)
        rm = ResultManager(bundle, mode="train", experiment_name_override=bundle.identity.experiment_name)
        rm.save_train_result(train_info=res.train_info, history=res.history, quick_baseline_metrics=res.quick_baseline_metrics, quick_gnn_metrics=res.quick_gnn_metrics, quick_sim=res.quick_sim, model=res.model, quick_gnn_timeseries=res.quick_gnn_timeseries)
        print(f"Results saved to: {rm.run_dir}")
        return

    if args.mode == "repeat":
        res = repeat_training_evaluation(bundle, repeat_runs=args.repeat_runs, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, dataset_store_root=args.dataset_store_root, dataset_id=args.dataset_id, dataset_dir=args.dataset_dir, use_latest_matching_dataset=args.use_latest_matching_dataset)
        rm = ResultManager(bundle, mode="repeat", experiment_name_override=bundle.identity.experiment_name)
        rm.save_repeat_result(repeat_summary=res.summary, all_results=res.all_results)
        print(f"Results saved to: {rm.run_dir}")
        return

    raise NotImplementedError(f"{args.mode} is not implemented in the cleaned entrypoint.")


if __name__ == "__main__":
    main()
