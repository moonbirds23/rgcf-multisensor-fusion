from __future__ import annotations

import copy

from configs.experiment_presets import build_experiment_config
from core.types import (
    ExperimentBundle,
    ExperimentIdentity,
    FaultRuntimeSpec,
    ModelRuntimeSpec,
    RunRequest,
    RuntimeFlags,
    SceneRuntimeSpec,
    TrainRuntimeSpec,
)


def _infer_runtime_flags(mode: str) -> RuntimeFlags:
    mode = mode.lower()
    if mode in ("simulate", "generate_dataset"):
        return RuntimeFlags(do_simulate=True)
    if mode == "train":
        return RuntimeFlags(do_simulate=True, do_train=True, do_eval=True)
    if mode == "repeat":
        return RuntimeFlags(do_simulate=True, do_train=True, do_eval=True, do_repeat=True)
    if mode == "plot_weights":
        return RuntimeFlags(do_plot_weights=True)
    if mode in ("fault_test", "generalization_test"):
        return RuntimeFlags(do_simulate=True, do_train=True, do_eval=True)
    raise ValueError(f"Unsupported mode: {mode}")


def _scene_spec(scenario) -> SceneRuntimeSpec:
    return SceneRuntimeSpec(scenario, scenario.scene_name, scenario.motion.dt, scenario.motion.T, len(scenario.sensor_layout.sensors))


def _fault_spec(fault) -> FaultRuntimeSpec:
    return FaultRuntimeSpec(fault, fault.mode, bool(fault.dropout.enabled), bool(fault.pollution.enabled))


def _model_spec(model) -> ModelRuntimeSpec:
    return ModelRuntimeSpec(model, model.model_name, model.in_dim, model.hidden_dim, model.use_post_stream, model.use_meas_stream, model.use_gate)


def _train_spec(train) -> TrainRuntimeSpec:
    return TrainRuntimeSpec(train, train.epochs, train.lr, train.batch_size, train.repeat_runs)


def load_experiment_bundle(request: RunRequest) -> ExperimentBundle:
    exp_cfg = build_experiment_config(request.preset_name)
    base = copy.deepcopy(exp_cfg.base)
    scenario = copy.deepcopy(exp_cfg.scenario)
    fault = copy.deepcopy(exp_cfg.fault)
    model = copy.deepcopy(exp_cfg.model)
    train = copy.deepcopy(exp_cfg.train)

    if request.device is not None:
        base.runtime.device = request.device
    if request.epochs is not None:
        train.epochs = request.epochs
    if request.lr is not None:
        train.lr = request.lr
    if request.batch_size is not None:
        train.batch_size = request.batch_size
    if request.repeat_runs is not None:
        train.repeat_runs = request.repeat_runs
    if request.hidden_dim is not None:
        model.hidden_dim = request.hidden_dim
        model.meas_hidden_dim = request.hidden_dim
        model.gate_hidden_dim = request.hidden_dim

    identity = ExperimentIdentity(
        preset_name=request.preset_name,
        experiment_name=request.experiment_name or exp_cfg.name,
        scene_name=scenario.scene_name,
        model_name=model.model_name,
        fault_mode=fault.mode,
    )
    return ExperimentBundle(
        identity=identity,
        base=base,
        scenario=scenario,
        fault=fault,
        model=model,
        train=train,
        runtime_flags=_infer_runtime_flags(request.mode),
        scene_spec=_scene_spec(scenario),
        fault_spec=_fault_spec(fault),
        model_spec=_model_spec(model),
        train_spec=_train_spec(train),
    )


def clone_bundle(bundle: ExperimentBundle) -> ExperimentBundle:
    return copy.deepcopy(bundle)
