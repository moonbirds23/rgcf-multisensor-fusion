"""Frozen, data-independent configuration for AV2 Pilot Protocol v1.0.

This module is deliberately limited to protocol constants.  It does not locate,
read, or inspect AV2 data, and it must not be adapted from pilot results.  A
protocol correction requires a new version and a full pilot rerun.
"""

from __future__ import annotations

from dataclasses import dataclass


PROTOCOL_VERSION = "AV2_PILOT_V1.0"
PROTOCOL_DOCUMENT_RELATIVE_PATH = "docs/AV2_PILOT_PROTOCOL_V1.md"
# SHA-256 of the frozen protocol document at the time these constants were made.
PROTOCOL_DOCUMENT_SHA256 = "f41ecebc0c33ede4859545d8c91433d114cc4a93c5caa602c6b5d76d95461290"


@dataclass(frozen=True)
class ClosedRange:
    minimum: float
    maximum: float

    def validate(self, name: str) -> None:
        if self.minimum > self.maximum:
            raise ValueError(f"{name}: minimum must not exceed maximum")


@dataclass(frozen=True)
class SensorSpec:
    name: str
    role: str
    measurement_model: str
    local_position_xy_m: tuple[float, float] | None
    rate_hz: float
    noise_standard_deviations: tuple[tuple[str, float], ...]

    def validate(self) -> None:
        if self.role not in {"posterior", "evidence"}:
            raise ValueError(f"{self.name}: unknown sensor role {self.role!r}")
        if self.rate_hz <= 0.0:
            raise ValueError(f"{self.name}: rate_hz must be positive")
        if any(value <= 0.0 for _, value in self.noise_standard_deviations):
            raise ValueError(f"{self.name}: noise standard deviations must be positive")


@dataclass(frozen=True)
class EkfSpec:
    state_fields: tuple[str, ...]
    motion_model: str
    timestamp_derived_dt: bool
    acceleration_sigma_mps2: float
    initial_velocity_mps: tuple[float, float]
    initial_velocity_variance_m2ps2: float
    gps_initial_position_variance_m2: tuple[float, float]
    radar_initialization: str
    radar_minimum_position_eigenvalue_m2: float
    estimate_velocity_from_future_samples: bool

    def validate(self) -> None:
        if self.state_fields != ("px", "py", "vx", "vy"):
            raise ValueError("AV2 Pilot requires state [px, py, vx, vy]")
        if not self.timestamp_derived_dt or self.acceleration_sigma_mps2 <= 0.0:
            raise ValueError("AV2 Pilot EKF must use timestamp dt and positive process noise")
        if self.initial_velocity_variance_m2ps2 <= 0.0 or self.radar_minimum_position_eigenvalue_m2 <= 0.0:
            raise ValueError("AV2 Pilot EKF initialization variances must be positive")
        if self.estimate_velocity_from_future_samples:
            raise ValueError("AV2 Pilot forbids future-sample velocity initialization")


@dataclass(frozen=True)
class TrajectoryEligibilitySpec:
    focal_vehicle_only: bool
    required_object_type: str
    minimum_valid_states: int
    maximum_internal_missing_gap_steps: int
    minimum_displacement_m: float
    minimum_mean_speed_mps: float
    local_coordinates_from_time_zero_only: bool
    warmup_seconds: float
    causal_processing: bool
    future_state_as_input: bool

    def validate(self) -> None:
        if not self.focal_vehicle_only or self.required_object_type != "VEHICLE":
            raise ValueError("AV2 Pilot is restricted to focal VEHICLE tracks")
        if self.minimum_valid_states < 1 or self.maximum_internal_missing_gap_steps < 0:
            raise ValueError("AV2 Pilot eligibility limits are invalid")
        if self.warmup_seconds <= 0.0 or not self.causal_processing or self.future_state_as_input:
            raise ValueError("AV2 Pilot time protocol must be causal with a positive warmup")


@dataclass(frozen=True)
class BiasRampSpec:
    posterior_source_probability: float
    evidence_source_probability: float
    onset_seconds: ClosedRange
    ramp_duration_seconds: ClosedRange
    gps_terminal_bias_m: ClosedRange
    radar_range_terminal_bias_m: ClosedRange
    radar_bearing_terminal_bias_deg: ClosedRange
    radar_range_or_bearing_probability: float
    aoa_terminal_bias_deg: ClosedRange
    uwb_terminal_bias_m: ClosedRange
    random_signs_where_applicable: bool

    def validate(self) -> None:
        if self.posterior_source_probability + self.evidence_source_probability != 1.0:
            raise ValueError("bias ramp source probabilities must sum to one")
        if not 0.0 <= self.radar_range_or_bearing_probability <= 1.0:
            raise ValueError("radar range-or-bearing probability must lie in [0, 1]")
        for name, value in (
            ("bias onset", self.onset_seconds),
            ("bias ramp duration", self.ramp_duration_seconds),
            ("GPS terminal bias", self.gps_terminal_bias_m),
            ("Radar range terminal bias", self.radar_range_terminal_bias_m),
            ("Radar bearing terminal bias", self.radar_bearing_terminal_bias_deg),
            ("AOA terminal bias", self.aoa_terminal_bias_deg),
            ("UWB terminal bias", self.uwb_terminal_bias_m),
        ):
            value.validate(name)


@dataclass(frozen=True)
class FaultSpec:
    condition_probabilities: tuple[tuple[str, float], ...]
    exactly_one_principal_condition: bool
    bias_ramp: BiasRampSpec
    dropout_onset_seconds: ClosedRange
    dropout_duration_seconds: ClosedRange
    posterior_dropout_predict_without_update: bool
    posterior_measurement_node_invalid: bool
    posterior_node_remains_available: bool
    evidence_dropout_mask_zero: bool
    evidence_dropout_reuses_prior_measurement: bool
    delay_steps: tuple[int, ...]
    early_delayed_samples_invalid: bool
    delay_uses_future_backfill: bool
    covariance_mismatch_alphas: tuple[float, ...]
    covariance_mismatch_changes_ekf_internal_covariance: bool
    covariance_mismatch_changes_only_fusion_report: bool
    evidence_noise_beta_training: ClosedRange
    evidence_noise_beta_formal_sweep: tuple[float, ...]

    def validate(self) -> None:
        probability_sum = sum(value for _, value in self.condition_probabilities)
        if abs(probability_sum - 1.0) > 1e-12 or not self.exactly_one_principal_condition:
            raise ValueError("fault condition probabilities must sum to one with one principal condition")
        if any(value < 0.0 for _, value in self.condition_probabilities):
            raise ValueError("fault condition probabilities must be non-negative")
        self.bias_ramp.validate()
        self.dropout_onset_seconds.validate("dropout onset")
        self.dropout_duration_seconds.validate("dropout duration")
        self.evidence_noise_beta_training.validate("evidence noise beta")
        if self.delay_steps != (1, 2, 3) or not self.early_delayed_samples_invalid or self.delay_uses_future_backfill:
            raise ValueError("AV2 Pilot delay must be causal with choices (1, 2, 3)")
        if self.covariance_mismatch_changes_ekf_internal_covariance or not self.covariance_mismatch_changes_only_fusion_report:
            raise ValueError("covariance mismatch must change only fusion-facing covariance")


@dataclass(frozen=True)
class ManifestSpec:
    stage: str
    source_split: str
    train_scenarios: int
    validation_scenarios: int
    development_holdout_scenarios: int
    split_by_scenario_before_variants: bool
    require_sha256: bool
    reads_official_validation: bool

    def validate(self) -> None:
        if self.source_split != "train" or self.reads_official_validation:
            raise ValueError(f"{self.stage}: Pilot may only use official train")
        if min(self.train_scenarios, self.validation_scenarios, self.development_holdout_scenarios) <= 0:
            raise ValueError(f"{self.stage}: manifest counts must be positive")
        if not self.split_by_scenario_before_variants or not self.require_sha256:
            raise ValueError(f"{self.stage}: manifests must split first and be hashed")


@dataclass(frozen=True)
class SeedSpec:
    model_seeds: tuple[int, ...]
    measurement_seeds: tuple[int, ...]
    purpose: str

    def validate(self) -> None:
        if not self.model_seeds:
            raise ValueError(f"{self.purpose}: at least one model seed is required")
        if len(set(self.model_seeds)) != len(self.model_seeds) or len(set(self.measurement_seeds)) != len(self.measurement_seeds):
            raise ValueError(f"{self.purpose}: seeds must be unique")


@dataclass(frozen=True)
class EngineeringGates:
    successful_scene_rate_minimum: float
    finite_output_rate_minimum: float
    positive_covariance_diagonal_rate_minimum: float
    spd_covariance_rate_minimum: float
    causal_leakage_tests_must_pass: bool
    manifest_overlap_count_maximum: int

    def validate(self) -> None:
        if not all(0.0 <= value <= 1.0 for value in (
            self.successful_scene_rate_minimum,
            self.finite_output_rate_minimum,
            self.positive_covariance_diagonal_rate_minimum,
            self.spd_covariance_rate_minimum,
        )):
            raise ValueError("engineering-rate gates must be probabilities")
        if not self.causal_leakage_tests_must_pass or self.manifest_overlap_count_maximum != 0:
            raise ValueError("causal leakage and manifest overlap gates are mandatory")


@dataclass(frozen=True)
class DecisionGates:
    nominal_rmse_degradation_vs_ci_maximum: float
    nominal_p95_degradation_vs_ci_maximum: float
    nominal_motion_group_rmse_degradation_vs_ci_maximum: float
    mixed_rmse_improvement_vs_ci_minimum: float
    mixed_p95_improvement_vs_ci_minimum: float
    external_evidence_rmse_improvement_minimum: float
    external_evidence_p95_improvement_minimum: float
    nll_degradation_vs_ci_maximum: float
    coverage_level: float
    coverage_must_not_be_materially_worse_than_ci: bool
    uncertainty_metrics_must_be_finite: bool
    conditional_nominal_rmse_degradation: ClosedRange
    conditional_mixed_rmse_improvement: ClosedRange
    conditional_mixed_p95_improvement: ClosedRange
    no_go_nominal_rmse_degradation_exclusive_minimum: float
    no_go_motion_group_rmse_degradation_exclusive_minimum: float

    def validate(self) -> None:
        if not 0.0 < self.coverage_level < 1.0:
            raise ValueError("coverage level must lie in (0, 1)")
        for name, value in (
            ("conditional nominal RMSE", self.conditional_nominal_rmse_degradation),
            ("conditional mixed RMSE", self.conditional_mixed_rmse_improvement),
            ("conditional mixed P95", self.conditional_mixed_p95_improvement),
        ):
            value.validate(name)
        if not self.coverage_must_not_be_materially_worse_than_ci or not self.uncertainty_metrics_must_be_finite:
            raise ValueError("uncertainty-quality gates are mandatory")


@dataclass(frozen=True)
class ResourceBudget:
    pilot_gpu_hours_maximum: float
    pilot_cpu_preprocessing_hours_maximum: float
    pilot_wall_clock_days_maximum: float
    formal_gpu_hours_preferred: ClosedRange
    formal_gpu_hours_maximum: float
    formal_wall_clock_days_maximum: float

    def validate(self) -> None:
        self.formal_gpu_hours_preferred.validate("formal preferred GPU hours")
        if min(self.pilot_gpu_hours_maximum, self.pilot_cpu_preprocessing_hours_maximum, self.pilot_wall_clock_days_maximum) <= 0.0:
            raise ValueError("Pilot resource budget must be positive")
        if self.formal_gpu_hours_preferred.maximum > self.formal_gpu_hours_maximum:
            raise ValueError("formal preferred GPU hours must not exceed the hard cap")


@dataclass(frozen=True)
class Av2PilotProtocol:
    version: str
    sensors: tuple[SensorSpec, ...]
    ekf: EkfSpec
    trajectory: TrajectoryEligibilitySpec
    faults: FaultSpec
    smoke_manifest: ManifestSpec
    decision_manifest: ManifestSpec
    smoke_seeds: SeedSpec
    decision_seeds: SeedSpec
    engineering_gates: EngineeringGates
    decision_gates: DecisionGates
    budget: ResourceBudget
    parameters_frozen_before_pilot: bool
    automatic_threshold_adjustment: bool

    def validate(self) -> None:
        if self.version != PROTOCOL_VERSION or not self.parameters_frozen_before_pilot or self.automatic_threshold_adjustment:
            raise ValueError("AV2 Pilot protocol must remain frozen at v1.0")
        if tuple(sensor.name for sensor in self.sensors) != ("T1", "T2", "T3", "E1", "E2"):
            raise ValueError("AV2 Pilot requires T1/T2/T3/E1/E2 in fixed order")
        if tuple(sensor.role for sensor in self.sensors) != ("posterior", "posterior", "posterior", "evidence", "evidence"):
            raise ValueError("only T1/T2/T3 may be fusion posteriors")
        for sensor in self.sensors:
            sensor.validate()
        self.ekf.validate()
        self.trajectory.validate()
        self.faults.validate()
        self.smoke_manifest.validate()
        self.decision_manifest.validate()
        self.smoke_seeds.validate()
        self.decision_seeds.validate()
        self.engineering_gates.validate()
        self.decision_gates.validate()
        self.budget.validate()


AV2_PILOT_V1 = Av2PilotProtocol(
    version=PROTOCOL_VERSION,
    sensors=(
        SensorSpec("T1", "posterior", "gps_2d", None, 10.0, (("x_m", 3.0), ("y_m", 3.0))),
        SensorSpec("T2", "posterior", "range_bearing", (80.0, -50.0), 5.0, (("range_m", 2.5), ("bearing_deg", 0.6))),
        SensorSpec("T3", "posterior", "range_bearing", (-60.0, 70.0), 5.0, (("range_m", 3.0), ("bearing_deg", 0.8))),
        SensorSpec("E1", "evidence", "aoa_bearing", (90.0, 65.0), 5.0, (("bearing_deg", 1.0),)),
        SensorSpec("E2", "evidence", "uwb_range", (-85.0, 50.0), 10.0, (("range_m", 1.5),)),
    ),
    ekf=EkfSpec(
        state_fields=("px", "py", "vx", "vy"),
        motion_model="constant_velocity",
        timestamp_derived_dt=True,
        acceleration_sigma_mps2=2.0,
        initial_velocity_mps=(0.0, 0.0),
        initial_velocity_variance_m2ps2=36.0,
        gps_initial_position_variance_m2=(9.0, 9.0),
        radar_initialization="first_range_bearing_jacobian_propagated",
        radar_minimum_position_eigenvalue_m2=1.0,
        estimate_velocity_from_future_samples=False,
    ),
    trajectory=TrajectoryEligibilitySpec(True, "VEHICLE", 105, 2, 15.0, 2.0, True, 1.0, True, False),
    faults=FaultSpec(
        condition_probabilities=(("nominal", 0.60), ("bias_ramp", 0.15), ("covariance_mismatch", 0.10), ("dropout", 0.10), ("time_delay", 0.05)),
        exactly_one_principal_condition=True,
        bias_ramp=BiasRampSpec(0.70, 0.30, ClosedRange(2.0, 4.0), ClosedRange(1.5, 3.0), ClosedRange(4.0, 8.0), ClosedRange(4.0, 8.0), ClosedRange(0.4, 0.8), 0.5, ClosedRange(0.5, 1.0), ClosedRange(2.0, 5.0), True),
        dropout_onset_seconds=ClosedRange(2.0, 8.0),
        dropout_duration_seconds=ClosedRange(0.5, 1.5),
        posterior_dropout_predict_without_update=True,
        posterior_measurement_node_invalid=True,
        posterior_node_remains_available=True,
        evidence_dropout_mask_zero=True,
        evidence_dropout_reuses_prior_measurement=False,
        delay_steps=(1, 2, 3),
        early_delayed_samples_invalid=True,
        delay_uses_future_backfill=False,
        covariance_mismatch_alphas=(0.25, 0.50, 2.0, 4.0),
        covariance_mismatch_changes_ekf_internal_covariance=False,
        covariance_mismatch_changes_only_fusion_report=True,
        evidence_noise_beta_training=ClosedRange(0.8, 1.5),
        evidence_noise_beta_formal_sweep=(0.5, 1.0, 1.5, 2.0, 3.0),
    ),
    smoke_manifest=ManifestSpec("smoke", "train", 50, 20, 20, True, True, False),
    decision_manifest=ManifestSpec("decision", "train", 500, 100, 200, True, True, False),
    smoke_seeds=SeedSpec((0,), (), "smoke"),
    decision_seeds=SeedSpec((0, 1, 2), (100, 101, 102), "decision"),
    engineering_gates=EngineeringGates(0.99, 1.0, 1.0, 0.999, True, 0),
    decision_gates=DecisionGates(
        nominal_rmse_degradation_vs_ci_maximum=0.02,
        nominal_p95_degradation_vs_ci_maximum=0.03,
        nominal_motion_group_rmse_degradation_vs_ci_maximum=0.07,
        mixed_rmse_improvement_vs_ci_minimum=0.03,
        mixed_p95_improvement_vs_ci_minimum=0.05,
        external_evidence_rmse_improvement_minimum=0.01,
        external_evidence_p95_improvement_minimum=0.02,
        nll_degradation_vs_ci_maximum=0.05,
        coverage_level=0.95,
        coverage_must_not_be_materially_worse_than_ci=True,
        uncertainty_metrics_must_be_finite=True,
        conditional_nominal_rmse_degradation=ClosedRange(0.02, 0.05),
        conditional_mixed_rmse_improvement=ClosedRange(0.01, 0.03),
        conditional_mixed_p95_improvement=ClosedRange(0.02, 0.05),
        no_go_nominal_rmse_degradation_exclusive_minimum=0.05,
        no_go_motion_group_rmse_degradation_exclusive_minimum=0.15,
    ),
    budget=ResourceBudget(8.0, 8.0, 1.0, ClosedRange(12.0, 30.0), 36.0, 4.0),
    parameters_frozen_before_pilot=True,
    automatic_threshold_adjustment=False,
)

# Validate eagerly so a hand edit cannot silently create an internally-invalid protocol.
AV2_PILOT_V1.validate()

