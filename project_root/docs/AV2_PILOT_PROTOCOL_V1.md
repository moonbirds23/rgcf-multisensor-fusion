# AV2 Pilot Protocol v1.0 (Frozen)

Status: frozen before AV2 data download and coding.  Results must not be used to
change the parameters or decision thresholds below.  A verified implementation
bug, data leakage, unit error, mask error, numerical divergence, invalid local
coordinate conversion, or unloaded configuration invalidates the Pilot; it must
then be versioned as `pilot_v1.1` and fully rerun.

## Scientific boundary

- Task: causal multi-sensor state estimation, not motion forecasting.
- Use the complete AV2 focal-vehicle trajectory as an offline reference sequence.
  At time `k`, every filter, feature, and model input may use only information at
  or before `k`; future truth is supervision/evaluation only.
- Derive `dt` from timestamps.  Use `warmup_seconds = 1.0` rather than a fixed
  frame count.  Generate measurements and run filters on the full sequence; use
  only `elapsed_time >= warmup_seconds` for loss and reported metrics.
- Use only `focal_track_id` with `object_type = VEHICLE`.  Keep a sample only if
  it has at least 105 valid states, displacement at least 15 m, mean speed at
  least 2 m/s, and no internal missing gap longer than two frames.
- Construct local coordinates from position and heading at time zero only.

## Dataset isolation

Pilot train/validation/development holdout all come from AV2 official `train`:
`500 / 100 / 200`.  Smoke uses `50 / 20 / 20`.  Official `validation` must not
be read during Pilot and is reserved for the formal final test.  Split by
`scenario_id` before generating any sensor variants.  Freeze and SHA-256 every
manifest.

## Sensor layout and nominal parameters

| Node | Role / model | Local position (m) | Noise / rate |
|---|---|---:|---|
| T1 | GPS 2D posterior | n/a | sigma x=y=3.0 m; 10 Hz |
| T2 | range-bearing Radar posterior | (80, -50) | sigma r=2.5 m, sigma bearing=0.6 deg; 5 Hz |
| T3 | range-bearing Radar posterior | (-60, 70) | sigma r=3.0 m, sigma bearing=0.8 deg; 5 Hz |
| E1 | AOA-only evidence | (90, 65) | sigma bearing=1.0 deg; 5 Hz |
| E2 | UWB range-only evidence | (-85, 50) | sigma range=1.5 m; 10 Hz |

The state is `[px, py, vx, vy]`.  All three independent EKFs use a constant
velocity transition computed from the timestamp delta and acceleration process
noise sigma = 2.0 m/s².  Initialise velocity to zero with variance 36.0;
initial GPS position covariance is diag(9, 9), while initial radar position
covariance is the Jacobian-propagated measurement covariance floored at a
minimum position eigenvalue of 1.0 m².  Do not estimate initial velocity from
future samples.

T1/T2/T3 produce posterior state and covariance. E1/E2 are evidence-only and
must never be represented as fusion posteriors. Pair features must remain
posterior-specific.

## Pilot training faults

Use exactly one principal condition per training trajectory:

| Condition | Probability |
|---|---:|
| Nominal | 60% |
| Bias ramp | 15% |
| Reported covariance mismatch | 10% |
| Dropout | 10% |
| Time delay | 5% |

For bias ramp, select a posterior with probability 0.70 or evidence source with
probability 0.30. Onset is uniformly 2.0--4.0 s, ramp duration 1.5--3.0 s, then
the bias persists. GPS terminal bias is 4--8 m in a random direction; Radar is
either 4--8 m range or 0.4--0.8 deg bearing bias; AOA is 0.5--1.0 deg; UWB is
2--5 m.  Signs are random where applicable.

Dropout onset is 2.0--8.0 s and duration 0.5--1.5 s. A posterior EKF performs
prediction without the measurement update; its measurement node is invalid but
the posterior remains. Evidence dropout sets its evidence mask to zero and does
not reuse a prior measurement.

Time delay is exactly one of 1, 2, or 3 measurement steps. At `k`, use only
`z[k-delay]`; early delayed samples are invalid, and no delay compensation or
future backfill is allowed.

For covariance mismatch, retain EKF internals and set only the fusion-facing
reported covariance to `alpha * P_internal`, where alpha is one of 0.25, 0.50,
2.00, or 4.00. Features and baselines at the fusion centre must receive the
reported covariance.

For evidence-noise training variation, jointly scale actual and reported
measurement covariance with beta in [0.8, 1.5]. The later formal sweep is beta
in {0.5, 1.0, 1.5, 2.0, 3.0}.

## Decision Pilot and gates

Decision Pilot seeds: model `[0, 1, 2]`; measurement `[100, 101, 102]`.
All compared methods receive identical posteriors, sensor implementations,
measurement seeds, manifests, and test samples. Test data cannot be used for
early stopping or hyperparameter selection.

Engineering gates (all mandatory): successful-scene rate >= 99%, finite output
rate = 100%, positive covariance diagonal rate = 100%, SPD covariance rate >=
99.9%, causal-leakage tests pass, and manifest overlap count = 0.

GO also requires:

- versus CI under nominal data: position-RMSE degradation <= 2%, P95 degradation
  <= 3%, and no motion group RMSE degradation > 7%;
- versus CI under mixed quality: position-RMSE improvement >= 3% or P95
  improvement >= 5%;
- full PEFNet versus without-external-evidence: position-RMSE improvement >= 1%
  or P95 improvement >= 2%; and
- stable, finite NEES/coverage/NLL; PEFNet NLL may not worsen versus CI by more
  than 5%, and 95% coverage may not be materially worse than CI.

Conditional GO permits only the frozen reduced formal protocol when engineering
gates pass, nominal RMSE degradation is 2--5%, and mixed-quality improvement is
in the stated borderline ranges (RMSE 1--3% or P95 2--5%) with interpretable
evidence behaviour.  NO-GO includes nominal RMSE degradation > 5%, no mixed
RMSE and P95 improvement, full PEFNet persistently below its evidence ablation,
unstable covariance, seed dependence, or a main motion class degrading > 15%.

## Resource cap

Pilot: at most 8 GPU hours, 8 CPU preprocessing hours, and one wall-clock day.
Formal work is allowed only after GO. Its preferred budget is 12--30 GPU hours,
with a hard maximum of 36 GPU hours and four days wall time.

