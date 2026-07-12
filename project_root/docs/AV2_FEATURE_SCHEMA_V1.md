# AV2 Pilot Feature Schema v1

Schema identifier: `av2_pilot_feature_v1`.  This is the frozen feature-shard
boundary for AV2 Pilot Protocol v1.0.  It applies after causal sensor/EKF
processing and before a shard is serialized or supplied to PEFNet.  It does
not alter the legacy Phase1R feature builders.

| Field | Shape | Semantics |
|---|---:|---|
| `post_feat` | `[T, 3, 9]` | T1/T2/T3 posterior-node features. Indices 0:4 are scaled state, 4:8 are `log1p` posterior covariance diagonal, and 8 is the posterior-valid mask. |
| `post_mask` | `[T, 3]` | Exact binary copy of `post_feat[..., 8]`. |
| `meas_feat` | `[T, 3, 18]` | The measurement-side feature for the T1/T2/T3 local updates. Indices 0:2 are whitened innovation, 2 log-NIS, 3:5 reported-R diagonal, 5:8 relative prediction geometry, 8:12 type one-hot, 12 mask, 13:14 rolling diagnostics, 15:16 peer diagnostics, 17 bias term. |
| `meas_mask` | `[T, 3]` | Exact binary copy of `meas_feat[..., 12]`. A dropped measurement is zeroed even while its posterior may remain valid via prediction. |
| `evidence_feat` | `[T, 2, 16]` | E1 AOA-only and E2 UWB-range-only evidence features, respectively. They are measurement/evidence nodes, never posteriors. Index 13 is its mask. |
| `evidence_mask` | `[T, 2]` | Exact binary copy of `evidence_feat[..., 13]`. |
| `mp_pair_feat` | `[T, 3, 5, 8]` | Posterior × (T1/T2/T3/E1/E2) directed relation features. Channels are self-track, other-track, evidence, log residual, bounded local log residual, local unit residual score, posterior-valid, measurement-valid. Channels 3–5 depend only on the current posterior–evidence pair; no peer ranking or cross-posterior centering is permitted. |
| `target` | `[T, 4]` optional | Local-frame reference `[px, py, vx, vy]`; supervision/evaluation only. |
| `timestamps_ns` | `[T]` optional | Strictly increasing integer timestamps. |
| `eval_mask` | `[T]` optional | Timestamp-derived post-warmup reporting/loss mask. |

## Non-negotiable invariants

- Source ordering is fixed: posterior sources `T1,T2,T3`; evidence sources
  `E1,E2`.  E1/E2 must never be inserted into `post_feat` or `post_mask`.
- Every stored number is finite. Every mask is exactly zero or one. Feature
  rows with a zero node mask are all zero, preventing stale delayed/dropout
  values from being read as current measurements.
- Pair channels 0:3 have the canonical time-invariant role layout. Pair
  channel 6 repeats the owning posterior mask and channel 7 repeats the
  paired measurement/evidence mask. Track-pair relation channels 3:6 are
  reserved as zero in v1.
- Evidence pair relation channels 3:6 must be posterior-specific. The
  validator rejects a shard where an evidence relation is broadcast to every
  eligible posterior throughout the shard. Builder unit tests should use
  `pair_specificity="per_timestep"` for the strongest check.

Use `data.av2.feature_schema.validate_av2_feature_shard()` immediately before
writing a shard and after reading it.  The validator accepts only assembled
arrays; it does not load AV2 files, call an EKF, or inspect model code.
