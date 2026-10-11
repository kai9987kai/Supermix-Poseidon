# Verified learning foundation Implementation Plan

> **For agentic workers:** Use subagent-driven development for the independent tasks and root integration/review. Checkboxes track completed evidence, not intentions.

**Goal:** Repair demonstrated verification/runtime defects and connect a useful, explicitly selected trained predictor to reproducible CPU experiments.

**Architecture:** Extend the existing research ORACLE, give both cognitive coordinators an explicit trained-artifact interface, and retain frozen Tidal fallback whenever predictive evidence is unavailable. Repair exact transition identity and episode replay before evaluating candidates. Subsequent learned memory and transfer depend on these measured foundations.

**Tech Stack:** Python 3.12, CPU PyTorch, pytest, existing TidePool and research modules; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-11-learning-transfer-design.md`.

## Global constraints

- One experiment worker, two Torch threads, explicit finite budgets; no automatic model promotion.
- Preserve historical checkpoints, original receipts, concurrent changes and `tidepool-v1` dynamics.
- Keep fit, selection, calibration and final evaluation episodes disjoint; report missing death coverage honestly.
- Use source/model identities and raw trajectories; a recomputable checksum is not semantic verification.
- The user approved advancement based on the reviewed design. Execution uses scoped parallel tasks in the existing feature branch and root integration; no repeated design permission is needed for these reversible changes.

## Review focus

- Resealed but contradictory receipt fields must fail semantic verification.
- Noisy terminal observations must not change true death/truncation outcomes.
- Missing, malformed or unsupported model artifacts must not silently substitute random heads.
- Cancelled/rejected candidate work must not restore or commit unrelated working files.
- Old artifact protocol versions must retain explicit legacy interpretation rather than silently adopting new data semantics.

### Task 1: Research artifact and data contracts

Own `poseidon/research/oracle_training.py`, `contracts.py`, `candidate_evaluation.py`, `oracle_model.py`, and `tests/test_oracle_research.py`.

- [ ] Add failing resealed-claim, nonfinite-weight, missing-split and protocol-mismatch tests.
- [ ] Bind all duplicated receipt claims to artifacts; validate finite tensor/config/calibration structure and complete partitions.
- [ ] Add an explicit versioned whole-episode behavior mixture with failure coverage, per-action exposure and identical ID calibration/test generation. Preserve legacy artifact replay interpretation.
- [ ] Retain fixed episode-bootstrap draws and report episode/member exposure; add a linear action-conditioned predictive baseline.
- [ ] Run focused tests and inspect the recorded protocol before fitting the bounded pilot.

Interface: `fit_oracle(...) -> dict`, `load_oracle_artifact(path) -> dict`, `_load_model(payload) -> OracleEnsemble`, `verify_fit_receipt(path, replay_data=True) -> dict`; retain existing calls and make new protocol options explicit.

### Task 2: Exact cognitive trajectories and verification

Own `poseidon/cognitive_agent.py`, `cognitive_experiment.py`, `worldlab.py` and their dedicated tests. Root will integrate the artifact adapter only after handoff.

- [ ] Add failing empty/resealed receipt, terminal/death, reused-environment snapshot and prediction-time alignment tests.
- [ ] Generate/cache predictions for the executed action before matching its realized successor; reset all episode-local state.
- [ ] Capture snapshots after reset, preserve sensor RNG in POMDP snapshots, derive true terminal status and final health from the environment.
- [ ] Verify a closed receipt schema, complete unique pairs, bound model/source identity, all transitions and recomputed summaries. Return the existing boolean success contract.
- [ ] Run normal and noisy replay tests plus mutation fixtures.

Interface: `CognitiveAgent.observe_transition(...)`, `.rollout(...)`, `verify_cognitive_receipt(receipt, core_path=None) -> bool`; preserve old callers while rejecting semantically incomplete evidence.

### Task 3: Isolated parameter search

Own `tools/auto_improver.py` and new focused tests.

- [ ] Test that a rejected candidate leaves unrelated dirty files and the checkout's HEAD unchanged.
- [ ] Replace repository mutation/global restore with candidate-local temporary source copies or isolated workspaces; seed proposal randomness and record baseline/candidate identities.
- [ ] Keep selected candidates as inactive artifacts; remove automatic shared-tree commit/promotion. Retain original CLI bounds with explicit deadlines.
- [ ] Report raw fixed-seed outcomes as selection results and use a separate audit seed set for any claimed improvement.
- [ ] Run meaningful isolation and timeout/rejection tests without a long search.

### Task 4: One explicit trained predictor interface

Root owns `poseidon/research/oracle_adapter.py`, `horizon_x.py`, `hyper_agent.py`, runtime/CLI integration and adapter tests, then updates cognitive constructor after Task 2.

- [ ] Implement `ResearchOracleAdapter(artifact_path)` with legacy planner-compatible `forward(latent, observation, actions)` and `imagine_trajectory(...)`, explicit artifact/status metadata, finite input/output checks and support counters.
- [ ] Expose state/reward scales and uncalibrated death-target status; do not manufacture trained depletion heads or probability guarantees.
- [ ] Add `oracle_checkpoint` parameters to both agents and appropriate runtime/CLI paths. Default missing-artifact behavior preserves frozen Tidal actions; untrained heads remain available only as explicit diagnostics.
- [ ] Test direct research/adapter prediction parity, invalid artifact rejection, source/contract mismatch, inactive baseline parity and actual loaded-model decision telemetry.

### Task 5: Bounded experiment and integration review

- [ ] Run fixture and focused suites, then applicable full tests once integration is stable.
- [ ] Fit a fresh CPU pilot using the versioned data protocol and independently evaluate its prediction targets and control behavior; retain failures and inconclusive outcomes.
- [ ] Compare learned prediction with trained/simple baselines, report failure/action exposure, calibrated interval usefulness, support and actual inference work.
- [ ] Perform fresh code review, fix material findings and run `git diff --check` plus source/package checks appropriate to changed runtime paths.
- [ ] Update readiness/audit documentation with actual completed work and outstanding learned-memory/structural-transfer gates. Do not claim completion of later design stages from foundation tests.

## Execution ledger

- Historical research sources preserved in `outputs/architecture-audit-2026-10-11/source-snapshot` before edits.
- Ruling: preserve and reuse the existing dirty feature checkout — the changes are from this audit and scoped file ownership avoids overwrite; new unrelated commits require reinspection.
- Ruling: this plan is the prerequisite implementation tranche of the larger design. Later memory/transfer/adaptation stages remain governed by their evidence gates rather than being presented as completed by this foundation.
