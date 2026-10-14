# NEXUS-Ω phaseworld v2 viability and shock-recovery study

**Date:** 14 October 2026  
**Status:** Synthetic 64-step robustness study completed; exploratory, no controller promotion  
**Receipts:** [32-step refinement](../../runs/nexus_omega/phaseworld_v2_viability_assay_refined_20261014.json), [64-step robustness replication](../../runs/nexus_omega/phaseworld_v2_robustness_h64_20261014.json), [128-step robustness extension](../../runs/nexus_omega/phaseworld_v2_robustness_h128_20261014.json)  
**Runner:** [run_nexus_phaseworld_v2_viability_assay.py](../../tools/run_nexus_phaseworld_v2_viability_assay.py)

## Question

The original `POMDPPhaseWorld` drains hydration on every action and offers no replenishment action. A three-tick pulse-memory controller therefore still dies from starvation, making survival an invalid outcome for comparing controllers. The versioned `nexus-phaseworld-v2` environment preserves the eight observation channels and four actions, while adding +0.12 hydration to Rest before the unchanged -0.05 basal drain. The original environment and its receipts remain untouched.

The first v2 run exposed two model/controller errors: the inherited prior predicted health recovery from storm Shelter even though the environment provides none, and the safety fallback always chose Shelter even when Rest had the higher lower-confidence score and replenished hydration. After those were corrected, a 32-step run found that all aligned agents survived, but an initially hidden storm could leave health at 0.2 and energy near zero. This motivated the next, separately measured change: a v2-only latent-regime update when a realized health loss is substantially worse than predicted.

## V2 changes

- The v2 factored prior removes the inherited +0.05 Shelter health gain during storms and models Rest's +0.12 hydration recovery.
- The safety planner keeps its original Shelter fallback by default. The v2 agent opts into a pair of fallback actions (Shelter, Rest) and selects the action with the higher model lower-confidence score.
- After a transition, the v2 agent treats an unexpected health residual at or below -0.25 as a one-step storm witness, raising latent storm duration to at least one. It then records this event in the episode trace. The v1 profile does not use this update.

These are explicit structured heuristics. The neural residual ensemble remains randomly initialized; this is not a trained world model. The lower-confidence score is an uncalibrated heuristic and carries no conformal or safety guarantee.

The design follows portfolio lessons without importing upstream code: prediction-error feedback as a control signal from [GhostInTheMachine](https://github.com/kai9987kai/GhostInTheMachine), matched deterministic interventions from [Causeway](https://github.com/kai9987kai/Causeway) and [MOLT](https://github.com/kai9987kai/MOLT), and explicit separation of evidence admission from promotion as in [TESSERA](https://github.com/kai9987kai/TESSERA).

## Protocol

Both runs used 64 environment seeds, two fixed agent initializations, seven arms, full action/reward/observation/vital traces, and exact deterministic replay of every recorded episode. For each environment seed, the v1 structured prior and v2-aligned prior shared the same agent initialization and environment seed. Controls were the legacy v1 trace-memory policy, v2 reactive cue-only policy, and v2 hand-coded three-tick pulse-memory policy. That pulse-memory policy is a viability positive control, not a learned policy.

| Run | Environment seeds | Agent initializations | Horizon | Replay |
|---|---|---|---:|---:|
| Initial v2 refinement | 99201–99264 | 55101, 55102 | 32 | 448/448 |
| Shock-recovery replication | 99301–99364 | 55201, 55202 | 64 | 448/448 |
| Long-horizon extension | 99401–99464 | 55301, 55302 | 128 | 448/448 |

The h64 and h128 runs were fresh seed blocks after the shock update. Neither reuses seeds from the 32-step experiment or from the other long-horizon block.

## Results

| h64 arm | Survival | Mean reward | Mean terminal health | Mean terminal energy | Mean terminal hydration |
|---|---:|---:|---:|---:|---:|
| Legacy v1 trace-memory on v1 | 0/64 | 3.892 | 0.000 | 0.950 | -0.170 |
| V2 reactive cue-only | 0/64 | -1.131 | 0.000 | 0.893 | 0.728 |
| V2 hand-coded pulse trace | 64/64 | 14.009 | 0.756 | 0.950 | 0.881 |
| V2 NEXUS, v1 prior, init 55201 | 0/64 | 5.384 | 0.000 | 0.499 | -0.077 |
| V2 NEXUS, v1 prior, init 55202 | 0/64 | 5.384 | 0.000 | 0.499 | -0.077 |
| V2 NEXUS, aligned prior, init 55201 | 64/64 | 15.858 | 0.788 | 0.905 | 0.457 |
| V2 NEXUS, aligned prior, init 55202 | 64/64 | 15.940 | 0.788 | 0.913 | 0.421 |

At h128, the same aligned profile again survived every episode at both fresh initializations. Its mean reward was 30.881 (init 55301) and 30.773 (55302); mean terminal health was 0.788 for both, terminal energy 0.883/0.895, and terminal hydration 0.416/0.459. Across all h128 episodes, minimum health was 0.60, minimum energy 0.75, and minimum hydration 0.24/0.30. The per-episode hydration change averaged -0.024 and +0.027 across steps 65–128, after a larger first-half drawdown of -0.560 and -0.568. The shock update fired in 34 of 64 environment seeds at each initialization. The v1-prior comparison and cue-only control survived 0/64, while the hand-coded pulse-memory control survived 64/64; all 448 traces replayed exactly. Paired v2-minus-v1 reward changes were +25.655 (95% interval [24.988, 26.368]) and +25.548 ([24.925, 26.222]). This supports a stable resource trajectory over the measured second half, not indefinite viability.

The v2 agent completed all 64 steps in every episode. Across both initializations, the lowest observed health was 0.60, the lowest energy was 0.75, and the lowest hydration was 0.26–0.32. The unexpected-health update activated in 34 of 64 environment seeds for each initialization; the two initializations agreed on which episodes triggered it. The paired v2-minus-v1 reward changes were +10.474 (95% environment-seed bootstrap interval [9.837, 11.164]) and +10.556 ([9.930, 11.225]). Each initialization had 64/64 pairs in which only the aligned v2 prior survived. These are two fixed initializations on one synthetic environment, not independent model-family replications.

The earlier h32 refinement, before the shock update, also achieved 64/64 survival for each aligned initialization. It revealed why horizon and resource margins matter: its minimum health fell to 0.20, one initialization reached energy 0.05, and the other reached 0.28 minimum hydration. The shock-recovery rule raised the h64 minimum health to 0.60 and avoided near-empty energy and hydration on the new seed block.

## Interpretation and limits

The key result is not the 100% simulator survival rate in isolation. It is that an explicit model-error event can update a hidden hazard belief quickly enough to protect the next action, while a versioned viable benchmark distinguishes that behavior from a cue-only reactive controller. The evidence is bounded to the exact synthetic transition rules, two fixed random initializations per block, 64 fresh seeds per block, and horizons up to 128 steps. The v2 agent still ends with lower average hydration than the hand-coded positive control, and the controller has not been tested under altered storm durations, observation noise, resource schedules, or adversarial transition changes.

The assay does not establish real-world safety, biological memory, learned causal inference, or independent replication. It does not promote the v2 profile for deployment. The complete episode traces, source hashes, replay counts, and receipt digests are retained in the two machine-readable receipts.
