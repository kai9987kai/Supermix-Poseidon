# NEXUS-Ω phaseworld v2 viability and shock-recovery study

**Date:** 14 October 2026  
**Status:** Synthetic 64-step robustness study completed; exploratory, no controller promotion  

**Receipts:** [32-step refinement](../../runs/nexus_omega/phaseworld_v2_viability_assay_refined_20261014.json), [64-step robustness replication](../../runs/nexus_omega/phaseworld_v2_robustness_h64_20261014.json), [128-step robustness extension](../../runs/nexus_omega/phaseworld_v2_robustness_h128_20261014.json), [five-tick stress run](../../runs/nexus_omega/phaseworld_v2_storm5_h128_20261014.json), [source-bound repeat](../../runs/nexus_omega/phaseworld_v2_storm5_h128_sourcebound_20261014.json), [eight-tick adaptive stress run](../../runs/nexus_omega/phaseworld_v2_storm8_h128_adaptive_20261014.json)
**Runner:** [run_nexus_phaseworld_v2_viability_assay.py](../../tools/run_nexus_phaseworld_v2_viability_assay.py)

## Question

The original `POMDPPhaseWorld` drains hydration on every action and offers no replenishment action. A three-tick pulse-memory controller therefore still dies from starvation, making survival an invalid outcome for comparing controllers. The versioned `nexus-phaseworld-v2` environment preserves the eight observation channels and four actions, while adding +0.12 hydration to Rest before the unchanged -0.05 basal drain. The original environment and its receipts remain untouched.

The first v2 run exposed two model/controller errors: the inherited prior predicted health recovery from storm Shelter even though the environment provides none, and the safety fallback always chose Shelter even when Rest had the higher lower-confidence score and replenished hydration. After those were corrected, a 32-step run found that all aligned agents survived, but an initially hidden storm could leave health at 0.2 and energy near zero. A 128-step run extended nominal-horizon survival, then a five-tick storm stress pilot showed that fixed one- and three-tick surprise persistence still failed. The current v2 refinement estimates storm duration relative to the observed pulse and permits tightly screened resupply when the model believes the storm has cleared.

## V2 changes

- The v2 factored prior removes the inherited +0.05 Shelter health gain during storms and models Rest's +0.12 hydration recovery.
- The safety planner keeps its original Shelter fallback by default. The v2 agent opts into a pair of fallback actions (Shelter, Rest) and selects the action with the higher model lower-confidence score.
- The v2 agent tracks the number of steps since the transient pulse. A health residual at or below -0.25 beyond the expected storm tail increases its duration estimate up to five ticks and retains only the inferred remaining duration in the latent regime.
- The guard admits Forage while health is threatened only when energy is below 0.50, hydration exceeds 0.25, cue/memory/regime all indicate calm, and the model predicts stable health plus energy recovery. The decision marks this `recovery_admission` explicitly.
- The v1 profile does not use duration inference or calm-recovery admission.

These are explicit structured heuristics. The neural residual ensemble remains randomly initialized; this is not a trained world model. The lower-confidence score is an uncalibrated heuristic and carries no conformal or safety guarantee.

The design follows portfolio lessons without importing upstream code: prediction-error feedback as a control signal from [GhostInTheMachine](https://github.com/kai9987kai/GhostInTheMachine), matched deterministic interventions from [Causeway](https://github.com/kai9987kai/Causeway) and [MOLT](https://github.com/kai9987kai/MOLT), and explicit separation of evidence admission from promotion as in [TESSERA](https://github.com/kai9987kai/TESSERA).

## Protocol

Both runs used 64 environment seeds, two fixed agent initializations, seven arms, full action/reward/observation/vital traces, and exact deterministic replay of every recorded episode. For each environment seed, the v1 structured prior and v2-aligned prior shared the same agent initialization and environment seed. Controls were the legacy v1 trace-memory policy, v2 reactive cue-only policy, and v2 hand-coded three-tick pulse-memory policy. That pulse-memory policy is a viability positive control, not a learned policy.

| Run | Environment seeds | Agent initializations | Horizon | Replay |
|---|---|---|---:|---:|
| Initial v2 refinement | 99201–99264 | 55101, 55102 | 32 | 448/448 |
| One-step shock update replication | 99301–99364 | 55201, 55202 | 64 | 448/448 |
| One-step shock update extension | 99401–99464 | 55301, 55302 | 128 | 448/448 |
| Pulse-age duration stress | 99701–99764 | 55601, 55602 | 128 | 576/576 |

All blocks use fresh seed ranges except the source-bound repeat, which reran the identical 99701–99764 stress block and compared every stored episode. The first three receipts predate the current pulse-age estimator; their source hashes identify those earlier revisions. The final stress receipt binds the current agent, planner, decision contract, memory implementation, environments, and runner, and includes fixed-three-tick, duration-matched-five-tick, and privileged oracle controls.

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

### Five-tick hidden-storm stress

The stress run changed the storm schedule from 3 of every 6 actions to 5 of every 6, while retaining the v2 hydration mechanics and a 128-step horizon. The aligned NEXUS profile survived 64/64 seeds at both initializations. Both runs had mean reward 29.070, mean terminal health 0.294, mean terminal energy 0.607, and mean terminal hydration 0.471. The lowest observed health/energy/hydration were 0.20/0.30/0.28 and 0.20/0.35/0.28. The duration estimate reached its configured maximum of five ticks in all episodes; each initialization recorded 113 shock events and 672 screened calm-recovery admissions across its 64 episodes.

| Five-tick storm arm | Survival | Mean steps | Mean reward |
|---|---:|---:|---:|
| V2 NEXUS, aligned prior, init 55601 | 64/64 | 128.0 | 29.070 |
| V2 NEXUS, aligned prior, init 55602 | 64/64 | 128.0 | 29.070 |
| V2 NEXUS, v1 prior, each initialization | 0/64 | 17.5 | 2.434 |
| Fixed 3-tick pulse trace | 0/64 | 6.0 | -2.420 |
| Duration-matched 5-tick pulse trace | 35/64 | 71.4 | 6.039 |
| Privileged oracle-state controller | 64/64 | 128.0 | 13.843 |

The oracle chooses Rest during the simulator's hidden storm phase and Forage otherwise. It checks that this harsher environment admits a viable policy; its privileged information makes it an upper bound, not an agent baseline. The duration-matched pulse controller still fails when episodes begin partway through a storm before the first cue. In contrast, the NEXUS candidate completed all episodes and exceeded the oracle's reward, but its mean terminal health is only 0.294 and its minimum health is 0.20. This demonstrates performance inside the simulator, not a robust safety margin. The paired candidate-minus-v1-prior reward change was +26.635 for each initialization (95% environment-seed bootstrap intervals [26.424, 26.845] and [26.424, 26.843]); all 64 pairs favored the candidate on survival. Every one of 576 complete trajectories replayed exactly with zero failures. The source-bound repeat produced identical episode records and arm summaries, and its source hashes match the current files.

### Eight-tick hidden-storm adaptive stress

An eight-tick storm schedule (8 ticks storm out of 10-tick cycle, horizon 128, seeds 100201–100264, agent initializations 56001 and 56002) tested the adaptive hazard guard and pulse latch under severe environmental duress. The aligned NEXUS-Ω agent achieved 64/64 survival across both initializations (100% viability) with 0 replay failures across all 576 complete trajectories. In contrast, the legacy v1 prior survived 0/64 (survival delta +1.0, paired sign test p = 1.08e-19, 95% bootstrap CI [1.0, 1.0]). The paired reward delta was +3.232 (95% CI [3.070, 3.391]) for initialization 56001 and +3.180 (95% CI [3.030, 3.330]) for initialization 56002.

## Interpretation and limits

The evidence now includes three separate mechanisms: a versioned sustainable environment, pulse-age inference from health prediction error, and a resource-recovery admission that is conditioned on the inferred storm state. The five-tick stress result suggests the second and third mechanisms address two distinct failures: hazard persistence and energy depletion from over-sheltering. Evidence remains bounded to one synthetic storm schedule, two fixed initializations, 64 seeds in the stress block, and a 128-step horizon. Health frequently falls to 0.20; longer storms, different intervals, observation noise, altered resource schedules, and adversarial changes remain untested.

The assay does not establish real-world safety, biological memory, learned causal inference, or independent replication. It does not promote the v2 profile for deployment. The complete episode traces, source hashes, replay counts, and receipt digests are retained in the machine-readable receipts linked above.
