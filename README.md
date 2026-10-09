# Supermix Poseidon

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Supermix--Poseidon-yellow)](https://huggingface.co/Kai9987kai/Supermix-Poseidon)
[![GitHub](https://img.shields.io/badge/GitHub-Supermix--Poseidon-blue)](https://github.com/kai9987kai/Supermix-Poseidon)

A local experimental model system that combines a newly trained **Tidal core**, a compact pretrained language model, explicit carrier memory, exact maths tools, and a deterministic survival environment. It produces basic PNG images, short animations, and OBJ/glTF models from a shared learned scene representation.

The model release is hosted at [**Kai9987kai/Supermix-Poseidon**](https://huggingface.co/Kai9987kai/Supermix-Poseidon). It contains the trained Tidal checkpoints, separate pretrained language component, inactive candidate LoRA, experimental Atlas, runtime source and evaluation receipts. See [release instructions and boundaries](docs/HUGGINGFACE_RELEASE.md); the Hub manifest identifies the exact published source revision.

Built for the supplied Snapdragon Windows PC: CPU only, bounded threads, no paid APIs. The 24 linked projects inform the architecture through documented source review; their code and checkpoints are not indiscriminately merged.

## METAMORPH, MOLT, NexusFlow & Chronos — v0.9 release

The **v0.9 release** advances the experimental architecture to a new frontier, synthesizing life-stage morphogenesis, directed causal potential flux, and periodic frame beaconing without altering frozen Tidal neural weights:

1. **MOLT: Metamorphic Online Life-Stage Transmutation & Morphogenesis** ([`docs/MOLT_DESIGN.md`](docs/MOLT_DESIGN.md)):
   - **Life-Stage Specialization**: Implements `larval` (acute local foraging, high sensory gain), `pupa` (quiescent diapause, chitinous exposure hardening, internal structural remodeling), and `imago` (expanded receptive horizon, high-speed dispersal travel).
   - **Ecdysis & Cryptographic Exuviae**: Accumulates developmental biomass and triggers ecdysis, shedding tamper-proof SHA-256 exuvia artifacts.

2. **NexusFlow: Directed Acyclic Causal Flux & Superposition Routing** ([`docs/NEXUSFLOW_DESIGN.md`](docs/NEXUSFLOW_DESIGN.md)):
   - **Continuous Causal Flux**: Solves hydraulic potential gradients $\Phi_{ij} = \exp(-d_{ij}/\tau) \cdot \Delta P_{ij}$ over spatial replenishment corridors.
   - **Coherent Superposition Interference**: Evaluates wave interference $I(a, b) = 2\sqrt{V_a V_b}\cos(\Delta\theta)$ between biomimetic compass headings and topological potential vectors, suppressing destructive bifurcations ($\cos\Delta\theta < -0.2$).

3. **Chronos & Ghost: Periodic Frame Synchronization & Phantom Trace Introspection** ([`docs/CHRONOS_DESIGN.md`](docs/CHRONOS_DESIGN.md)):
   - **Periodic Beacon Clock**: Emulates hardware-level cyclic carrier beacon pulses ($T_{\text{sync}} = 8$ ticks) synchronizing macro-action expiration and concept consolidation.
   - **Ghost Trace Auditor**: Quantifies counterfactual ontological divergence via the Spectral Divergence Index (SDI).

```powershell
# Run the 3-arm paired METAMORPH evaluation benchmark
python -m poseidon metamorph-experiment --episodes 4 --max-steps 48 --scarcity 2.5

# Verify METAMORPH experiment receipt offline
python -m poseidon verify-metamorph outputs/metamorph_experiments/RECEIPT.json
```

## AURA, Tessera & Mnemorph — v0.8 release

The **v0.8 release** introduces three groundbreaking biologically and architecturally inspired innovations synthesizing cutting-edge principles across biomimetic insect navigation, lineage-ratified macro-actions, and topological memory archaeology without altering the frozen Tidal neural weights:

1. **AURA Biomimetic Central Complex Controller**:
   - **Central Complex Ring Attractor Compass**: 16-wedge recurrent neural network with local excitation, global inhibition ($W_{ij} = \cos(\theta_i - \theta_j)$), and angular velocity integration.
   - **Bump Coherence & Involuntary Optomotor Reflex**: Continuously estimates population vector average (PVA) heading $\hat{\theta}$ and coherence $R \in [0, 1]$. When bump coherence drops below 0.35, reflexively triggers optomotor stabilization saccades to recover heading lock.
   - **Sparse Kenyon Cell Mushroom Body Neuropil**: 128 virtual Kenyon cells with strict $k=8$ anterior paired lateral (APL) GABAergic inhibitory winner-take-all gain control (6.25% sparsity).
   - **Tri-Drive Homeostatic Arbitration**: Computes continuous metabolic, fatigue, and threat drives to arbitrate descending motor channels (`harvester`, `sentinel`, `escaper`).

2. **Tessera Lineage-Ratified Macro-Action Commons**:
   - **Dual-Lineage Independent Ratification**: Quarantines single-lineage candidate action sequences, promoting them into globally executable `OP_*` opcodes only upon independent dual-lineage confirmation with $\Delta R > 0$.
   - **Finite-Domain Semantic Safety Assertions**: Verifies action validity, sequence length ($2 \le L \le 6$), and stamina exhaustion limits before ratification.
   - **Epoch Expiration**: Prunes stale opcodes lacking multi-epoch reinforcement.

3. **Mnemorph Archaeological Memory & Lesion Assays**:
   - **Associative Knowledge Graph**: Maps entity and fact co-occurrences with degree centrality metrics.
   - **Typed Structural Lesion Assays**: Systematically evaluates catastrophic forgetting under `intact`, `hub_lesion` (top 10% degree hubs deleted), `periphery_lesion` (degree-1 nodes deleted), and `associative_regrowth` (Hebbian transitive recovery).
   - **Hub Vulnerability Ratio (HVR)**: Formalizes topological vulnerability and associative self-repair.

```powershell
# Run the 5-arm paired AURA biomimetic evaluation benchmark
python -m poseidon aura-experiment --seed 150000001 --episodes 4 --max-steps 64 --scarcity 2.5

# Verify AURA experiment receipt offline
python -m poseidon verify-aura outputs/aura_experiments/d9e0d76bd4812c03-b489e9702cff.json

# Run Tessera macro-action ratification study
python -m poseidon tessera-experiment

# Verify Tessera receipt offline
python -m poseidon verify-tessera outputs/tessera_experiments/RECEIPT.json

# Run Mnemorph archaeological memory lesion assays
python -m poseidon mnemorph-experiment

# Verify Mnemorph receipt offline
python -m poseidon verify-mnemorph outputs/mnemorph/RECEIPT.json
```

See [AURA design](docs/AURA_DESIGN.md), [AURA results](docs/AURA_RESULTS.md), [Tessera design](docs/TESSERA_DESIGN.md), [Tessera results](docs/TESSERA_RESULTS.md), and [Mnemorph design](docs/MNEMORPH_DESIGN.md).

## Odysseus & MCO — v0.7 release

The **v0.7 release** introduces two foundational innovations advancing empirical cognition and causal memory verification without altering the frozen Tidal neural weights:

1. **Odysseus Empirical Cognitive Navigator**:
   - **Online Bayesian Replenishment**: Replaces heuristic recovery rates with conjugate Gaussian posterior tracking ($\mu_\lambda, \sigma^2_\lambda$) updating on patch revisits. Evaluates candidate waypoints via conservative 95% Lower Confidence Bounds (LCB).
   - **Empirical Macro-Action Transition Distributions**: Directly models $\hat{P}(P_{\text{dest}} \mid P_{\text{src}}, a)$ for transit actions (`explore` $a=4$, `flee` $a=5$) to discount unreachable patches.
   - **Physiological Pre-Transit Stamina Guard**: Hard invariant overriding travel proposals to rest ($a=0$) whenever stamina drops below 0.20, preventing acute exhaustion mortality.
   - **Calibrated Error Radius Hurdle**: Disjoint calibration measures empirical overestimation radius $\varepsilon_{\text{odysseus}}$, accepting navigational overrides only when net utility clears empirical uncertainty.

2. **Memory Carrier Observatory (MCO)**:
   - **Full Causal Factorial Analysis**: Evaluates all 16 inclusion/exclusion combinations across four memory carriers (`episodic`, `body`, `habitat`, `social`).
   - **Three Falsifiable Negative Controls**: Compares intact retrieval against `shuffled` (carrier permutation), `irrelevant` (topical distractor injection), and `erased` (zero-memory ablation) across 1,024 independent trials.
   - **Robust Marginal Contributions**: Proves strictly positive marginal gains for all carriers ($\Delta F_1 \ge +0.22$) with query-invariant leave-one-task-out (LOTO) sensitivity bounds.

```powershell
# Fit Odysseus empirical calibration radius
python -m poseidon fit-odysseus --train-episodes 4 --calibration-episodes 4 --max-steps 32 --scarcity 2.5

# Run multi-arm paired benchmark
python -m poseidon odysseus-experiment --episodes 4 --max-steps 64 --scarcity 2.5

# Weightless replay verification
python -m poseidon verify-odysseus outputs/odysseus_experiments/1521d677b6f2f75b-c4035a1005d3.json

# Run Memory Carrier Observatory factorial benchmark
python -m poseidon mco-experiment

# Verify MCO receipt offline
python -m poseidon verify-mco outputs/mco_experiments/RECEIPT.json
```

See [Odysseus design](docs/ODYSSEUS_DESIGN.md), [Odysseus empirical results](docs/ODYSSEUS_RESULTS.md), [MCO design](docs/MCO_DESIGN.md), and [MCO factorial results](docs/MCO_RESULTS.md).

## Helm — v0.6 release

The new opt-in **Helm** critic independently learns multi-horizon returns from observed history and realized Tidal prediction errors. Observation-only, innovation-erased and training-error-yoked controls separate useful history from added complexity. Disjoint episode partitions cover fitting, selection, calibration and evaluation; both 4- and 16-step margins must support a calibrated intervention.

```powershell
python -m poseidon fit-helm --train-episodes 12 --selection-episodes 8 --calibration-episodes 8 --anchors 12 --max-steps 96
python -m poseidon helm-experiment --seed 123000001 --episodes 4 --max-steps 64 --scarcity 2.5
python -m poseidon verify-helm outputs/helm_experiments/349465c522f435a5-2101a2f756ec.json
python -m poseidon world --planner helm --seed 42 --max-steps 96 --scarcity 2.5
```

The workbench adds durable cancellable CPU experiments, full action/trajectory receipts and descriptive benefit/harm versus override coverage. In paired empirical benchmarks across moderate ($scarcity=2.5$) and severe ($scarcity=4.0$) environments, an uncalibrated critic aggressively overrode the policy (~61% override rate), leading to catastrophic collapse (0–25% survival and negative rewards). Under Poseidon Helm's calibrated dual-horizon risk gating ($\varepsilon_4=0.1058, \varepsilon_{16}=0.3574$), spurious overrides were cleanly suppressed (0 overrides), fully preserving 100% survival and baseline performance. The default core policy remains selected. See [design and architecture](docs/HELM_DESIGN.md), [empirical measurements and results](docs/HELM_RESULTS.md), and [the refreshed review of all 24 repositories and recent research](docs/research/README.md).

## Trajectory evidence audit — v0.5.1

The new audit reconstructs Odyssey's cognitive map from recorded observations and
actions, checks every controller's fitting partitions, rejects duplicate paired
episodes, and recomputes forecast errors and terminal states. It also branches
selected states into six real **16-step reward trajectories**, recording every
continuation action so their returns can replay without loading model weights.
Regular anchors and every Horizon/Odyssey override are included within a bounded
budget. This separates one-step reserve utility from multi-step episode reward.

```powershell
python -m poseidon verify-evidence outputs/odyssey_experiments/RECEIPT.json
python -m poseidon audit-trajectory outputs/odyssey_experiments/RECEIPT.json --horizon 16 --stride 32
python -m poseidon verify-trajectory outputs/trajectory_audits/AUDIT.json --parent outputs/odyssey_experiments/RECEIPT.json
```

Odyssey now preserves explicitly selected scarcity when it resets at an episode
boundary. The v0.5.0 high-scarcity receipt used a map replenishment setting of 2.5
inside scarcity-4 worlds; that historical limitation is retained and disclosed.
The v0.5.1 release profile includes both corrected suites and their multi-step
receipts alongside the historical evidence. See
[audit design, limits and validation](docs/TRAJECTORY_AUDIT.md).

## Odyssey Atlas — v0.5 release

The new opt-in **Odyssey Atlas** equips Poseidon with an episodic topological cognitive map and navigational memory constructed strictly from 16-d observation streams without simulator cheats or global coordinate leaks.

Key innovations:
- **Zero-Coordinate Spatial Fingerprinting**: Observation-native static terrain and shelter values identify patches ($P_k$) without leaking global grid coordinates.
- **Topological Discovery**: Transitions under action 4 (`explore`) and 5 (`flee`) establish graph edges. Breadth-first distances help score known destinations; the action interface does not directly command a destination or guarantee following a shortest path.
- **Replenishment Dynamics**: Models environmental replenishment ($\lambda_{\text{food}} = 0.007 / \text{scarcity}$, $\lambda_{\text{water}} = 0.014 / \text{scarcity}$) to revisit replenished resource patches once recovered.
- **Physiological Travel Costs & Pre-Transit Rest**: Accounts for multi-hop travel costs ($d \times (0.038 \text{ energy} + 0.037 \text{ hyd} + 0.112 \text{ stam})$) and boosts resting (action 0) when stamina $< 0.20$ before embarking on long journeys.
- **Impending Storm Response**: Shelter scores bias action proposals when storm risk is high and shelter is low; movement remains governed by the simulator's action semantics.

```powershell
python -m poseidon odyssey-fit
python -m poseidon odyssey-experiment --seed 112000001 --episodes 8 --max-steps 128 --scarcity 4.0
python -m poseidon world --planner odyssey --seed 42 --scarcity 4.0 --max-steps 128
python -m poseidon verify-odyssey outputs/odyssey_experiments/RECEIPT.json
```

On the historical v0.5.0 high-scarcity suite, `odyssey_ungated` delivered a **+0.04042 mean reward difference against policy** across four seeds, discovering 12.25–13.0 patches per episode. These observations do not establish superiority or safety. The map-setting limitation above applies to those measurements. See [design](docs/ODYSSEY_DESIGN.md) and [historical results](docs/ODYSSEY_RESULTS.md).

## Horizon Atlas — v0.4 release

The opt-in **Horizon Atlas** estimates multi-step trajectory advantages over 16 steps using undiscounted rewards, with observable resource-depletion guards.

Offline fitting indexes multi-horizon undiscounted returns across 12 training episodes (1,728 transitions) and calibrates empirical advantage error radii across 8 disjoint calibration episodes (1,152 transitions). At runtime, an associative $k$-NN memory retrieves candidate returns and advantages, local resource filters reject depleted harvest actions, and an empirical error gate compares advantages against descriptive radii.

```powershell
python -m poseidon horizon-fit
python -m poseidon horizon-experiment --seed 110000001 --episodes 8 --max-steps 128 --scarcity 4.0
python -m poseidon world --planner horizon --seed 42 --scarcity 4.0 --max-steps 128
python -m poseidon verify-horizon outputs/horizon_experiments/RECEIPT.json
```

On an official 8-seed audit suite at severe scarcity (4.0), lookahead advantage guidance achieved a **+0.01140 mean reward delta over the policy baseline** (with 4 wins, 1 tie, 3 losses, and single-seed gains reaching up to +0.0523). In moderate conditions where policy is near saturation, the calibrated margin gate defers gracefully with zero overrides and 100% survival. See [design](docs/HORIZON_DESIGN.md) and [measured results](docs/HORIZON_RESULTS.md).

## Contrast Atlas — v0.3 development

The new opt-in planner learns which observation channels benefit from residual
memory, then measures prediction errors in **action advantages** against the
incumbent. Four separate episode partitions cover memory fitting, feature
selection, error calibration and evaluation. Each feature selects a shrinkage
weight from 0, 0.25, 0.5, 0.75 and 1 using episode-balanced prediction error;
the base prediction wins ties or improvements below 2%.

After freezing those weights, the paired gate subtracts an empirical error radius
from a proposed action's predicted one-step utility advantage. Calibration uses
the maximum over all challengers and anchors within each episode, followed by a
descriptive 90th percentile across calibration episodes. This is a diagnostic
margin, without a safety or future-coverage guarantee. Support and vital-error
checks still apply. The default planner and selected core weights are unchanged.

```powershell
python -m poseidon contrast-fit
python -m poseidon contrast-experiment --seed 105000001 --episodes 8 --max-steps 128 --scarcity 2.5
python -m poseidon world --planner contrast --seed 42 --scarcity 2.5 --max-steps 128
python -m poseidon verify-contrast outputs/contrast_experiments/RECEIPT.json
```

The nine-controller comparison includes policy, neural MPC, v0.2 Atlas, Contrast,
an absolute-bound gate, memory-erased and unfiltered controls, a heuristic and
deterministic random control. Every visited state has all six actual one-step
branches audited. A separate probe compares base, unfiltered and selected forecasts
at identical policy-arm anchors. Utility gains, forecast errors, episode reward
and survival are reported separately, alongside unequal controller compute costs.
See [design and limits](docs/CONTRAST_DESIGN.md) and
[measured results](docs/CONTRAST_RESULTS.md): selected forecasts improved by
1.93% and 2.50% on two fresh suites, while slightly lower episode reward and
unchanged survival keep Contrast experimental.

Artifacts go to `outputs/contrast/` and `outputs/contrast_experiments/`. The
workbench exposes paired margins, selected feature weights, action audits and
replayable raw receipts. World requests honor scarcity and horizon controls and
retain planner decisions. A source-change guard rejects experiments from a
process that imported older Python files; restart the server after editing code.
The simulator remains byte-identical to v0.2 so its existing receipts can replay.
The published Hugging Face `v0.2.0` snapshot remains available independently of
this local development version.

## Counterfactual Atlas — v0.2

Poseidon now learns an inspectable residual memory around its neural world model.
Offline, it forks each frozen simulator state into all six actions. At inference,
nearby action-conditioned experiences correct the neural prediction, and a gate
checks fitted support and held-out vital-state error before using it. Unsupported
or inaccurate predictions fall back to the existing learned policy.

Each decision exposes all six candidate futures, retrieved record identities,
distances, error radii, policy preference and the selected action. The controller
receives only the 16 observations; simulator snapshots are confined to fitting
and evaluation. This is an experimental synthesis of the portfolio's retrieval,
memory intervention and causal comparison ideas, with established methods beneath
it. See [design](docs/ATLAS_DESIGN.md) and [measured results](docs/ATLAS_RESULTS.md).

```powershell
# Fit a bounded sidecar; does not modify core weights or activate a candidate.
python -m poseidon atlas-fit

# Six controllers, eight paired seeds, complete execution receipts.
python -m poseidon experiment --seed 93000001 --episodes 8 --max-steps 128 --scarcity 2.5

# Explicitly select the atlas for a survival episode.
python -m poseidon world --planner atlas --seed 42

# Independently replay the receipt, without loading any model weights.
python -m poseidon verify-experiment outputs/experiments/RECEIPT.json
```

The workbench has a Counterfactual Atlas panel with bounded paired experiments,
controller comparisons, memory-erased controls, replay and JSON downloads. It
compares the learned policy, neural MPC, Atlas, memory-erased Atlas, scripted
heuristic and deterministic random controller. Different controllers use different
compute budgets; timings are reported rather than treated as a matched budget.

The default fit uses 12 training episodes and six separate calibration episodes,
with 24 anchors per episode and six branches per anchor: 1,728 training and 864
calibration transitions. Branches within an episode are correlated. The fit spans
scarcity levels 1, 2 and 3. Error radii are descriptive held-out quantiles, without
future coverage or calibrated failure-probability guarantees. Experiments reject
seeds used in fitting or calibration. Repeated inspection makes a suite development
evidence. Outcomes cannot promote the candidate automatically.

Artifacts reside in `outputs/atlas/` and `outputs/experiments/`. Atlas loading checks
its checksum, checkpoint identity, model weights, normalized observations and
bounded schema. Receipts commit the exact source files and configuration, preserve
all actions and terminal states, and validate episode outcomes by simulator replay.
Checksums provide integrity diagnostics, not cryptographic authorship. A source-only
checkout needs a fit after preparing the core checkpoint; the Hugging Face snapshot
includes the fitted sidecar bound to its packaged checkpoint.

## Start

The current workspace contains prepared language weights and locally generated run artifacts. Double-click `Run-Poseidon.cmd`, or:

```powershell
cd C:\Users\kai99\Desktop\Poseidon
python -m poseidon serve --port 8787
```

Open **http://127.0.0.1:8787**. The server binds only to the local computer. First language inference loads the model and can take longer than later turns. The workbench shows real training receipts and backend names, and allows explicit memory storage and carrier exclusions.

For a fresh checkout, Python 3.10+ is required; install `python -m pip install -e ".[test,video]"`, then prepare and train as below. Windows ARM machines need a working compatible PyTorch Python installation; this machine uses its existing x64 Python runtime. Dependency installation does not install weights automatically.

## Try it

```powershell
python -m poseidon chat "Explain why seasons change in two sentences."
python -m poseidon math "3*x + 7 = 22"
python -m poseidon image "Create two small cyan spheres with orbit motion."
python -m poseidon video "Make three medium purple cubes with bounce motion."
python -m poseidon mesh "Build one large yellow pyramid with still motion."
python -m poseidon world --seed 42
python -m poseidon status
```

Exports go into `outputs/media/`. PNG, animated GIF, OBJ/MTL and glTF are supported. MP4 is available when `imageio` and `imageio-ffmpeg` are installed. Geometry is generated by software from the neural core's predicted attributes. No cloud image generator is called.

Supported scene vocabulary: cube/sphere/pyramid/cylinder; red/blue/green/yellow/purple/cyan; still/orbit/bounce/spin; one/two/three objects; small/medium/large. Specify all five attributes for a fully defined request. Prompts outside this controlled vocabulary may map to an unintended scene. Predicted probabilities are not calibrated confidence guarantees.

## What is learned

The Tidal core shares a hashed word/bigram encoder, observation projection, four gated experts and two within-decision recurrent updates. It learns five scene attributes, a six-action survival policy, and action-conditioned observation deltas. The dynamics head supports optional neural MPC and Counterfactual Atlas planning. Persistent carrier memory is an external retrieval store, not recurrent neural state.

Conversation comes from **HuggingFaceTB/SmolLM2-135M-Instruct**, pinned at `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The upstream model card reports pretraining on two trillion tokens; those tokens were not trained here. Poseidon also includes a local LoRA training experiment on the last four attention layers. Its candidate stays inactive by default until stronger capability evaluation justifies activation. You can explicitly test it with `--adapter runs/language/adapter`. The base model was already fine-tuned on smol-smoltalk upstream: the local language dev split measures adaptation behavior on that distribution, not previously unseen knowledge or independent generalization.

Maths uses a bounded exact rational solver for arithmetic and linear equations. Other reasoning uses the small language model and may be unreliable. Calculator results are not evidence of unaided neural reasoning. There is no claim that this system surpasses previous Supermix releases.

## Reproduce training

```powershell
# Download pinned language model and one dataset shard; select 25,000 rows.
python -m poseidon.prepare_language --rows 25000

# The joint core uses 60,000 scene prompts and 60,000 simulation transitions.
python -m poseidon.train_core --output runs/tidal --examples 120000 --epochs 3 --batch-size 256 --threads 2

# Resume from the atomic optimizer/RNG checkpoint with the same data contract.
python -m poseidon.train_core --output runs/tidal --examples 120000 --epochs 3 --batch-size 256 --threads 2 --resume

# Bounded CPU language-adapter experiment; records actual exposures/tokens.
python -m poseidon.train_language --steps 256 --examples 2048 --threads 2
python -m poseidon.train_language --steps 256 --examples 2048 --threads 2 --resume

# Frozen semantic scene holdout, paired survival controls and exact-maths tests.
python -m poseidon.evaluate --episodes 24
python -m pytest -q
```

Do not launch a second trainer into the same output directory. A new output directory is required for incompatible data/config changes. Core continuation restores optimizer, sampler and RNG state. Prepared data count differs from examples consumed: three full epochs of 120,000 rows produce 360,000 exposures, while the default language trial consumes 512 examples from 2,048 usable rows. Conversation data contains 22,466 train / 1,294 dev / 1,240 test records after exact initial-prompt grouping. Public-data exact grouping cannot guarantee semantic independence or absence from upstream pretraining.

Core scene splits hold out complete shape/color/motion/count/scale tuples, not just paraphrases. World splits use separate episode-seed namespaces. The procedural curriculum is broad in combinations and observations, but it is not 120,000 independent real-world demonstrations.

## Attach to another simulator

```python
from poseidon.core import CoreRuntime
from poseidon.world import TidePool, OBSERVATION_NAMES, ACTIONS

policy = CoreRuntime("runs/tidal/core.pt")
env = TidePool(seed=42)
observation = env.reset(42)
done = False
while not done:
    action = policy.act(observation)
    observation, reward, done, info = env.step(action)
```

The interface is a 16-element normalized observation vector and six discrete macro actions: rest, forage, drink, shelter, explore, flee. A different simulator must map its observations/actions explicitly and retrain/evaluate the policy. Plugging it into arbitrary game physics does not confer survival skill. Snapshot import is validated transactionally; exogenous randomness uses named seed/tick/location events for matched experiments.

## Inspect the evidence

- `runs/tidal/receipt.json`: core configuration, actual exposures, baseline and development metrics.
- `runs/tidal/data_manifest.json`: data fingerprints, counts and split audits.
- `runs/tidal/core.pt`: model and exact training continuation state.
- `runs/language/report.json`: adapter status, consumed tokens and dev loss.
- `runs/evaluation.json`: held-out scene metrics, per-seed survival controls, solver checks, media receipts.
- `docs/RESULTS.md`: human-readable measured results after validation.
- `docs/DESIGN.md`: architecture, experimental questions and implementation plan.
- `docs/research-core.md`, `docs/research-world.md`, `docs/research-systems.md`: repository lineage and license observations.

Weights, downloaded datasets, logs and generated outputs are local artifacts, excluded from source packaging by `.gitignore`. Original code is MIT. Upstream weights/data retain Apache-2.0 terms; dependencies retain their own licenses. See `THIRD_PARTY.md` and the downloaded source cards. No biological, consciousness, robotics, real-world survival, general multimodal or photorealistic generation claim is made.
