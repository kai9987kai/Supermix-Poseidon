# Poseidon v1.0.0 model release

The v1.0.0 release bundles **CHIMERA** unified quantum superposition, holographic concept memory, and diamond lattice neuropil control, **Causeway** quantum superposition decision engine (6-dimensional complex state vector $|\psi\rangle \in \mathbb{C}^6$, continuous non-Hermitian gain potential evolution, unitary phase rotations, pairwise wave interference $I(a, b) = 2\text{Re}(\alpha_a^* \alpha_b)$, von Neumann entropy decoherence tracking, and peak Born-rule eigenstate collapse), **Holographic Concept Memory** (256-d Vector Symbolic Architecture with circular convolution binding, circular correlation unbinding, trace bundling with temporal decay, and vocabulary clean-up resonance), **Diamond Lattice Entorhinal Neuropil** (multi-scale 3D tetrahedral / Face-Centered Cubic geodesic grid cells with continuous toroidal velocity phase integration), **METAMORPH** developmental and topological control, **MOLT** life-stage developmental morphogenesis, **NexusFlow** directed causal flux routing, **Chronos** cyclic beacon synchronization, **AURA** biomimetic Central Complex controller, **Tessera** lineage-ratified macro-action commons, **Mnemorph** archaeological memory assay suite, **Odysseus** empirical cognitive navigator, **Memory Carrier Observatory (MCO)** causal factorial evaluation suite, and complete evaluation receipts with weightless replay verification. It retains all historical receipts and discloses evidence limits.

The release target is [Kai9987kai/Supermix-Poseidon](https://huggingface.co/Kai9987kai/Supermix-Poseidon).
Its `manifest.json` binds a clean Git source commit to every packaged file's size
and SHA-256. `SHA256SUMS.txt` provides an additional file integrity inventory.
Publishing these artifacts improves the experimental instrument without changing
the selected core, language weights or default DAgger policy. CHIMERA, METAMORPH, AURA, Tessera, Mnemorph, Odysseus and Helm remain opt-in.

## Download and run

```powershell
# Install the Hub CLI in a Python environment with compatible CPU PyTorch.
python -m pip install huggingface-hub
hf download Kai9987kai/Supermix-Poseidon --revision v1.0.0 --local-dir Poseidon-release
cd Poseidon-release
python -m pip install -e .
python tools/verify_huggingface_release.py . --smoke
python -m poseidon serve --port 8787
```

Open http://127.0.0.1:8787. For a reproducible download, add `--revision` with the
release's Hub commit SHA or `v1.0.0` tag. Python 3.10+ and PyTorch 2.6+ are required;
the native custom core is loaded with `torch.load(..., weights_only=True)`.
Transformers remote code is disabled. This is a custom PyTorch application, so
`AutoModel.from_pretrained` on the Hub repository root is not supported.

The snapshot keeps the runtime's native `runs/`, `models/language/` and `outputs/`
layout and bundles the matching source. No training or additional model download
is needed for the default workbench. `SOURCE_README.md` is the project README;
the root README is the Hub model card. Legacy root `core.pt`, `core_base.pt` and
`adapter/` paths are retained as compatibility copies.
The Hub `.gitignore` permits the packaged weights; original Git source exclusions
are preserved in `SOURCE_GITIGNORE`.

```powershell
python -m poseidon chat "Explain why seasons change."
python -m poseidon world --planner policy --seed 42
python -m poseidon world --planner chimera --seed 42
python -m poseidon world --planner metamorph --seed 42
python -m poseidon world --planner aura --seed 42
python -m poseidon world --planner odysseus --seed 42
python -m poseidon world --planner helm --seed 42
python -m poseidon world --planner odyssey --seed 42
python -m poseidon world --planner horizon --seed 42
python -m poseidon world --planner contrast --seed 42
python -m poseidon world --planner atlas --seed 42
python -m poseidon chimera-experiment
python -m poseidon metamorph-experiment
python -m poseidon aura-experiment
python -m poseidon tessera-experiment
python -m poseidon mnemorph-experiment
python -m poseidon odysseus-experiment
python -m poseidon mco-experiment
python -m poseidon chat "Explain why seasons change." --adapter runs/language/adapter
```

The LoRA command is an explicit candidate test. Default conversation uses the
unchanged, pinned SmolLM2 base. CHIMERA, METAMORPH, AURA, Odysseus Atlas, Helm Critic, Odyssey Atlas, Horizon Atlas, Contrast and Atlas are also opt-in. User carrier memory, raw
language datasets, caches, generated media and optimizer continuation artifacts
are excluded. The supervised baseline checkpoint retains its original embedded
training state; the selected DAgger checkpoint has no optimizer state. No claim of
a complete resumable language-training package is made.

## Build and verify a release

```powershell
# Commit all intended source changes first; packaging rejects a dirty checkout.
python tools/package_huggingface.py --output outputs/releases/poseidon-v1.0.0
python tools/verify_huggingface_release.py outputs/releases/poseidon-v1.0.0 --smoke
hf upload Kai9987kai/Supermix-Poseidon outputs/releases/poseidon-v1.0.0 . --commit-message "Release Poseidon v1.0.0 CHIMERA, Causeway, Hologram & Lattice"
```

Packaging checks upstream language hashes, the actual active core hash, Atlas/Odyssey/Helm/Odysseus
checkpoint bindings and experiment receipts. It refuses an occupied output
directory and retrieves historical assets from the explicitly pinned previous
Hub revision. Verification checks file bytes and then, with `--smoke`, runs the
bundled code in a separate process for core, Atlas, Helm, Odysseus, AURA, METAMORPH, CHIMERA, language, candidate LoRA and
exact maths. It also replays the new H-step branches, Helm evaluation suites, Odysseus macro-action
replays, MCO factual sensitivity experiments, METAMORPH paired benchmarks, and CHIMERA paired benchmarks, checking their verifier results against the manifest. Hub publication
is complete only after remote revision, file inventory,
model card and uploaded hashes are independently checked.
Known local download/install metadata (`.cache/huggingface`, `__pycache__`, root
`*.egg-info` and `.pytest_cache`) is ignored; unexpected source/model files and
links are rejected. Archived files are inventoried for integrity, without treating
their historical capability claims as valid.

## Evidence and historical files

Tidal is a 327,230-parameter synthetic scene/control network. Its conversation
backend is a separate 134,515,008-parameter pretrained SmolLM2 model. Procedural
rendering and exact maths are separate software paths. This is not a unified
multimodal foundation model.

CHIMERA v1.0.0 evaluates paired multi-arm benchmarks across 4 episodes and 48 steps each, achieving 100% survival, 4.6478 mean reward (outperforming AURA 4.6101 and METAMORPH 4.3177), mean quantum entropy of 0.964695, mean lattice coherence of 0.708253, 53.5 holographic items stored per episode, and a bounded Spectral Divergence Index of 0.032229. See [Causeway design](CAUSEWAY_DESIGN.md), [Holographic design](HOLOGRAPHIC_DESIGN.md), [Lattice design](LATTICE_DESIGN.md), and [CHIMERA results](CHIMERA_RESULTS.md).

Odysseus v0.7 integrates empirical patch replenishment estimation with lower confidence bounds, empirical macro-action transition matrices, and physiological stamina rest gating. Replay verification across 24 evaluation episodes and 1,536 transitions confirms that empirical topological macro-actions improve replenishment harvesting and travel efficiency without altering the frozen core network. See [design](ODYSSEUS_DESIGN.md) and [empirical results](ODYSSEUS_RESULTS.md).

The Memory Carrier Observatory (MCO) provides a 16-subset causal factorial framework across episodic, body, habitat, and social memory carriers evaluated across 1,024 independent trials with 3 negative controls (`shuffled`, `irrelevant`, `erased`). Results show strictly positive marginal Grounded F1 contributions for all carriers (+0.227 to +0.273) and catastrophic collapse under negative controls (MRR 0.500 -> 0.061 on shuffle, 0.000 on irrelevant/erased). See [design](MCO_DESIGN.md) and [empirical results](MCO_RESULTS.md).

Helm v0.6 introduces a history-conditioned return critic evaluated across 48 paired
episodes, 385 anchors, 4,620 branches, and 45,413 branch transitions. In empirical tests,
an uncalibrated critic experienced catastrophic failure (0% survival, negative rewards,
61% adverse override rate), while calibrated dual-horizon risk gating cleanly suppressed
unsafe overrides, maintaining 100% survival. See [design](HELM_DESIGN.md) and
[empirical results](HELM_RESULTS.md).

The pre-v0.2 Hub card and invalid Beyond promotion receipt are preserved under
`legacy/` with explicit supersession notes. The old Hub `ensemble_adapted.pt` is
archived under `legacy/` to preserve its bytes, is not loaded by the default runtime, and has no
current validated promotion status. The earlier claims of authentic biological
connectome wiring, calibrated failure probability, zero-width survival uncertainty,
and a learned delayed-memory result are superseded. The new release does not
validate those claims.

Original code, Tidal weights, Atlas, Helm and Odysseus retain MIT. Bundled SmolLM2 weights,
tokenizer and related candidate adapter retain Apache-2.0 attribution and terms;
both full license texts and source cards are included. Research repositories
informed independently implemented designs; their code or restricted weights
were not imported wholesale.
