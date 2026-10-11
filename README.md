# Supermix Poseidon v2.0: Cognitive Platform

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Supermix--Poseidon-yellow)](https://huggingface.co/Kai9987kai/Supermix-Poseidon)
[![GitHub](https://img.shields.io/badge/GitHub-Supermix--Poseidon-blue)](https://github.com/kai9987kai/Supermix-Poseidon)

Poseidon is an experimental **cognitive AI workbench** combining a trained Tidal core with optional prediction, memory, planning and adaptation modules. Subsystem availability and passing unit tests do not establish improved learning or transfer. The [11 October architecture audit](docs/research/architecture-audit-2026-10-11.md) records current wiring, verification and evaluation limits.

## v2.0 Cognitive Platform Architecture

The v2.0 coordinator instantiates six experimental subsystems around the frozen Tidal core checkpoint:

1. **ORACLE**: `poseidon/oracle.py` provides independently initialized heads, a fitted disagreement/error curve and trajectory imagination. CognitiveAgent and HyperAgent currently construct fresh, uncalibrated heads. The separately trained, episode-bootstrap predictor in `poseidon/research/` has replayable fit/evaluation receipts and remains inactive; it is not yet wired into those agents.
2. **MNEMOSYNE (`poseidon/mnemosyne.py`)**: Biologically inspired 3-tier memory hierarchy:
   - **Tier 1 (Working Memory)**: Recurrent GRU context cell dynamically updated by prediction surprise ($e_{t-1} = \|o_t - \hat{o}_t\|_2$).
   - **Tier 2 (Episodic Store)**: Value-gated experiential memory weighted by reward, surprise, and recency.
   - **Tier 3 (Spatial Map)**: Online Bayesian replenishment estimation with conservative lower-confidence bounds (LCB).
   - **Matched Lesion Modes**: `intact`, `erased`, `shifted`, `random_retrieval`, and `memory_free` are implemented. Memory outputs currently feed v2 telemetry; causal decision benefit has not been established.
3. **HORIZON-X (`poseidon/horizon_x.py`)**: Planner with default horizons `{1,4,8,16}` and reserve-dependent risk penalties. It currently uses the fresh runtime ORACLE; reserve modulation is a heuristic requiring outcome validation.
4. **ADAPT (`poseidon/adapt.py`)**: Experimental residual adaptation, replay and anchor checks. Useful adaptation and retention require separate held-out evidence.
5. **NEURO (`poseidon/neuro.py`)**: Sparse routing with graph masks and gain scaling. The v2 coordinator reports its outputs as telemetry; they do not currently alter the planner input.
6. **WORLDLAB (`poseidon/worldlab.py`)**: Synthetic partially observed TidePool with fog, sensor noise, tidal effects and an external simulation bridge contract. It is not a validated physical or biological model.

See the [subsystem documentation](docs/POSEIDON_V2_RESEARCH.md), [24-project source review](docs/research/portfolio-refresh-2026-10-11.md) and [proposed learning and transfer operation](docs/superpowers/specs/2026-10-11-learning-transfer-design.md). The new v3/OmniPlane modules are additional experiments; their presence does not resolve the predictor integration or evidence gaps.

## Run locally

Use Python 3.10 or newer with a compatible CPU PyTorch installation. From the repository root:

```powershell
python -m pip install -e .
python -m poseidon status
python -m poseidon serve --port 8792
```

Open `http://127.0.0.1:8792`. The workbench can run a seeded TRIDENT episode from the command line:

```powershell
python -m poseidon world --planner trident --seed 42 --max-steps 64 --scarcity 2.5
```

Run and replay the paired TRIDENT experiment:

```powershell
python -m poseidon trident-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 88000000
python -m poseidon verify-trident outputs/trident_experiments/RECEIPT.json
```

The receipt records all six arms and is sealed to the core checkpoint and TidePool source hashes. A short benchmark is useful for reproducibility checks, not for a robust performance conclusion.

## Model components and boundaries

| Component | Role | Release status |
|---|---|---|
| Tidal | Compact locally trained scene, action, and dynamics policy | Existing selected checkpoint; unchanged in this upgrade |
| SmolLM2 | Pretrained conversation model | Separate backend with its own upstream license |
| TITAN, TRIDENT, and other planners | Optional controllers for TidePool | Experimental; no default-policy promotion |
| Image, video, and mesh tools | Bounded procedural outputs | Separate software paths |
| Math and carrier memory | Exact solver and external retrieval | Separate tools; not weights in Tidal |

TidePool is a deterministic toy environment. TITAN's ecology, fluid, quantum-inspired, and device-inspired telemetry is simulated on a conventional computer. Poseidon makes no claim of consciousness, biological validation, physical RF harvesting, NTAG hardware, analog video, quantum hardware, or general autonomous learning. The LoRA remains an inactive candidate unless explicitly selected.

## Release and licensing

The release manifest binds packaged files to a source commit and SHA-256 inventory. Use the [release guide](docs/HUGGINGFACE_RELEASE.md) to package, verify, and publish a version. The Hub repository root is a custom PyTorch application package; `AutoModel.from_pretrained` is not its runtime interface.

Poseidon code and Tidal artifacts use MIT terms. Bundled pretrained language weights and tokenizer retain their upstream Apache-2.0 terms and attribution. See the packaged license files and upstream model card before redistribution.
