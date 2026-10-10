# Supermix Poseidon

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Supermix--Poseidon-yellow)](https://huggingface.co/Kai9987kai/Supermix-Poseidon)
[![GitHub](https://img.shields.io/badge/GitHub-Supermix--Poseidon-blue)](https://github.com/kai9987kai/Supermix-Poseidon)

Poseidon is a local research workbench with separate components for language, a compact learned scene-and-control policy, a deterministic survival simulation, procedural media, exact arithmetic, and external memory. It is a software system, not a single multimodal foundation model.

## v1.3.0 TRIDENT research upgrade

Version 1.3.0 adds **TRIDENT**, an opt-in controller around the unchanged Tidal core. It corrects one-step predictions with recent observed prediction errors, admits directed travel only after enough local visits unless that transition has been observed, and requires an override to beat erased and time-shifted residual controls by configured margins. It does not change or train model weights.

The six-arm paired benchmark compares the frozen core, TRIDENT, erased residual, shifted residual, directed-support gate-off, and residual-gate-off controls on identical seeds. Receipts retain decisions and observed transitions and can replay both controller outputs and TidePool outcomes. The bundled four-seed result is descriptive synthetic evidence, not proof of a general performance gain; it does not promote a model.

Earlier TITAN mechanisms remain optional software simulations. Archimedes fluid calculations, Genesis ecology, classical state-vector circuits, intermittent-energy storage, scanline telemetry, and static action hooks do not operate physical devices or establish biological evidence.

See the [TRIDENT design](docs/TRIDENT_DESIGN.md), [TRIDENT results](docs/TRIDENT_RESULTS.md), [TITAN design](docs/TITAN_DESIGN.md), [TITAN results](docs/TITAN_RESULTS.md), and [review of the supplied repositories and research](docs/research/README.md).

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
