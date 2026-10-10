# Supermix Poseidon

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Supermix--Poseidon-yellow)](https://huggingface.co/Kai9987kai/Supermix-Poseidon)
[![GitHub](https://img.shields.io/badge/GitHub-Supermix--Poseidon-blue)](https://github.com/kai9987kai/Supermix-Poseidon)

Poseidon is a local research workbench with separate components for language, a compact learned scene-and-control policy, a deterministic survival simulation, procedural media, exact arithmetic, and external memory. It is a software system, not a single multimodal foundation model.

## v1.2.0 research upgrade

The opt-in **TITAN** controller combines selected ideas from the reviewed project portfolio around the existing Tidal policy. Its six-arm experiment compares the frozen core, AURA, METAMORPH, CHIMERA, HYPERION, and TITAN on matched seeds. Receipts contain decisions and observed transitions, then replay both controller outputs and TidePool outcomes. A controller receives reward only after the environment returns it. These measurements describe a synthetic task; they do not establish general-agent or real-world capability, and the experiment never promotes or replaces a model.

The upgrade also adds deterministic per-episode resets for TITAN's auxiliary Genesis ecology and Tessera action memory. Archimedes fluid calculations, Genesis ecology, classical state-vector circuits, simulated intermittent-energy storage, synthetic scanline telemetry, and static action hooks are local software mechanisms. They do not operate physical devices or provide biological evidence.

See the [TITAN design](docs/TITAN_DESIGN.md), [TITAN results](docs/TITAN_RESULTS.md), [Archimedes design](docs/ARCHIMEDES_DESIGN.md), [Genesis design](docs/GENESIS_DESIGN.md), and [review of the supplied repositories and research](docs/research/README.md).

## Run locally

Use Python 3.10 or newer with a compatible CPU PyTorch installation. From the repository root:

```powershell
python -m pip install -e .
python -m poseidon status
python -m poseidon serve --port 8792
```

Open `http://127.0.0.1:8792`. The workbench can also run a seeded TITAN episode from the command line:

```powershell
python -m poseidon world --planner titan --seed 42 --max-steps 64 --scarcity 2.5
```

Run and replay the paired experiment:

```powershell
python -m poseidon titan-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 99100000
python -m poseidon verify-titan outputs/titan_experiments/RECEIPT.json
```

The receipt records all six arms and is sealed to the core checkpoint and TidePool source hashes. A short benchmark is useful for reproducibility checks, not for a robust performance conclusion.

## Model components and boundaries

| Component | Role | Release status |
|---|---|---|
| Tidal | Compact locally trained scene, action, and dynamics policy | Existing selected checkpoint; unchanged in this upgrade |
| SmolLM2 | Pretrained conversation model | Separate backend with its own upstream license |
| TITAN and other planners | Optional controllers for TidePool | Experimental; no default-policy promotion |
| Image, video, and mesh tools | Bounded procedural outputs | Separate software paths |
| Math and carrier memory | Exact solver and external retrieval | Separate tools; not weights in Tidal |

TidePool is a deterministic toy environment. TITAN's ecology, fluid, quantum-inspired, and device-inspired telemetry is simulated on a conventional computer. Poseidon makes no claim of consciousness, biological validation, physical RF harvesting, NTAG hardware, analog video, quantum hardware, or general autonomous learning. The LoRA remains an inactive candidate unless explicitly selected.

## Release and licensing

The release manifest binds packaged files to a source commit and SHA-256 inventory. Use the [release guide](docs/HUGGINGFACE_RELEASE.md) to package, verify, and publish a version. The Hub repository root is a custom PyTorch application package; `AutoModel.from_pretrained` is not its runtime interface.

Poseidon code and Tidal artifacts use MIT terms. Bundled pretrained language weights and tokenizer retain their upstream Apache-2.0 terms and attribution. See the packaged license files and upstream model card before redistribution.
