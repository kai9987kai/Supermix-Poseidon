# Supermix Poseidon v2.0: Cognitive Platform

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Supermix--Poseidon-yellow)](https://huggingface.co/Kai9987kai/Supermix-Poseidon)
[![GitHub](https://img.shields.io/badge/GitHub-Supermix--Poseidon-blue)](https://github.com/kai9987kai/Supermix-Poseidon)

Poseidon v2.0 evolves the workbench into an experimental **cognitive AI platform** that predicts, remembers, plans, and adapts through experience while scientifically evaluating its own decisions across matched lesion assays and partially observable environments (POMDP).

## v2.0 Cognitive Platform Architecture

Poseidon v2.0 unifies six cognitive subsystems around the immutable, frozen Tidal core checkpoint:

1. **ORACLE (`poseidon/oracle.py`)**: Bootstrapped predictive world-model ensemble ($K$-head) with empirical quantile-calibrated uncertainty $U_{\text{cal}}(s, a)$ and multi-step recursive trajectory imagination.
2. **MNEMOSYNE (`poseidon/mnemosyne.py`)**: Biologically inspired 3-tier memory hierarchy:
   - **Tier 1 (Working Memory)**: Recurrent GRU context cell dynamically updated by prediction surprise ($e_{t-1} = \|o_t - \hat{o}_t\|_2$).
   - **Tier 2 (Episodic Store)**: Value-gated experiential memory weighted by reward, surprise, and recency.
   - **Tier 3 (Spatial Map)**: Online Bayesian replenishment estimation with conservative lower-confidence bounds (LCB).
   - **Matched Lesion Assays**: Validated across `intact`, `erased`, `shifted`, `random_retrieval`, and `memory_free` modes.
3. **HORIZON-X (`poseidon/horizon_x.py`)**: Multi-horizon ($H \in \{1, 4, 8, 16, 32\}$) risk-sensitive planner with dynamic metabolic reserve modulation ($\lambda_t = \lambda_0 \cdot \max(\mu_{\min}, \text{health} \cdot \min(\text{energy}, \text{hydration}))$) to avoid the "pessimism trap" under starvation.
4. **ADAPT (`poseidon/adapt.py`)**: Evidence-gated continual learning pipeline with prioritized transition replay, anti-forgetting anchor datasets ($D_{\text{KL}}$ regularization), and audited promotion gates ($\le 2\%$ anchor regression).
5. **NEURO (`poseidon/neuro.py`)**: Neuromodulated sparse connectome routing implementing small-world Watts-Strogatz graph masks and metabolic gain scaling.
6. **WORLDLAB (`poseidon/worldlab.py`)**: High-fidelity partially observable Markov decision processes (`POMDPTidePool`) with fog-of-war scent occlusion, sensor noise, dynamic tidal oscillations, and a standardized simulation bridge (`WorldBridgeProtocol`).

See the complete research documentation in [docs/POSEIDON_V2_RESEARCH.md](docs/POSEIDON_V2_RESEARCH.md).

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
