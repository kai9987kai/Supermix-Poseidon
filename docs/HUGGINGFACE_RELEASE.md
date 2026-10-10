# Hugging Face release procedure

Poseidon is a custom PyTorch runtime package hosted at [Kai9987kai/Supermix-Poseidon](https://huggingface.co/Kai9987kai/Supermix-Poseidon). The repository root is not a Transformers `AutoModel` checkpoint. The v1.2.0 release adds the opt-in TITAN controller and a replayable six-arm TidePool experiment; it does not replace or retrain the selected Tidal core.

## Download and run

```powershell
python -m pip install huggingface-hub
hf download Kai9987kai/Supermix-Poseidon --revision v1.2.0 --local-dir Poseidon-release
Set-Location Poseidon-release
python -m pip install -e .
python tools/verify_huggingface_release.py . --smoke
python -m poseidon serve --port 8792
```

Pin the immutable Hub commit SHA instead of the tag when exact revision provenance is needed. Use Python 3.10+ with compatible CPU PyTorch. The packaged snapshot contains the default runtime weights and tokenizer; loading the custom project uses the Poseidon runtime rather than `AutoModel.from_pretrained`.

To evaluate TITAN locally:

```powershell
python -m poseidon world --planner titan --seed 42 --max-steps 64 --scarcity 2.5
python -m poseidon titan-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 99100000
python -m poseidon verify-titan outputs/titan_experiments/RECEIPT.json
```

The receipt is bound to the active Tidal checkpoint and TidePool source bytes. It includes action-level decisions and actual observed environment transitions for the core, AURA, METAMORPH, CHIMERA, HYPERION, and TITAN arms. Replay verifies integrity and deterministic behavior; it is not an independent scientific replication or evidence of performance beyond the recorded synthetic task.

## Build, verify, and upload

Start with a clean Git checkout and a completed `outputs/titan_experiments/RECEIPT.json`. The packager checks the active checkpoints, pinned upstream language files, existing experiment evidence, and the TITAN receipt. It rejects dirty source so the manifest can bind a commit.

```powershell
python tools/package_huggingface.py --output outputs/releases/poseidon-v1.2.0
python tools/verify_huggingface_release.py outputs/releases/poseidon-v1.2.0 --smoke
hf upload Kai9987kai/Supermix-Poseidon outputs/releases/poseidon-v1.2.0 . --commit-message "Release Poseidon v1.2.0 with replayable TITAN evaluation"
```

After upload, verify the Hub's new commit SHA, `v1.2.0` tag, model card, manifest, file inventory, and remote file hashes against the local release package. A local package or successful CLI upload alone is not proof that publication completed. Preserve the existing Hub history and release assets; do not force-push.

## Scope and licensing

Tidal is a compact learned scene/control model. SmolLM2 is a separate pretrained conversation model with separate upstream terms. Procedural media, exact arithmetic, external carrier retrieval, and TidePool are distinct runtime components. The TITAN additions that resemble ecology, fluids, quantum circuits, intermittent power, or device telemetry are software simulations. They do not operate physical devices or validate biological or general-agent claims.

Poseidon source and Tidal artifacts retain MIT terms. SmolLM2 weights, tokenizer, and attribution retain the upstream Apache-2.0 terms. Review included license files and upstream model documentation before redistribution.
