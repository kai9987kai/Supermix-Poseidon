# Poseidon v0.5.1 model release

The v0.5.1 profile bundles the scarcity-reset correction, strict parent evidence
checks, two corrected evaluation suites and their real 16-step trajectory audits.
It retains historical receipts and discloses their one-step audit and map-setting
limits. Historical Horizon/Odyssey evidence is checked with `verify-evidence`
and matching sidecar artifacts; the older native verifier requires frozen source.

The release target is [Kai9987kai/Supermix-Poseidon](https://huggingface.co/Kai9987kai/Supermix-Poseidon).
Its `manifest.json` binds a clean Git source commit to every packaged file's size
and SHA-256. `SHA256SUMS.txt` provides an additional file integrity inventory.
Publishing these artifacts improves the experimental instrument without changing
the selected core, language weights or default DAgger policy. Odyssey remains opt-in.

## Download and run

```powershell
# Install the Hub CLI in a Python environment with compatible CPU PyTorch.
python -m pip install huggingface-hub
hf download Kai9987kai/Supermix-Poseidon --revision v0.5.1 --local-dir Poseidon-release
cd Poseidon-release
python -m pip install -e .
python tools/verify_huggingface_release.py . --smoke
python -m poseidon serve --port 8787
```

Open http://127.0.0.1:8787. For a reproducible download, add `--revision` with the
release's Hub commit SHA or `v0.5.1` tag. Python 3.10+ and PyTorch 2.6+ are required;
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
python -m poseidon world --planner odyssey --seed 42
python -m poseidon world --planner horizon --seed 42
python -m poseidon world --planner contrast --seed 42
python -m poseidon world --planner atlas --seed 42
python -m poseidon chat "Explain why seasons change." --adapter runs/language/adapter
```

The LoRA command is an explicit candidate test. Default conversation uses the
unchanged, pinned SmolLM2 base. Odyssey Atlas, Horizon Atlas, Contrast and Atlas are also opt-in. User carrier memory, raw
language datasets, caches, generated media and optimizer continuation artifacts
are excluded. The supervised baseline checkpoint retains its original embedded
training state; the selected DAgger checkpoint has no optimizer state. No claim of
a complete resumable language-training package is made.

## Build and verify a release

```powershell
# Commit all intended source changes first; packaging rejects a dirty checkout.
python tools/package_huggingface.py --output outputs/releases/poseidon-v0.5.1
python tools/verify_huggingface_release.py outputs/releases/poseidon-v0.5.1 --smoke
hf upload Kai9987kai/Supermix-Poseidon outputs/releases/poseidon-v0.5.1 . --commit-message "Release Poseidon v0.5.1 trajectory evidence audit"
```

Packaging checks upstream language hashes, the actual active core hash, Atlas
checkpoint binding and experiment receipts. It refuses an occupied output
directory and retrieves two historical assets from the explicitly pinned previous
Hub revision. Verification checks file bytes and then, with `--smoke`, runs the
bundled code in a separate process for core, Atlas, language, candidate LoRA and
exact maths. It also reconstructs packaged maps and replays the new H-step
branches, checking their verifier results against the manifest. Hub publication
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

Atlas reduced calibration vital-state error by 19.4%, but broader prediction error
worsened and its conservative gate accepted no action overrides on 16 fresh
evaluation seeds. Survival matched the incumbent. See [measured results](ATLAS_RESULTS.md).
The 61,440-parameter LoRA candidate's small development-loss change is not evidence
of improved conversation or reasoning.

The two corrected four-seed suites each compare ten controllers for up to 64
steps. Calibrated Odyssey matched the core policy with zero overrides. The
separate audit replayed 546 real 16-step branches; six of 11 ungated overrides
had negative return advantages. No planner promotion or general superiority is
claimed. See [trajectory evidence and limits](TRAJECTORY_AUDIT.md).

The pre-v0.2 Hub card and invalid Beyond promotion receipt are preserved under
`legacy/` with explicit supersession notes. The old Hub `ensemble_adapted.pt` is
archived under `legacy/` to preserve its bytes, is not loaded by the default runtime, and has no
current validated promotion status. The earlier claims of authentic biological
connectome wiring, calibrated failure probability, zero-width survival uncertainty,
and a learned delayed-memory result are superseded. The new release does not
validate those claims.

Original code, Tidal weights and Atlas retain MIT. Bundled SmolLM2 weights,
tokenizer and related candidate adapter retain Apache-2.0 attribution and terms;
both full license texts and source cards are included. Research repositories
informed independently implemented designs; their code or restricted weights
were not imported wholesale.
