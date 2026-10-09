"""Build a frozen, allowlisted Poseidon model release from a clean Git checkout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Kai9987kai/Supermix-Poseidon"
LANGUAGE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PREVIOUS_HUB_REVISION = "cfe2b0e6a3e5599eaac8c14b56a6ab3b47f5f953"
EXPERIMENTS = (
    "c260cb232f1635a1-949a615206d8.json",
    "875452fefbf277df-3b1edfb49993.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def model_card(revision: str, core_hash: str, atlas_hash: str) -> str:
    return f"""---
language:
- en
license: other
license_name: mit-and-apache-2.0
license_link: https://huggingface.co/{REPO_ID}/blob/main/THIRD_PARTY.md
library_name: pytorch
base_model:
- HuggingFaceTB/SmolLM2-135M-Instruct
tags:
- world-models
- imitation-learning
- dagger
- mixture-of-experts
- model-predictive-control
- counterfactual-memory
- lora
- cpu
- experimental
- synthetic-simulation
---

# Supermix Poseidon v0.2.0: Counterfactual Atlas

A CPU-local experimental system with a trained synthetic scene/control core,
a separate pretrained conversation model, and an inspectable action-conditioned
residual memory. This release bundles matching runtime source and model artifacts.
It is a custom PyTorch application, not a Transformers model at the repository root.

[Source commit](https://github.com/kai9987kai/Supermix-Poseidon/tree/{revision}) ·
[Measured results](docs/ATLAS_RESULTS.md) · [Release guide](docs/HUGGINGFACE_RELEASE.md)

## Components and activation

| Component | Artifact | Status |
|---|---|---|
| Tidal core: 327,230 parameters | `runs/tidal_dagger/core.pt` | Existing selected DAgger policy, default scene/control backend |
| Supervised baseline | `runs/tidal/core.pt` | Comparison checkpoint, original embedded training state |
| SmolLM2: 134,515,008 stored parameters | `models/language/model.safetensors` | Separate unchanged pretrained conversation backend |
| LoRA: 61,440 parameters | `runs/language/adapter/` | Experimental candidate, inactive by default |
| Counterfactual Atlas | `outputs/atlas/atlas.json` | Fitted residual memory, opt-in planner |

Tidal uses 512 hashed lemma/bigram features, a 128-wide state, four gated experts,
and two recurrent updates within each decision. It predicts five controlled scene
attributes, six simulation actions and 16 state deltas. Its trained routing is
dense across four experts; experimental sparse/ensemble modules are separate.
The default protocol resets hidden state at each decision. Fluent conversation,
procedural geometry/rendering, external carrier storage and exact maths use
separate components. No unified multimodal, real-world survival, biological
connectome, consciousness or general autonomous-learning capability is claimed.

## Download and run

```bash
python -m pip install huggingface-hub
hf download {REPO_ID} --revision v0.2.0 --local-dir Poseidon-release
cd Poseidon-release
python -m pip install -e .
python tools/verify_huggingface_release.py . --smoke
python -m poseidon serve --port 8787
```

Open http://127.0.0.1:8787. Use Python 3.10+ with a compatible CPU PyTorch 2.6+
installation. The snapshot includes weights and tokenizer; no additional model
download is needed for the workbench. Native checkpoints use
`torch.load(..., weights_only=True)` and language loading disables remote code.
`AutoModel.from_pretrained('{REPO_ID}')` is unsupported: use the packaged runtime.
For immutable provenance, pin the Hub commit SHA returned by the release upload.

```python
from poseidon.runtime import Poseidon
runtime = Poseidon('.')  # run from the downloaded release directory
print(runtime.respond('Survive', mode='world', seed=42, planner='policy'))
print(runtime.respond('Survive', mode='world', seed=42, planner='atlas'))
```

Default language uses pinned `HuggingFaceTB/SmolLM2-135M-Instruct` at
`{LANGUAGE_REVISION}`. Explicitly test the candidate with
`python -m poseidon chat 'Explain why seasons change.' --adapter runs/language/adapter`.
Generation uses at most a 1,024-token local input budget and 256 generated tokens.
This release does not automatically activate the adapter or Atlas.

## Training and measured limits

The supervised Tidal run used 120,000 synthetic examples and 360,000 exposures.
DAgger selected round two using validation seeds; 769,600 exposures is the aggregate
over the four-round run, not a precise exposure count of the selected weights.
Stored evaluation reports describe controlled scene vocabulary and synthetic
TidePool survival. Reported 100% results do not imply real-world generalization.

The Atlas fit contains 1,728 training and 864 calibration transitions, with six
correlated action branches per anchor. Its held-out calibration maximum vital-state
error fell from 0.03950 to 0.03183 (19.4%). On 16 fresh evaluation seeds, Atlas and
the incumbent survived 15/16 episodes and chose identical actions: the conservative
gate accepted zero overrides. Full 16-channel prediction MSE worsened in both
suites. There is no measured survival improvement. The six-controller receipts
replay 96 episodes and 11,079 transitions; these are bounded development evidence.
Controller compute budgets differ, and timings are not a fair speed ranking.

The candidate LoRA ran 256 steps, saw 512 examples and 42,104 tokens, and changed
development loss from 0.960879 to 0.958487. That does not establish better conversation
or reasoning; the dataset overlaps the upstream model's instruction-training
distribution. The candidate remains inactive. The conversation model can hallucinate.

## Provenance and file integrity

Source revision: `{revision}`.
Active core file SHA-256: `{core_hash}`.
Atlas canonical payload SHA-256: `{atlas_hash}` (distinct from file-byte SHA-256).
`manifest.json` and `SHA256SUMS.txt` inventory the staged files. The manifest binds
source, weights, upstream provenance and the two exact final experiment receipts.
Checksums diagnose integrity; they do not prove scientific validity or authorship.
`SOURCE_README.md` is the project's operational README. Root `core.pt`,
`core_base.pt`, and `adapter/` are compatibility copies of the native layout.
User memory, raw language datasets, caches, logs, rejected Atlas artifacts and
language optimizer continuation state are excluded.

## Superseded historical release

The old `ensemble_adapted.pt` is archived as `legacy/ensemble_adapted.pt` and retained
in Hub history. Its archived bytes are inventoried for integrity, without any
capability validation, and are not loaded by the default runtime. Its former
promotion claim is invalidated. The old demonstration video is also archived.
`beyond_audit_receipt.json` now records supersession, while original evidence and
the previous model card are preserved under `legacy/`. The earlier card's claims
of authentic fly-connectome wiring, calibrated failure probabilities, learned
delayed-memory performance, zero-width survival intervals and verified ensemble
promotion are not supported by this release. See the repaired source documentation.

## License and attribution

Original Poseidon code, Tidal checkpoints and Atlas are MIT (`LICENSE`). Bundled
SmolLM2 weights/tokenizer, source card and related adapter retain Apache-2.0 terms
and attribution (`licenses/APACHE-2.0.txt`, `models/language/README.md`). The local
adapter used `HuggingFaceTB/smol-smoltalk` at
`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`; its source card and selection manifest
are included, without raw dataset redistribution. Dependencies retain their own
licenses. See `THIRD_PARTY.md`. The Hub metadata uses a composite license label
to avoid relicensing upstream artifacts as MIT.
"""


def package(output: Path, repo_id: str = REPO_ID) -> dict:
    if repo_id != REPO_ID:
        raise ValueError("this release card is bound to Kai9987kai/Supermix-Poseidon")
    if git("status", "--porcelain"):
        raise ValueError("commit intended source changes before packaging; checkout must be clean")
    revision = git("rev-parse", "HEAD")
    output = output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output must not contain or replace the source checkout")
    if output.exists():
        raise ValueError("output already exists; choose a fresh directory")

    sys.path.insert(0, str(ROOT))
    import torch
    from poseidon.atlas import CounterfactualAtlas
    from poseidon.core import CoreRuntime, active_core_path
    from poseidon.experiments import load_and_verify
    torch.set_num_threads(2)
    core_path = active_core_path(ROOT)
    core_hash = sha256(core_path)
    pointer = json.loads((ROOT / "runs/active_core.json").read_text(encoding="utf-8"))
    if pointer.get("sha256") != core_hash:
        raise ValueError("active pointer does not identify actual checkpoint bytes")
    core = CoreRuntime(core_path)
    atlas = CounterfactualAtlas.load(core, ROOT / "outputs/atlas/atlas.json")
    atlas_hash = atlas.artifact["sha256"]
    experiment_verification = {name: load_and_verify(ROOT / "outputs/experiments" / name) for name in EXPERIMENTS}
    upstream = json.loads((ROOT / "models/language/manifest.json").read_text(encoding="utf-8"))
    if upstream["revision"] != LANGUAGE_REVISION:
        raise ValueError("unexpected language source revision")
    for name, expected in upstream["files"].items():
        if sha256(ROOT / "models/language" / name) != expected:
            raise ValueError(f"upstream language file changed: {name}")
    from huggingface_hub import hf_hub_download
    historical_assets = {name: Path(hf_hub_download(repo_id, name, revision=PREVIOUS_HUB_REVISION))
                         for name in (".gitattributes", "replay.mp4")}
    legacy_ensemble_hash = "820cfc0cadc9f28f887715a9487bc3f04a7071e8a21e23d5806429bd75ae59bd"
    if sha256(ROOT / "runs/beyond_adapted/ensemble_adapted.pt") != legacy_ensemble_hash:
        raise ValueError("historical ensemble bytes changed")

    def copy(relative: str, target: str | None = None) -> None:
        requested = ROOT / relative
        source = requested.resolve()
        if not source.is_relative_to(ROOT) or requested.is_symlink() or not source.is_file():
            raise ValueError(f"invalid source artifact: {relative}")
        destination = output / (target or relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    output.mkdir(parents=True)
    source_files = git("ls-files", "-z").split("\0")
    for name in source_files:
        if name:
            copy(name, "SOURCE_README.md" if name == "README.md" else None)
    source_hashes = {name: sha256(ROOT / name) for name in source_files if name}

    artifacts = [
        "runs/tidal_dagger/core.pt", "runs/tidal/core.pt",
        "runs/language/adapter/adapter_model.safetensors",
        "outputs/atlas/atlas.json", "outputs/atlas/fit_receipt.json",
        "models/language/manifest.json", "data/language/manifest.json",
        "data/language/SOURCE_CARD.md",
    ] + [f"models/language/{name}" for name in upstream["files"]]
    artifacts += [f"outputs/experiments/{name}" for name in EXPERIMENTS]
    for relative in artifacts:
        copy(relative)
    copy("docs/history/HF_MODEL_CARD_v0.1.md", "legacy/model-card-before-v0.2.md")
    copy("runs/beyond_adapted/ensemble_adapted.pt", "legacy/ensemble_adapted.pt")
    shutil.copyfile(historical_assets[".gitattributes"], output / ".gitattributes")
    shutil.copyfile(historical_assets["replay.mp4"], output / "legacy/replay-before-v0.2.mp4")

    # Original bytes remain available; a current envelope retracts the old promotion.
    superseded = json.loads((ROOT / "runs/beyond_adapted/audit_receipt.json").read_text(encoding="utf-8"))
    write_json(output / "legacy/beyond-audit-before-v0.2.json", superseded["historical_receipt"])
    write_json(output / "beyond_audit_receipt.json", superseded)
    (output / "legacy/README.md").write_text(
        "# Historical, superseded evidence\n\nThese documents preserve the pre-v0.2 release. "
        "They do not establish current ensemble promotion, calibrated risk, biological wiring, "
        "or learned recurrent-memory performance. Read the root model card and docs/ATLAS_RESULTS.md.\n",
        encoding="utf-8",
    )
    for source, target in {
        "runs/tidal_dagger/core.pt": "core.pt", "runs/tidal/core.pt": "core_base.pt",
        "runs/active_core.json": "active_core.json", "runs/tidal_dagger/report.json": "dagger_report.json",
        "runs/tidal/receipt.json": "receipt.json", "runs/tidal/data_manifest.json": "data_manifest.json",
        "runs/calibration.json": "calibration.json", "runs/evaluation.json": "evaluation.json",
    }.items():
        copy(source, target)
    write_json(output / "config.json", core.payload["config"])

    adapter_config = json.loads((ROOT / "runs/language/adapter/adapter_config.json").read_text(encoding="utf-8"))
    adapter_config.update(base_model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct", revision=LANGUAGE_REVISION)
    adapter_card = """---
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
license: apache-2.0
tags:
- peft
- lora
- experimental
---
# Poseidon candidate LoRA (inactive)

61,440 trainable parameters: rank 8, alpha 16, q_proj/v_proj on layers 26-29.
256 steps, 512 examples, 42,104 tokens; dev loss 0.960879 to 0.958487.
No improvement in conversation or reasoning has been established. Load explicitly
with the packaged runtime and the pinned SmolLM2 base; default inference omits it.
See runs/language/report.json, data/language/SOURCE_CARD.md and the root model card.
"""
    for directory in ("runs/language/adapter", "adapter"):
        copy("runs/language/adapter/adapter_model.safetensors", directory + "/adapter_model.safetensors")
        write_json(output / directory / "adapter_config.json", adapter_config)
        (output / directory / "README.md").write_text(adapter_card, encoding="utf-8")
    (output / "README.md").write_text(model_card(revision, core_hash, atlas_hash), encoding="utf-8")

    if git("status", "--porcelain") or git("rev-parse", "HEAD") != revision:
        raise ValueError("source checkout changed while packaging")
    for name, expected in source_hashes.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"source changed while packaging: {name}")
    if sha256(core_path) != core_hash or sha256(ROOT / "outputs/atlas/atlas.json") != sha256(output / "outputs/atlas/atlas.json"):
        raise ValueError("core or Atlas changed while packaging")

    files = [{"path": p.relative_to(output).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    manifest = {
        "schema": "poseidon-huggingface-release-v1", "repo_id": repo_id,
        "version": "0.2.0", "source_revision": revision, "files": files,
        "previous_hub_revision": PREVIOUS_HUB_REVISION,
        "historical_artifacts": ["legacy/ensemble_adapted.pt", "legacy/replay-before-v0.2.mp4",
                                 "legacy/beyond-audit-before-v0.2.json", "legacy/model-card-before-v0.2.md"],
        "core_sha256": core_hash, "atlas_artifact_sha256": atlas_hash,
        "upstream_language": upstream, "source_file_sha256": source_hashes,
        "release_transforms": {"README.md": "Hub model card; original becomes SOURCE_README.md",
                               "runs/language/adapter/README.md": "authored inactive candidate card",
                               "runs/language/adapter/adapter_config.json": "portable upstream id and pinned revision"},
        "experiment_verification": experiment_verification,
        "build_environment": {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "safetensors", "numpy", "huggingface-hub")},
        "activation": {"core": "existing-dagger-selection", "language": "unchanged-upstream-base",
                       "language_adapter": "inactive-candidate", "atlas": "opt-in-experiment",
                       "legacy_ensemble": "historical-unvalidated-not-loaded"},
    }
    write_json(output / "manifest.json", manifest)
    sums = files + [{"path": "manifest.json", "sha256": sha256(output / "manifest.json")}]
    (output / "SHA256SUMS.txt").write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in sums), encoding="utf-8")
    return {"output": str(output), "source_revision": revision, "files": len(files) + 2,
            "bytes": sum(row["size"] for row in files), "manifest_sha256": sha256(output / "manifest.json"),
            "core_sha256": core_hash, "atlas_artifact_sha256": atlas_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.repo_id), indent=2))


if __name__ == "__main__":
    main()
