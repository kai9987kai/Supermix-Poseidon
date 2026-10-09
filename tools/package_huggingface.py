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
HORIZON_EXPERIMENTS = (
    "f83c77efdb0a70ad-ac7304a77e02.json",
    "7eb3ec0702f503b1-a4bac8a67483.json",
)
ODYSSEY_EXPERIMENTS = (
    "cdaf779874a04de6-7dc45ea640b3.json",
    "e492031f870f05da-c9ba9d72f3be.json",
    "3ed044592794f58c-29888499b43a.json",
    "50ce54c9e486ce8f-0fb6374e203a.json",
)
ODYSSEUS_EXPERIMENTS = (
    "1521d677b6f2f75b-c4035a1005d3.json",
)
AURA_EXPERIMENTS = (
    "d9e0d76bd4812c03-b489e9702cff.json",
)
TRAJECTORY_AUDITS = {
    "8491880ca77a2dba-c7284642c7ab.json": "3ed044592794f58c-29888499b43a.json",
    "8b0e6f16a03ed340-a22ce43a85ac.json": "50ce54c9e486ce8f-0fb6374e203a.json",
}


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


def model_card(revision: str, core_hash: str, atlas_hash: str, horizon_hash: str, contrast_hash: str, odyssey_hash: str, version: str, helm_hash: str = "", odysseus_hash: str = "") -> str:
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
- multi-horizon-advantage
- spatial-memory
- cognitive-mapping
- topological-navigation
- bayesian-replenishment
- empirical-transitions
- memory-carrier-observatory
- lora
- cpu
- experimental
- synthetic-simulation
---

# Supermix Poseidon v{version}: AURA Central Complex, Tessera Macro-Commons & Mnemorph Archaeological Memory

Version 0.8 introduces three biomimetic, macro-action, and memory archaeology additions:
1. **AURA Biomimetic Central Complex Controller**: 16-wedge recurrent ring attractor compass with cosine kernel and continuous angular velocity integration, Population Vector Average (PVA) heading tracking and bump coherence $R$, optomotor stabilization reflex ($R < 0.35$), and 128 virtual Kenyon cell Mushroom Body neuropil with $k=8$ winner-take-all APL inhibitory gain control (6.25% sparsity) arbitrating metabolic, fatigue, and threat homeostatic drives into descending motor channels (`harvester`, `sentinel`, `escaper`).
2. **Tessera Lineage-Ratified Macro-Action Commons**: Decentralized macro-action discovery where single-lineage candidate bursts are quarantined, and ratified into global `OP_*` opcodes only upon independent dual-lineage confirmation ($\Delta R > 0$) with finite-domain stamina and horizon safety assertions.
3. **Mnemorph Archaeological Memory Assays**: Associative concept graph assays evaluating catastrophic forgetting and topological resilience across typed structural interventions (`intact`, `hub_lesion`, `periphery_lesion`, `associative_regrowth`), measuring Hub Vulnerability Ratio (HVR) and associative recovery.

[Source commit](https://github.com/kai9987kai/Supermix-Poseidon/tree/{revision}) ·
[AURA design](docs/AURA_DESIGN.md) · [AURA results](docs/AURA_RESULTS.md) · [Tessera design](docs/TESSERA_DESIGN.md) · [Tessera results](docs/TESSERA_RESULTS.md) · [Mnemorph design](docs/MNEMORPH_DESIGN.md) · [Odysseus design](docs/ODYSSEUS_DESIGN.md) · [MCO design](docs/MCO_DESIGN.md) · [Release guide](docs/HUGGINGFACE_RELEASE.md)

## Components and activation

| Component | Artifact | Status |
|---|---|---|
| Tidal core: 327,230 parameters | `runs/tidal_dagger/core.pt` | Existing selected DAgger policy, default scene/control backend |
| Supervised baseline | `runs/tidal/core.pt` | Comparison checkpoint, original embedded training state |
| SmolLM2: 134,515,008 stored parameters | `models/language/model.safetensors` | Separate unchanged pretrained conversation backend |
| LoRA: 61,440 parameters | `runs/language/adapter/` | Experimental candidate, inactive by default |
| AURA Biomimetic Controller | `poseidon/aura.py` | 16-wedge CX-ring attractor & 128-KC sparse neuropil arbiter, opt-in controller |
| Tessera Macro-Commons | `outputs/tessera_experiments/RECEIPT.json` | Independent dual-lineage ratified macro-actions, opt-in commons |
| Mnemorph Memory Assays | `outputs/mnemorph/RECEIPT.json` | Typed structural lesion and associative regrowth receipts |
| Odysseus Navigator | `outputs/odysseus/atlas.json` | Empirical macro-action transitions & Bayesian replenishment LCB, opt-in candidate |
| Helm Critic v0.6 | `outputs/helm/atlas.json` | Fitted multi-horizon critic with observed history, opt-in candidate |
| Odyssey Atlas v5 | `outputs/odyssey/atlas.json` | Spatial cognitive map and replenishment-decay waypoint planner, opt-in candidate |
| Horizon Atlas v4 | `outputs/horizon/atlas.json` | Fitted multi-horizon return advantage, opt-in planner |
| Contrast Atlas | `outputs/contrast/atlas.json` | Matched-action residual memory, opt-in candidate |
| Counterfactual Atlas | `outputs/atlas/atlas.json` | Fitted residual memory, opt-in candidate |

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
hf download {REPO_ID} --revision v{version} --local-dir Poseidon-release
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
print(runtime.respond('Survive', mode='world', seed=42, planner='odyssey'))
print(runtime.respond('Survive', mode='world', seed=42, planner='horizon'))
```

Default language uses pinned `HuggingFaceTB/SmolLM2-135M-Instruct` at
`{LANGUAGE_REVISION}`. Explicitly test the candidate with
`python -m poseidon chat 'Explain why seasons change.' --adapter runs/language/adapter`.
Generation uses at most a 1,024-token local input budget and 256 generated tokens.
This release does not automatically activate the adapter, Atlas, Contrast, Horizon or Odyssey.

## Training and measured limits

The supervised Tidal run used 120,000 synthetic examples and 360,000 exposures.
DAgger selected round two using validation seeds; 769,600 exposures is the aggregate
over the four-round run, not a precise exposure count of the selected weights.
Stored evaluation reports describe controlled scene vocabulary and synthetic
TidePool survival. Reported 100% results do not imply real-world generalization.

Odyssey Atlas v0.5 constructs an episodic spatial and topological cognitive map
strictly from 16-d observation streams without simulator cheats or global coordinate
leakage. It tracks environmental replenishment dynamics
($\\lambda_{{food}} = 0.007 / scarcity, \\lambda_{{water}} = 0.014 / scarcity$), plans
multi-hop navigational routes via breadth-first search with physical travel cost
penalties, enforces pre-transit rest guards when stamina < 0.20, and prioritizes
shelters during impending storms. The map's scarcity now matches the selected
world settings. The v0.5.0 scarcity-4 suite used the default map setting of 2.5
in 12 episodes; its preserved measurements have that limitation.

Two fresh four-seed, ten-arm, 64-step suites used scarcity 2.5 and 4.0 with no
tuning between them. Calibrated Odyssey matched policy with zero overrides.
Ungated Odyssey's mean episode reward differences were +0.002331 and +0.041827,
but the stress result was dominated by one seed. The separate H=16 auditor
replayed 91 anchors, 546 actual multi-step branches and 8,712 branch transitions.
Six of 11 ungated overrides had negative H-step return advantages. These small
synthetic development suites do not establish superiority or a safety guarantee.
The candidate remains opt-in and the core policy remains the default.

The historical experiment audit forked six one-step reserve-utility branches;
it did not validate multi-step reward margins. The new audit records every
continuation action and replays returns without neural weights. It verifies
execution and arithmetic, not neural authorship. Its strict parent checks also
reconstruct maps, validate fitting partitions, paired cardinality, forecasts
and terminal states. See docs/TRAJECTORY_AUDIT.md and docs/trajectory-evaluation.json.

Horizon Atlas v0.4 indexes multi-horizon returns over H=16 lookahead steps across
12 training episodes (1,728 transitions), evaluates physical depletion guards, and
calibrates empirical advantage error radii across 8 disjoint calibration episodes
(1,152 transitions). Returns use undiscounted rewards. Empirical radii describe
the sampled fitting data and are not calibrated failure probabilities. Historical
episode measurements and their one-step audit limits are in docs/HORIZON_RESULTS.md.

The candidate LoRA ran 256 steps, saw 512 examples and 42,104 tokens, and changed
development loss from 0.960879 to 0.958487. That does not establish better conversation
or reasoning; the dataset overlaps the upstream model's instruction-training
distribution. The candidate remains inactive. The conversation model can hallucinate.

## Provenance and file integrity

Source revision: `{revision}`.
Active core file SHA-256: `{core_hash}`.
Atlas canonical payload SHA-256: `{atlas_hash}`.
Contrast canonical payload SHA-256: `{contrast_hash}`.
Horizon canonical payload SHA-256: `{horizon_hash}`.
Odyssey canonical payload SHA-256: `{odyssey_hash}`.
Helm canonical payload SHA-256: `{helm_hash}`.
`manifest.json` and `SHA256SUMS.txt` inventory the staged files. The manifest binds
source, weights, upstream provenance and the exact experiment receipts.
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
    sys.path.insert(0, str(ROOT))
    from poseidon import __version__
    if __version__ != "0.8.0":
        raise ValueError(f"unsupported version {__version__}")
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
    from poseidon.contrast import ContrastAtlas
    from poseidon.horizon import HorizonAtlas
    from poseidon.odyssey import OdysseyAtlas
    from poseidon.core import CoreRuntime, active_core_path
    from poseidon.experiments import load_and_verify
    from poseidon.trajectory_audit import load_artifacts, load_json, verify_parent, verify_audit
    torch.set_num_threads(2)
    core_path = active_core_path(ROOT)
    core_hash = sha256(core_path)
    pointer = json.loads((ROOT / "runs/active_core.json").read_text(encoding="utf-8"))
    if pointer.get("sha256") != core_hash:
        raise ValueError("active pointer does not identify actual checkpoint bytes")
    core = CoreRuntime(core_path)
    atlas = CounterfactualAtlas.load(core, ROOT / "outputs/atlas/atlas.json")
    atlas_hash = atlas.artifact["sha256"]
    contrast = ContrastAtlas.load(core, ROOT / "outputs/contrast/atlas.json")
    contrast_hash = contrast.artifact["sha256"]
    horizon = HorizonAtlas.load(core, ROOT / "outputs/horizon/atlas.json")
    horizon_hash = horizon.artifact["sha256"]
    odyssey = OdysseyAtlas.load(core, ROOT / "outputs/odyssey/atlas.json")
    odyssey_hash = odyssey.artifact["sha256"]
    from poseidon.helm import HelmCritic, stable_json
    from poseidon.helm_experiment import verify_helm_receipt
    helm = HelmCritic.load(core, ROOT / "outputs/helm/atlas.json")
    helm_hash = helm.artifact["sha256"]
    evaluation = stable_json(ROOT / "docs/helm-evaluation.json")
    helm_names = evaluation.get("receipts")
    if (evaluation.get("status") != "completed-evaluation" or not isinstance(helm_names, list) or
            not 2 <= len(helm_names) <= 8 or len(set(helm_names)) != len(helm_names) or
            any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".json") for name in helm_names)):
        raise ValueError("Helm release requires a frozen complete evaluation profile")
    helm_experiment_verification = {
        name: verify_helm_receipt(stable_json(ROOT / "outputs/helm_experiments" / name)) for name in helm_names}
    if any(stable_json(ROOT / "outputs/helm_experiments" / name)["critic"]["sha256"] != helm_hash for name in helm_names):
        raise ValueError("Helm evaluation is bound to a different fitted critic")

    from poseidon.odysseus import OdysseusAtlas
    from poseidon.odysseus_experiment import verify_odysseus_receipt
    from poseidon.mco import verify_mco_receipt
    odysseus = OdysseusAtlas.load(core, ROOT / "outputs/odysseus/atlas.json")
    odysseus_hash = odysseus.artifact["sha256"]
    odysseus_experiment_verification = {
        name: verify_odysseus_receipt(json.loads((ROOT / "outputs/odysseus_experiments" / name).read_text(encoding="utf-8")))
        for name in ODYSSEUS_EXPERIMENTS
    }
    mco_experiment_verification = {
        "RECEIPT.json": verify_mco_receipt(json.loads((ROOT / "outputs/mco_experiments/RECEIPT.json").read_text(encoding="utf-8")))
    }

    from poseidon.aura_experiment import verify_aura_receipt
    from poseidon.tessera_experiment import verify_tessera_receipt
    from poseidon.mnemorph import verify_mnemorph_receipt
    aura_experiment_verification = {
        name: verify_aura_receipt(json.loads((ROOT / "outputs/aura_experiments" / name).read_text(encoding="utf-8")))
        for name in AURA_EXPERIMENTS
    }
    tessera_experiment_verification = {
        "RECEIPT.json": verify_tessera_receipt(json.loads((ROOT / "outputs/tessera_experiments/RECEIPT.json").read_text(encoding="utf-8")))
    }
    mnemorph_experiment_verification = {
        "RECEIPT.json": verify_mnemorph_receipt(json.loads((ROOT / "outputs/mnemorph/RECEIPT.json").read_text(encoding="utf-8")))
    }

    experiment_verification = {name: load_and_verify(ROOT / "outputs/experiments" / name) for name in EXPERIMENTS}
    evidence_artifacts = load_artifacts(ROOT)
    horizon_experiment_verification = {name: verify_parent(load_json(ROOT / "outputs/horizon_experiments" / name), evidence_artifacts) for name in HORIZON_EXPERIMENTS}
    odyssey_experiment_verification = {name: verify_parent(load_json(ROOT / "outputs/odyssey_experiments" / name), evidence_artifacts) for name in ODYSSEY_EXPERIMENTS}
    trajectory_audit_verification = {
        name: {"parent_path": "outputs/odyssey_experiments/" + parent,
               **verify_audit(load_json(ROOT / "outputs/trajectory_audits" / name),
                              load_json(ROOT / "outputs/odyssey_experiments" / parent), evidence_artifacts)}
        for name, parent in TRAJECTORY_AUDITS.items()
    }

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

    copied_sources = {}

    def copy(relative: str, target: str | None = None) -> None:
        requested = ROOT / relative
        source = requested.resolve()
        if not source.is_relative_to(ROOT) or requested.is_symlink() or not source.is_file():
            raise ValueError(f"invalid source artifact: {relative}")
        destination = output / (target or relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256(source)
        if relative in copied_sources and copied_sources[relative] != expected:
            raise ValueError(f"source artifact changed while packaging: {relative}")
        shutil.copyfile(source, destination)
        if sha256(destination) != expected or sha256(source) != expected:
            raise ValueError(f"source artifact changed while copying: {relative}")
        copied_sources[relative] = expected

    output.mkdir(parents=True)
    source_files = git("ls-files", "-z").split("\0")
    for name in source_files:
        if name:
            target = {"README.md": "SOURCE_README.md", ".gitignore": "SOURCE_GITIGNORE"}.get(name)
            copy(name, target)
    source_hashes = {name: sha256(ROOT / name) for name in source_files if name}
    # The Hub's preupload API applies incoming .gitignore to weight additions.
    # Source ignores training artifacts; a model release must include them.
    (output / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.cache/huggingface/\n*.egg-info/\n", encoding="utf-8")

    artifacts = [
        "runs/tidal_dagger/core.pt", "runs/tidal/core.pt",
        "runs/language/adapter/adapter_model.safetensors",
        "outputs/atlas/atlas.json", "outputs/atlas/fit_receipt.json",
        "outputs/contrast/atlas.json", "outputs/contrast/fit_receipt.json",
        "outputs/horizon/atlas.json", "outputs/horizon/fit_receipt.json",
        "outputs/odyssey/atlas.json", "outputs/odyssey/fit_receipt.json",
        "outputs/helm/atlas.json", "outputs/helm/fit_receipt.json",
        "outputs/odysseus/atlas.json", "outputs/odysseus/fit_receipt.json",
        "outputs/mco_experiments/RECEIPT.json",
        "outputs/tessera_experiments/RECEIPT.json",
        "outputs/mnemorph/RECEIPT.json",
        "models/language/manifest.json", "data/language/manifest.json",
        "data/language/SOURCE_CARD.md",
    ] + [f"models/language/{name}" for name in upstream["files"]]
    artifacts += [f"outputs/experiments/{name}" for name in EXPERIMENTS]
    artifacts += [f"outputs/horizon_experiments/{name}" for name in HORIZON_EXPERIMENTS]
    artifacts += [f"outputs/odyssey_experiments/{name}" for name in ODYSSEY_EXPERIMENTS]
    artifacts += [f"outputs/trajectory_audits/{name}" for name in TRAJECTORY_AUDITS]
    artifacts += [f"outputs/helm_experiments/{name}" for name in helm_names]
    artifacts += [f"outputs/odysseus_experiments/{name}" for name in ODYSSEUS_EXPERIMENTS]
    artifacts += [f"outputs/aura_experiments/{name}" for name in AURA_EXPERIMENTS]
    for relative in artifacts:
        copy(relative)
    copy("docs/history/HF_MODEL_CARD_v0.1.md", "legacy/model-card-before-v0.2.md")
    copy("runs/beyond_adapted/ensemble_adapted.pt", "legacy/ensemble_adapted.pt")
    shutil.copyfile(historical_assets[".gitattributes"], output / ".gitattributes")
    # The Hub adds exact LFS rules for newly uploaded large JSON receipts. Stage
    # them before hashing so the remote commit cannot silently change this file.
    attributes_path = output / ".gitattributes"
    attributes = attributes_path.read_text(encoding="utf-8").rstrip("\n")
    rules = set(attributes.splitlines())
    for path in sorted(output.rglob("*.json")):
        if path.stat().st_size >= 10 * 1024 * 1024:
            rule = path.relative_to(output).as_posix() + " filter=lfs diff=lfs merge=lfs -text"
            if rule not in rules:
                attributes += "\n" + rule
                rules.add(rule)
    attributes_path.write_text(attributes + "\n", encoding="utf-8")
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
    (output / "README.md").write_text(model_card(revision, core_hash, atlas_hash, horizon_hash, contrast_hash, odyssey_hash, __version__, helm_hash, odysseus_hash), encoding="utf-8")

    if git("status", "--porcelain") or git("rev-parse", "HEAD") != revision:
        raise ValueError("source checkout changed while packaging")
    for name, expected in source_hashes.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"source changed while packaging: {name}")
    for name, expected in copied_sources.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"source artifact changed while packaging: {name}")
    if (sha256(core_path) != core_hash or
            sha256(ROOT / "outputs/atlas/atlas.json") != sha256(output / "outputs/atlas/atlas.json") or
            sha256(ROOT / "outputs/odyssey/atlas.json") != sha256(output / "outputs/odyssey/atlas.json") or
            sha256(ROOT / "outputs/odysseus/atlas.json") != sha256(output / "outputs/odysseus/atlas.json")):
        raise ValueError("core, Atlas, Odyssey, or Odysseus changed while packaging")

    files = [{"path": p.relative_to(output).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    manifest = {
        "schema": "poseidon-huggingface-release-v1", "repo_id": repo_id,
        "version": __version__, "source_revision": revision, "files": files,
        "previous_hub_revision": PREVIOUS_HUB_REVISION,
        "historical_artifacts": ["legacy/ensemble_adapted.pt", "legacy/replay-before-v0.2.mp4",
                                 "legacy/beyond-audit-before-v0.2.json", "legacy/model-card-before-v0.2.md"],
        "core_sha256": core_hash, "atlas_artifact_sha256": atlas_hash,
        "contrast_artifact_sha256": contrast_hash,
        "horizon_artifact_sha256": horizon_hash,
        "odyssey_artifact_sha256": odyssey_hash,
        "helm_artifact_sha256": helm_hash,
        "odysseus_artifact_sha256": odysseus_hash,
        "upstream_language": upstream, "source_file_sha256": source_hashes,
        "release_transforms": {"README.md": "Hub model card; original becomes SOURCE_README.md",
                               ".gitignore": "Hub upload rules; original becomes SOURCE_GITIGNORE",
                               ".gitattributes": "historical LFS rules plus explicit large JSON receipt paths",
                               "runs/language/adapter/README.md": "authored inactive candidate card",
                               "runs/language/adapter/adapter_config.json": "portable upstream id and pinned revision"},
        "experiment_verification": experiment_verification,
        "horizon_experiment_verification": horizon_experiment_verification,
        "odyssey_experiment_verification": odyssey_experiment_verification,
        "trajectory_audit_verification": trajectory_audit_verification,
        "helm_experiment_verification": helm_experiment_verification,
        "odysseus_experiment_verification": odysseus_experiment_verification,
        "mco_experiment_verification": mco_experiment_verification,
        "aura_experiment_verification": aura_experiment_verification,
        "tessera_experiment_verification": tessera_experiment_verification,
        "mnemorph_experiment_verification": mnemorph_experiment_verification,
        "build_environment": {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "safetensors", "numpy", "huggingface-hub")},
        "activation": {"core": "existing-dagger-selection", "language": "unchanged-upstream-base",
                       "language_adapter": "inactive-candidate", "atlas": "opt-in-experiment",
                       "contrast": "opt-in-experiment", "horizon": "opt-in-experiment",
                       "odyssey": "opt-in-experiment",
                       "helm": "opt-in-experiment",
                       "odysseus": "opt-in-experiment",
                       "mco": "factual-observatory-benchmark",
                       "aura": "opt-in-biomimetic-controller",
                       "tessera": "ratified-macro-commons",
                       "mnemorph": "archaeological-memory-assays",
                       "legacy_ensemble": "historical-unvalidated-not-loaded"},
    }
    write_json(output / "manifest.json", manifest)
    sums = files + [{"path": "manifest.json", "sha256": sha256(output / "manifest.json")}]
    (output / "SHA256SUMS.txt").write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in sums), encoding="utf-8")
    return {"output": str(output), "source_revision": revision, "files": len(files) + 2,
            "bytes": sum(row["size"] for row in files), "manifest_sha256": sha256(output / "manifest.json"),
            "core_sha256": core_hash, "atlas_artifact_sha256": atlas_hash,
            "contrast_artifact_sha256": contrast_hash, "horizon_artifact_sha256": horizon_hash,
            "odyssey_artifact_sha256": odyssey_hash, "helm_artifact_sha256": helm_hash,
            "odysseus_artifact_sha256": odysseus_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.repo_id), indent=2))


if __name__ == "__main__":
    main()
