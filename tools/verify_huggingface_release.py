"""Verify a downloaded Poseidon release before loading its packaged models.

Run ``python tools/verify_huggingface_release.py PACKAGE --smoke``. Hash-only
verification has no third-party dependencies. Smoke verification imports only the
packaged Poseidon source in a separate, offline subprocess; it does not activate
the candidate LoRA or write evaluation outputs.

Downloaded/installed packages may contain local ``.cache/huggingface/`` metadata,
root ``*.egg-info/`` and ``.pytest_cache/``, and ``__pycache__/`` directories.
Only these generated directories are ignored in the inventory. Their contents
are still checked for symlinks/junctions; every manifest file remains verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

SCHEMA = "poseidon-huggingface-release-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
RESERVED = {"manifest.json", "SHA256SUMS.txt"}
IGNORED_INVENTORY_RULES = (
    ".cache/huggingface/ (Hub download metadata)",
    "*.egg-info/ at package root (editable install metadata)",
    ".pytest_cache/ at package root (test metadata)",
    "__pycache__/ at any depth (Python bytecode cache)",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Package paths must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in (".", "..") for part in path.parts):
        raise ValueError("Package paths must be canonical and stay inside the package")
    if any(ord(character) < 32 for character in value):
        raise ValueError("Package paths may not contain control characters")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(_relative(relative)).parts)
    for parent in (path, *path.parents):
        if _link(parent):
            raise ValueError("Package contains a symlink or junction: " + relative)
        if parent == root:
            break
    if not path.resolve().is_relative_to(root):
        raise ValueError("Package path escapes its root: " + relative)
    if not path.is_file():
        raise ValueError("Package file is missing: " + relative)
    return path


def _metadata_directory(relative: str) -> str | None:
    """Return the allowlisted metadata directory containing this local path."""
    parts = PurePosixPath(relative).parts
    if len(parts) >= 2 and parts[:2] == (".cache", "huggingface"):
        return ".cache/huggingface"
    if parts and (parts[0].endswith(".egg-info") or parts[0] == ".pytest_cache"):
        return parts[0]
    if "__pycache__" in parts:
        return "/".join(parts[:parts.index("__pycache__") + 1])
    return None


def _inventory(root: Path) -> tuple[set[str], dict]:
    """Inspect ignored directories too, rejecting links before traversing them."""
    actual, ignored_directories, ignored_files = set(), set(), 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if _link(path) or not path.resolve().is_relative_to(root):
                    raise ValueError("Package contains a symlink, junction or escaping path: " + relative)
                metadata = _metadata_directory(relative)
                if entry.is_dir(follow_symlinks=False):
                    if metadata:
                        ignored_directories.add(metadata)
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    if metadata:
                        # Only files inside a generated directory are ignored;
                        # a source/model file named '*.egg-info' is not metadata.
                        if relative != metadata:
                            ignored_directories.add(metadata)
                            ignored_files += 1
                            continue
                    actual.add(relative)
                else:
                    raise ValueError("Package contains a non-regular file: " + relative)
    return actual, {"rules": list(IGNORED_INVENTORY_RULES),
                    "directories": sorted(ignored_directories), "files_ignored": ignored_files}


def verify_hashes(package_dir: str | Path) -> tuple[dict, dict]:
    requested = Path(package_dir).absolute()
    if _link(requested) or not requested.is_dir():
        raise ValueError("Package root must be an existing directory, not a link")
    root = requested.resolve()
    manifest_path = _inside(root, "manifest.json")
    sums_path = _inside(root, "SHA256SUMS.txt")
    if manifest_path.stat().st_size > 4 * 1024 * 1024 or sums_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Package metadata exceeds the 4 MiB limit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported release manifest schema")
    if manifest.get("version") != "0.2.0" or not HEX40.fullmatch(str(manifest.get("source_revision", ""))):
        raise ValueError("Invalid release version or source revision")
    if not isinstance(manifest.get("repo_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest["repo_id"]):
        raise ValueError("Invalid Hugging Face repository identity")
    for key in ("core_sha256", "atlas_artifact_sha256"):
        if not isinstance(manifest.get(key), str) or not HEX64.fullmatch(manifest[key]):
            raise ValueError("Invalid release digest: " + key)
    files = manifest.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 10000:
        raise ValueError("Manifest must contain 1-10000 files")
    expected, seen = {}, set()
    total_bytes = 0
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise ValueError("Invalid manifest file record")
        relative = _relative(row["path"])
        if relative.casefold() in seen or relative in RESERVED:
            raise ValueError("Duplicate, colliding, or reserved manifest path: " + relative)
        seen.add(relative.casefold())
        if type(row["size"]) is not int or row["size"] < 0 or not isinstance(row["sha256"], str) or not HEX64.fullmatch(row["sha256"]):
            raise ValueError("Invalid file size or digest: " + relative)
        path = _inside(root, relative)
        if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            raise ValueError("Package file size or checksum mismatch: " + relative)
        expected[relative] = row["sha256"]
        total_bytes += row["size"]
    actual, ignored_metadata = _inventory(root)
    # A manifest entry inside a cache is still hash-verified and inventoried.
    actual.update(path for path in expected if _metadata_directory(path))
    if actual != set(expected) | RESERVED:
        raise ValueError("Package inventory differs from manifest: " + ", ".join(sorted(actual ^ (set(expected) | RESERVED))[:8]))
    expected["manifest.json"] = sha256_file(manifest_path)
    recorded = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("Malformed SHA256SUMS.txt entry")
        relative = _relative(match[2])
        if relative in recorded:
            raise ValueError("Duplicate SHA256SUMS.txt path")
        recorded[relative] = match[1]
    if recorded != expected:
        raise ValueError("SHA256SUMS.txt does not match manifest files and manifest.json")
    receipt = {
        "verified": True, "schema": SCHEMA, "repo_id": manifest["repo_id"],
        "source_revision": manifest["source_revision"], "version": manifest["version"],
        "files_verified": len(files), "payload_bytes": total_bytes,
        "manifest_sha256": expected["manifest.json"], "sums_sha256": sha256_file(sums_path),
        "core_sha256": manifest["core_sha256"], "atlas_artifact_sha256": manifest["atlas_artifact_sha256"],
        "ignored_local_metadata": ignored_metadata,
    }
    return manifest, receipt


SMOKE_SCRIPT = r'''
import gc, hashlib, json, math, pathlib, sys
root = pathlib.Path.cwd().resolve()
import poseidon
assert pathlib.Path(poseidon.__file__).resolve().is_relative_to(root), "Smoke imported live source"
from poseidon.runtime import Poseidon
from poseidon.language import LanguageRuntime
from poseidon.world import TidePool, rollout
from poseidon.experiments import load_and_verify
import torch
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
torch.set_num_threads(2)
torch.manual_seed(0)
runtime = Poseidon(root)
status = runtime.status()
assert status["version"] == manifest["version"]
assert status["core_ready"] and status["language_ready"] and status["atlas"]["ready"], "Packaged model readiness failed"
core, atlas = runtime.core(), runtime.atlas()
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()
assert digest(core.path) == manifest["core_sha256"], "Active core digest differs from manifest"
pointer = json.loads((root / "runs/active_core.json").read_text(encoding="utf-8"))
assert pointer["sha256"] == manifest["core_sha256"], "Active core pointer has a stale digest"
assert atlas.artifact["sha256"] == manifest["atlas_artifact_sha256"], "Atlas digest differs from manifest"
def finite(value):
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), 1048576):
            assert torch.isfinite(flat[start:start+1048576]).all().item(), "Non-finite native model tensor"
    elif isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, (tuple, list)):
        for child in value: finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value), "Non-finite native model value"
native_checkpoints = sorted(root.glob("runs/**/*.pt"))
assert native_checkpoints, "Native checkpoint missing"
for checkpoint in native_checkpoints:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    finite(payload)
scene = core.scene("Create two small cyan spheres with orbit motion.")
assert {k:scene[k] for k in ("shape","color","motion","count","scale")} == {"shape":"sphere","color":"cyan","motion":"orbit","count":2,"scale":"small"}, "Controlled scene prediction failed"
finite(scene)
observation = TidePool(99000001, max_steps=32).observe()
decision = atlas.plan(observation)
assert len(decision["candidates"]) == 6 and decision["artifact_sha256"] == manifest["atlas_artifact_sha256"]
finite(decision)
worlds = {}
for name, controller in (("policy", core), ("atlas", atlas)):
    episode = rollout(controller, seed=99000001, max_steps=32)
    assert 1 <= episode["steps"] <= 32 and isinstance(episode["survived"], bool)
    finite(episode)
    worlds[name] = {k:episode[k] for k in ("steps","survived","reward","death_reason")}
arithmetic = runtime.respond("3*x + 7 = 22", mode="math")
assert arithmetic["verified"] and arithmetic["answer"] == "5", "Exact arithmetic failed"
del runtime, core, atlas
gc.collect()
languages = {}
for name, adapter in (("base", None), ("candidate_lora", root / "runs/language/adapter")):
    if adapter is not None:
        assert (adapter / "adapter_model.safetensors").is_file(), "Candidate LoRA is missing"
    language = LanguageRuntime(root / "models/language", adapter=adapter, threads=2)
    first = language.generate("Say hello briefly.", max_new_tokens=8)
    second = language.generate("Say hello briefly.", max_new_tokens=8)
    assert first == second and 1 <= first["generated_tokens"] <= 8 and first["text"], "Language generation is not bounded and deterministic"
    finite(dict(language.model.named_parameters()))
    finite(dict(language.model.named_buffers()))
    languages[name] = first
    del language
    gc.collect()
replays = []
for path in sorted((root / "outputs/experiments").glob("*.json")):
    replays.append({"path":path.relative_to(root).as_posix(), **load_and_verify(path)})
assert replays, "Packaged experiment receipts are missing"
print("POSEIDON_RELEASE_SMOKE=" + json.dumps({
    "verified": True, "package_source": str(pathlib.Path(poseidon.__file__).resolve()),
    "native_checkpoints_verified": len(native_checkpoints), "scene": {k:scene[k] for k in ("shape","color","motion","count","scale")},
    "worlds": worlds, "arithmetic": arithmetic["answer"], "languages": languages,
    "receipts_verified": len(replays), "episodes_replayed":sum(x["episodes_replayed"] for x in replays),
    "transitions_replayed":sum(x["transitions_replayed"] for x in replays), "replays":replays,
}, allow_nan=False, sort_keys=True))
'''


def verify_package(package_dir: str | Path, *, smoke: bool = False, timeout: int = 900) -> dict:
    manifest, receipt = verify_hashes(package_dir)
    if smoke:
        root = Path(package_dir).resolve()
        environment = os.environ.copy()
        environment.update({"PYTHONPATH": str(root), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
        result = subprocess.run([sys.executable, "-B", "-c", SMOKE_SCRIPT], cwd=root, env=environment,
                                capture_output=True, text=True, timeout=timeout, check=False)
        marker = "POSEIDON_RELEASE_SMOKE="
        lines = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
        if result.returncode != 0 or len(lines) != 1:
            raise ValueError("Isolated packaged smoke verification failed:\n" + (result.stderr or result.stdout)[-6000:])
        receipt["smoke"] = json.loads(lines[0])
        # Inference must not change the downloadable package or any receipt.
        _, after = verify_hashes(root)
        protected = set(receipt) - {"smoke", "ignored_local_metadata"}
        if any(after[key] != receipt[key] for key in protected):
            raise ValueError("Packaged bytes changed during smoke verification")
        receipt["ignored_local_metadata"] = after["ignored_local_metadata"]
    else:
        receipt["smoke"] = {"verified": False, "reason": "not requested"}
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Load packaged models and independently replay experiment receipts offline")
    parser.add_argument("--timeout", type=int, default=900, help="Maximum isolated smoke seconds (default: 900)")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 3600:
        parser.error("--timeout must be between 1 and 3600 seconds")
    try:
        receipt = verify_package(args.package_dir, smoke=args.smoke, timeout=args.timeout)
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"verified": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
