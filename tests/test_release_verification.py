"""Model-free tests for the downloaded release's integrity boundary."""
import hashlib
import json

import pytest

from tools import verify_huggingface_release as release


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _seal(root, manifest):
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (root / "manifest.json").write_bytes(encoded)
    sums = [f'{row["sha256"]}  {row["path"]}' for row in manifest["files"]]
    sums.append(f"{_digest(encoded)}  manifest.json")
    (root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")


def _package(root, version="0.5.1"):
    payload = b"bounded test payload\n"
    (root / "model.bin").write_bytes(payload)
    manifest = {
        "schema": release.SCHEMA, "version": version,
        "source_revision": "1" * 40, "repo_id": "Kai9987kai/Poseidon",
        "core_sha256": "2" * 64, "atlas_artifact_sha256": "3" * 64,
        "files": [{"path": "model.bin", "size": len(payload), "sha256": _digest(payload)}],
    }
    if version in ("0.4.0", "0.5.0", "0.5.1", "0.6.0", "0.7.0", "0.8.0"):
        manifest.update(horizon_artifact_sha256="4" * 64, contrast_artifact_sha256="5" * 64)
    if version in ("0.5.0", "0.5.1", "0.6.0", "0.7.0", "0.8.0"):
        manifest["odyssey_artifact_sha256"] = "6" * 64
    if version in ("0.6.0", "0.7.0", "0.8.0"):
        manifest["helm_artifact_sha256"] = "7" * 64
        manifest["helm_experiment_verification"] = {"test.json": {"verified": True}}
    if version in ("0.7.0", "0.8.0"):
        manifest["odysseus_artifact_sha256"] = "8" * 64
        manifest["odysseus_experiment_verification"] = {"test.json": {"verified": True}}
        manifest["mco_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    if version == "0.8.0":
        manifest["aura_experiment_verification"] = {"test.json": {"verified": True}}
        manifest["tessera_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
        manifest["mnemorph_experiment_verification"] = {"RECEIPT.json": {"verified": True}}
    _seal(root, manifest)
    return manifest


@pytest.mark.parametrize("version", ("0.2.0", "0.4.0", "0.5.0", "0.5.1", "0.6.0", "0.7.0", "0.8.0"))
def test_hash_verification_preserves_supported_release_versions(tmp_path, version):
    manifest = _package(tmp_path, version)
    actual, receipt = release.verify_hashes(tmp_path)
    assert actual == manifest
    assert receipt["verified"] and receipt["version"] == version
    assert receipt["files_verified"] == 1
    assert receipt["payload_bytes"] == (tmp_path / "model.bin").stat().st_size


@pytest.mark.parametrize("key", ("helm_artifact_sha256", "helm_experiment_verification"))
def test_helm_release_requires_bound_critic_and_evidence(tmp_path, key):
    manifest = _package(tmp_path, "0.6.0")
    manifest.pop(key)
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("key", ("odysseus_artifact_sha256", "odysseus_experiment_verification", "mco_experiment_verification"))
def test_odysseus_and_mco_release_requires_bound_evidence(tmp_path, key):
    manifest = _package(tmp_path, "0.7.0")
    manifest.pop(key)
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("digest_key", (
    "horizon_artifact_sha256", "contrast_artifact_sha256", "odyssey_artifact_sha256",
))
@pytest.mark.parametrize("invalid", (None, "not-a-digest"))
def test_patch_release_requires_all_planning_artifact_digests(tmp_path, digest_key, invalid):
    manifest = _package(tmp_path)
    if invalid is None:
        manifest.pop(digest_key)
    else:
        manifest[digest_key] = invalid
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="Invalid release digest: " + digest_key):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("payload", (b"different contents!!\n", b"short"))
def test_payload_tampering_rejected_before_model_loading(tmp_path, payload):
    _package(tmp_path)
    (tmp_path / "model.bin").write_bytes(payload)
    with pytest.raises(ValueError, match="size or checksum mismatch"):
        release.verify_hashes(tmp_path)


def test_missing_payload_is_rejected(tmp_path):
    _package(tmp_path)
    (tmp_path / "model.bin").unlink()
    with pytest.raises(ValueError, match="file is missing"):
        release.verify_hashes(tmp_path)


def test_unmanifested_source_is_rejected(tmp_path):
    _package(tmp_path)
    (tmp_path / "extra.py").write_text("print('unexpected source')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs from manifest"):
        release.verify_hashes(tmp_path)


def test_generated_metadata_allowed_but_manifested_cache_payload_still_checked(tmp_path):
    manifest = _package(tmp_path)
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text("{}", encoding="utf-8")
    payload = b"manifested cached payload"
    (cache / "payload.bin").write_bytes(payload)
    manifest["files"].append({"path": ".cache/huggingface/payload.bin", "size": len(payload),
                              "sha256": _digest(payload)})
    _seal(tmp_path, manifest)
    _, receipt = release.verify_hashes(tmp_path)
    assert receipt["files_verified"] == 2
    assert receipt["ignored_local_metadata"]["directories"] == [".cache/huggingface"]
    (cache / "payload.bin").write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="size or checksum mismatch"):
        release.verify_hashes(tmp_path)


def test_sha256sums_must_bind_the_manifest_itself(tmp_path):
    manifest = _package(tmp_path)
    manifest["repo_id"] = "another-owner/Poseidon"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256SUMS.txt does not match"):
        release.verify_hashes(tmp_path)


@pytest.mark.parametrize("path", ("../model.bin", "outputs\\model.bin"))
def test_noncanonical_or_escaping_inventory_paths_rejected(tmp_path, path):
    manifest = _package(tmp_path)
    manifest["files"][0]["path"] = path
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="Package paths"):
        release.verify_hashes(tmp_path)


def test_case_colliding_manifest_entries_rejected_on_all_platforms(tmp_path):
    manifest = _package(tmp_path)
    manifest["files"].append({**manifest["files"][0], "path": "MODEL.BIN"})
    _seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="Duplicate, colliding, or reserved"):
        release.verify_hashes(tmp_path)


def test_isolated_smoke_script_is_valid_python():
    compile(release.SMOKE_SCRIPT, "packaged-release-smoke", "exec")
