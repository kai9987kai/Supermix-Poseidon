"""Composition boundary: every result declares its actual backend."""
from __future__ import annotations
from pathlib import Path
import copy
import json
import threading
from . import __version__
from .reasoning import extract_math, solve
from .language import LanguageRuntime

class Poseidon:
    def __init__(self, root=".", adapter=None):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.language = LanguageRuntime(self.root/"models/language", adapter=adapter)
        self._core = None
        self._core_stamp = None
        self._atlas = None
        self._atlas_stamp = None
        self._contrast = None
        self._contrast_stamp = None
        self._horizon = None
        self._horizon_stamp = None
        self._odyssey = None
        self._odyssey_stamp = None
        self._helm = None
        self._helm_stamp = None
        self._odysseus = None
        self._odysseus_stamp = None
        self._jobs = None
        self._jobs_lock = threading.Lock()
        self._tessera = None
        self._candidate_status_snapshots = {}
        self.memory_path = self.root/"outputs/memory.json"

    def atlas(self):
        from .atlas import CounterfactualAtlas
        path = self.root / "outputs/atlas/atlas.json"
        if not path.is_file():
            raise RuntimeError("Counterfactual Atlas is not fitted. Run python -m poseidon atlas-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._atlas is None or self._atlas_stamp != stamp:
            self._atlas = CounterfactualAtlas.load(core, path)
            self._atlas_stamp = stamp
        return self._atlas

    def atlas_status(self):
        return self._candidate_status("atlas")

    def contrast(self):
        from .contrast import ContrastAtlas
        path = self.root / "outputs/contrast/atlas.json"
        if not path.is_file():
            raise RuntimeError("Contrast Atlas is not fitted. Run python -m poseidon contrast-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._contrast is None or self._contrast_stamp != stamp:
            self._contrast = ContrastAtlas.load(core, path)
            self._contrast_stamp = stamp
        return self._contrast

    def contrast_status(self):
        return self._candidate_status("contrast")

    def horizon(self):
        from .horizon import HorizonAtlas
        path = self.root / "outputs/horizon/atlas.json"
        if not path.is_file():
            raise RuntimeError("Horizon Atlas is not fitted. Run python -m poseidon horizon-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._horizon is None or self._horizon_stamp != stamp:
            self._horizon = HorizonAtlas.load(core, path)
            self._horizon_stamp = stamp
        return self._horizon

    def horizon_status(self):
        return self._candidate_status("horizon")

    def odyssey(self):
        from .odyssey import OdysseyAtlas
        path = self.root / "outputs/odyssey/atlas.json"
        if not path.is_file():
            raise RuntimeError("Odyssey Atlas is not fitted. Run python -m poseidon odyssey-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._odyssey is None or self._odyssey_stamp != stamp:
            self._odyssey = OdysseyAtlas.load(core, path)
            self._odyssey_stamp = stamp
        return self._odyssey

    def odyssey_status(self):
        return self._candidate_status("odyssey")

    def helm(self):
        from .helm import HelmCritic
        path = self.root / "outputs/helm/atlas.json"
        if not path.is_file():
            raise RuntimeError("Helm is not fitted. Run python -m poseidon fit-helm first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._helm is None or self._helm_stamp != stamp:
            self._helm = HelmCritic.load(core, path)
            self._helm_stamp = stamp
        return self._helm

    def helm_status(self):
        return self._candidate_status("helm")

    def odysseus(self):
        from .odysseus import OdysseusAtlas
        path = self.root / "outputs/odysseus/atlas.json"
        if not path.is_file():
            raise RuntimeError("Odysseus Atlas is not fitted. Run python -m poseidon fit-odysseus first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._odysseus is None or self._odysseus_stamp != stamp:
            self._odysseus = OdysseusAtlas.load(core, path)
            self._odysseus_stamp = stamp
        return self._odysseus

    def odysseus_status(self):
        return self._candidate_status("odysseus")

    def tessera(self):
        if self._tessera is None:
            from .tessera import TesseraMacroCommons
            path = self.root / "outputs/tessera/commons.json"
            if path.is_file():
                self._tessera = TesseraMacroCommons.load(path)
            else:
                self._tessera = TesseraMacroCommons()
        return self._tessera

    def tessera_status(self):
        tess = self.tessera()
        return {
            "ready": True,
            "ratified_count": len(tess.ratified),
            "quarantine_count": len(tess.quarantine),
            "current_epoch": tess.current_epoch,
        }

    def aura(self):
        from .aura import AuraController
        return AuraController(core=self.core(), tessera=self.tessera())

    def aura_status(self):
        return {"ready": True, "type": "biomimetic_central_complex_and_neuropil"}

    def mnemorph_status(self):
        relative = "outputs/mnemorph/RECEIPT.json"
        path = self.root / relative
        if not path.is_file():
            return {"ready": False, "path": relative}
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            return {
                "ready": True,
                "path": relative,
                "receipt_sha256": receipt.get("receipt_sha256"),
                "intact_mean": receipt.get("intact_mean_retrieval"),
                "hub_vulnerability": receipt.get("hub_vulnerability_ratio"),
            }
        except Exception as error:
            return {"ready": False, "path": relative, "error": str(error)}

    def jobs(self):
        with self._jobs_lock:
            if self._jobs is None:
                import hashlib
                from .jobs import ExperimentJobs
                from .provenance import assert_source_current
                sources = assert_source_current()
                producer = {"version": __version__, "source_sha256": hashlib.sha256(
                    json.dumps(sources, sort_keys=True).encode()).hexdigest()}
                self._jobs = ExperimentJobs(self.root / "outputs/jobs", producer=producer)
            return self._jobs

    def close(self):
        if self._jobs is not None:
            self._jobs.close()

    def submit_helm_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "helm-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 112000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 32, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        # Fail missing/invalid producer readiness before accepting background work.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.helm()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("helm-experiment", normalized,
            lambda progress: self.helm_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def helm_experiment(self, seed=112000001, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        from .helm_experiment import run_helm_experiment, verify_helm_receipt
        from .experiments import write_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            result = run_helm_experiment(self.core(), self.helm(), seeds=list(range(seed, seed + episodes)),
                max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_helm_receipt(result, progress=check)
            check({"phase": "publishing"})
            path = write_receipt(result, self.root / "outputs/helm_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "selected")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "risk_coverage", "limits") if key in result} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}
        finally:
            self.lock.release()

    def submit_odysseus_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "odysseus-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 133000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 32, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.odysseus()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("odysseus-experiment", normalized,
            lambda progress: self.odysseus_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def odysseus_experiment(self, seed=133000001, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        from .odysseus_experiment import run_odysseus_experiment, verify_odysseus_receipt
        from .experiments import write_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            seeds = list(range(seed, seed + episodes))
            result = run_odysseus_experiment(self.core(), self.odysseus(), seeds=seeds,
                max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_odysseus_receipt(result)
            check({"phase": "publishing"})
            path = write_receipt(result, self.root / "outputs/odysseus_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "odysseus_calibrated")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "limits") if key in result} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}
        finally:
            self.lock.release()

    def submit_mco_experiment(self, settings=None):
        from .provenance import assert_source_current
        assert_source_current()
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {"kind"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "mco-experiment":
            raise ValueError("Unknown experiment job kind.")
        return self.jobs().submit("mco-experiment", settings,
            lambda progress: self.mco_experiment(progress=progress), timeout_seconds=1800)

    def mco_experiment(self, progress=None):
        import json
        from .mco import run_mco_experiment, verify_mco_receipt
        from .provenance import assert_source_current
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            receipt = run_mco_experiment(progress=check)
            check({"phase": "verifying"})
            verification = verify_mco_receipt(receipt)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/mco_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            return {
                "schema": receipt["schema"],
                "experiment_id": receipt["experiment_id"],
                "best_subset": receipt["best_subset"],
                "carrier_contributions": receipt["carrier_contributions"],
                "condition_comparison": receipt["condition_comparison"],
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": receipt["receipt_sha256"],
                "verification": verification,
            }
        finally:
            self.lock.release()

    def submit_aura_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "aura-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 144000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("aura-experiment", normalized,
            lambda progress: self.aura_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def aura_experiment(self, seed=144000001, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        from .aura_experiment import run_aura_experiment, verify_aura_receipt
        from .experiments import write_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            seeds = list(range(seed, seed + episodes))
            result = run_aura_experiment(self.core(), seeds=seeds, max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_aura_receipt(result)
            check({"phase": "publishing"})
            path = write_receipt(result, self.root / "outputs/aura_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "aura")
            return {key: result[key] for key in ("schema", "experiment_id", "summary") if key in result} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}
        finally:
            self.lock.release()

    def submit_tessera_experiment(self, settings=None):
        from .provenance import assert_source_current
        assert_source_current()
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "tessera-experiment":
            raise ValueError("Unknown experiment job kind.")
        return self.jobs().submit("tessera-experiment", settings,
            lambda progress: self.tessera_experiment(progress=progress), timeout_seconds=1800)

    def tessera_experiment(self, seeds=None, progress=None):
        import json
        from .tessera_experiment import run_tessera_experiment, verify_tessera_receipt
        from .provenance import assert_source_current
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        seeds = seeds or [155000001, 155000002, 155000003, 155000004]
        receipt = run_tessera_experiment(seeds=seeds, progress=check)
        verification = verify_tessera_receipt(receipt)
        receipt_dir = self.root / "outputs/tessera_experiments"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        path = receipt_dir / "RECEIPT.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "schema": receipt["schema"],
            "experiment_id": receipt["experiment_id"],
            "ratified_opcodes_count": receipt["ratified_opcodes_count"],
            "quarantined_candidates_count": receipt["quarantined_candidates_count"],
            "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
            "receipt_sha256": receipt["receipt_sha256"],
            "verification": verification,
        }

    def submit_mnemorph_experiment(self, settings=None):
        from .provenance import assert_source_current
        assert_source_current()
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {"kind"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "mnemorph-experiment":
            raise ValueError("Unknown experiment job kind.")
        return self.jobs().submit("mnemorph-experiment", settings,
            lambda progress: self.mnemorph_experiment(progress=progress), timeout_seconds=1800)

    def mnemorph_experiment(self, progress=None):
        import json
        from .mnemorph import MnemorphArchaeology, verify_mnemorph_receipt
        from .provenance import assert_source_current
        assert_source_current()
        arch = MnemorphArchaeology()
        receipt = arch.run_archaeological_study()
        verification = verify_mnemorph_receipt(receipt)
        receipt_dir = self.root / "outputs/mnemorph"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        path = receipt_dir / "RECEIPT.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "schema": receipt["schema"],
            "intact_mean_retrieval": receipt["intact_mean_retrieval"],
            "hub_lesion_mean_retrieval": receipt["hub_lesion_mean_retrieval"],
            "periphery_lesion_mean_retrieval": receipt["periphery_lesion_mean_retrieval"],
            "regrowth_mean_retrieval": receipt["regrowth_mean_retrieval"],
            "associative_regrowth_recovery_rate": receipt["associative_regrowth_recovery_rate"],
            "hub_vulnerability_ratio": receipt["hub_vulnerability_ratio"],
            "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
            "receipt_sha256": receipt["receipt_sha256"],
            "verification": verification,
        }

    def _candidate_status(self, name):
        """Readiness may be cached while a world or experiment owns the model."""
        relative = f"outputs/{name}/atlas.json"
        if not self.lock.acquire(blocking=False):
            snapshot = copy.deepcopy(self._candidate_status_snapshots.get(name, {
                "ready": False, "path": relative, "error": "Readiness check pending while Poseidon is busy.",
            }))
            return snapshot | {"status_cached": True, "busy": True}
        try:
            path = self.root / relative
            if not path.is_file():
                result = {"ready": False, "path": relative}
            else:
                try:
                    payload = getattr(self, name)().artifact
                    receipt = {key: payload[key] for key in ("partition", "selection", "calibration", "config") if key in payload}
                    if "records" in payload:
                        receipt["training_samples"] = len(payload["records"])
                    result = {"ready": True, "path": relative, "artifact_sha256": payload["sha256"],
                              "fit_receipt": receipt}
                except (ValueError, TypeError, OSError, RuntimeError, ImportError) as error:
                    result = {"ready": False, "path": relative, "error": str(error)}
            self._candidate_status_snapshots[name] = copy.deepcopy(result)
            return result | {"status_cached": False, "busy": False}
        finally:
            self.lock.release()

    def experiment(self, seed=93000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec, run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "atlas")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "atlas", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def contrast_experiment(self, seed=104000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec
        from .contrast_experiments import run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), self.contrast(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/contrast_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "contrast")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "contrast", "prediction_comparison", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def horizon_experiment(self, seed=106000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec
        from .horizon_experiments import run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), self.contrast(), self.horizon(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/horizon_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "horizon")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "horizon", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def odyssey_experiment(self, seed=108000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec
        from .odyssey_experiments import run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), self.contrast(), self.horizon(), self.odyssey(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/odyssey_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "odyssey")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "odyssey", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def core(self):
        from .core import active_core_path
        path = active_core_path(self.root)
        if not path.exists():
            raise RuntimeError("The Tidal checkpoint is not ready yet. Check training progress.")
        stamp = (str(path), path.stat().st_mtime_ns)
        if self._core is None or self._core_stamp != stamp:
            import torch
            from .core import CoreRuntime
            torch.set_num_threads(2)
            self._core = CoreRuntime(path)
            self._core_stamp = stamp
        return self._core

    def mco_status(self):
        relative = "outputs/mco_experiments/RECEIPT.json"
        path = self.root / relative
        if not path.is_file():
            return {"ready": False, "path": relative}
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            return {
                "ready": receipt.get("status") == "completed",
                "path": relative,
                "receipt_sha256": receipt.get("receipt_sha256"),
                "best_subset": receipt.get("best_subset"),
                "carrier_contributions": receipt.get("carrier_contributions"),
            }
        except Exception as error:
            return {"ready": False, "path": relative, "error": str(error)}

    def status(self):
        reports = {}
        for name, relative in [("core", "runs/tidal/receipt.json"), ("language", "runs/language/report.json"), ("evaluation", "runs/evaluation.json"), ("language_data", "data/language/manifest.json"), ("dagger", "runs/tidal_dagger/report.json"), ("calibration", "runs/calibration.json"), ("active_core", "runs/active_core.json")]:
            path = self.root/relative
            if path.exists():
                try:
                    reports[name] = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
        for name in ("core", "language"):
            path = self.root/("runs/tidal/training.jsonl" if name == "core" else "runs/language/metrics.jsonl")
            if path.exists():
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                    reports[name+"_recent"] = [json.loads(line) for line in lines[-12:]]
                except (ValueError, OSError):
                    pass
        from .core import active_core_path
        from .provenance import source_status
        core_error = None
        try:
            core_ready = active_core_path(self.root).is_file()
        except (ValueError, TypeError, OSError) as error:
            core_ready, core_error = False, str(error)
        result = {"version": __version__, "name": "Supermix Poseidon", "language_ready": (self.root/"models/language/model.safetensors").exists(), "core_ready": core_ready, "atlas": self.atlas_status(), "contrast": self.contrast_status(), "horizon": self.horizon_status(), "odyssey": self.odyssey_status(), "helm": self.helm_status(), "odysseus": self.odysseus_status(), "mco": self.mco_status(), "aura": self.aura_status(), "tessera": self.tessera_status(), "mnemorph": self.mnemorph_status(), "reports": reports, "limits": "Experimental composite system. Controlled geometric media. Tool-assisted maths. Synthetic survival."} | source_status()
        if core_error:
            result["core_error"] = core_error
        return result

    def _memory(self):
        from .memory import MemoryBank
        return MemoryBank.load(self.memory_path) if self.memory_path.exists() else MemoryBank()

    def remember(self, text, carrier="episodic"):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError("Memory must contain 1–2000 characters.")
        with self.lock:
            memory = self._memory()
            entry = memory.remember(text.strip(), carrier=carrier, metadata={"source": "explicit-user-entry"})
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory.save(self.memory_path)
        return {"saved": True, "entry": entry}

    def respond(self, prompt, mode="chat", history=None, seed=42, disabled_carriers=None, planner="policy", scarcity=1.0, max_steps=256):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 8000:
            raise ValueError("Prompt must contain 1–8000 characters.")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seed must be a nonnegative integer below 2^63.")
        with self.lock:
            if mode == "math" or (mode == "chat" and extract_math(prompt)):
                return solve(extract_math(prompt) or prompt)
            if mode in ("image", "video", "mesh"):
                from .media import render_scene
                core = self.core()
                planned = core.scene(prompt)
                result = render_scene(planned, self.root/"outputs/media", mode, seed)
                result.update({"text": f"Created {mode} from the learned scene prediction: {planned['count']} {planned['scale']} {planned['color']} {planned['shape']}, {planned['motion']} motion.", "backend": "Tidal scene predictor + procedural renderer", "prediction": planned, "verified": False})
                return result
            if mode == "world":
                from .world import _integer, _number
                from .world_controls import rollout_with_decisions
                _integer(max_steps, "max_steps", 1, 10000)
                scarcity = _number(scarcity, "scarcity", .5, 4)
                if planner not in ("policy", "mpc", "hybrid", "risk_aware", "uncertainty", "atlas", "contrast", "horizon", "odyssey", "helm", "odysseus", "aura"):
                    raise ValueError("Unknown survival planner.")
                core = self.core()
                if planner == "aura":
                    controller = self.aura()
                    controller.reset(scarcity)
                    backend_desc = "AURA biomimetic Central Complex ring attractor + sparse neuropil arbiter in TidePool"
                elif planner == "odysseus":
                    controller = self.odysseus()
                    controller.reset(scarcity)
                    backend_desc = "Odysseus empirical cognitive map + Bayesian replenishment in TidePool"
                elif planner == "helm":
                    controller = self.helm()
                    controller.reset(scarcity=scarcity, max_steps=max_steps)
                    backend_desc = "Helm independently fitted multi-horizon critic with observed history and empirical abstention in TidePool"
                elif planner == "odyssey":
                    controller = self.odyssey()
                    controller.reset(scarcity)
                    backend_desc = "Odyssey cognitive topological mapping + navigational memory in TidePool"
                elif planner == "horizon":
                    controller = self.horizon()
                    backend_desc = "Horizon Atlas multi-horizon advantage + depletion-aware empirical gate in TidePool"
                elif planner == "contrast":
                    controller = self.contrast()
                    backend_desc = "Contrast Atlas matched-action advantage memory + empirical gate in TidePool"
                elif planner == "atlas":
                    controller = self.atlas()
                    backend_desc = "Counterfactual Atlas residual memory + measured-error gate in TidePool"
                elif planner == "mpc":
                    from .planning import ModelPredictivePlanner, PlanningConfig
                    controller = ModelPredictivePlanner(core, PlanningConfig(policy_weight=0.0, mpc_weight=1.0))
                    backend_desc = "Tidal dynamics head pure MPC planner in TidePool"
                elif planner == "hybrid":
                    from .planning import ModelPredictivePlanner
                    controller = ModelPredictivePlanner(core)
                    backend_desc = "Tidal policy + dynamics head hybrid planner in TidePool"
                elif planner in ("risk_aware", "uncertainty"):
                    from .ensemble import UncertaintyAwarePlanner
                    controller = UncertaintyAwarePlanner(core)
                    backend_desc = "Supermix Beyond uncertainty-aware 3-model ensemble planner in TidePool"
                else:
                    controller = core
                    backend_desc = "learned Tidal policy in TidePool"
                episode = rollout_with_decisions(controller, seed=seed, max_steps=max_steps, scarcity=scarcity)
                episode["controller"] = planner
                episode["backend"] = backend_desc
                return {"text": f"Episode ended after {episode['steps']} steps. " + (f"The agent reached the {max_steps}-step horizon alive." if episode["survived"] else f"The agent died: {episode['death_reason']}."), "backend": backend_desc, "episode": episode, "verified": True}
            if mode == "memory":
                retrieved = self._memory().retrieve(prompt, disabled_carriers=disabled_carriers or [])
                return {"text": json.dumps(retrieved, ensure_ascii=False, indent=2), "backend": "explicit carrier memory retrieval", "memories": retrieved, "verified": False}
            if mode != "chat":
                raise ValueError("Unknown mode.")
            memories = self._memory().retrieve(prompt, disabled_carriers=disabled_carriers or []) if self.memory_path.exists() else []
            result = self.language.generate(prompt, history, memories)
            result["memories"] = memories
            return result
