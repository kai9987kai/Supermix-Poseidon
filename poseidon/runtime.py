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
        result = {"version": __version__, "name": "Supermix Poseidon", "language_ready": (self.root/"models/language/model.safetensors").exists(), "core_ready": core_ready, "atlas": self.atlas_status(), "contrast": self.contrast_status(), "horizon": self.horizon_status(), "reports": reports, "limits": "Experimental composite system. Controlled geometric media. Tool-assisted maths. Synthetic survival."} | source_status()
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
                if planner not in ("policy", "mpc", "hybrid", "risk_aware", "uncertainty", "atlas", "contrast", "horizon"):
                    raise ValueError("Unknown survival planner.")
                core = self.core()
                if planner == "horizon":
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
