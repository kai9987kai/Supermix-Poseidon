"""Composition boundary: every result declares its actual backend."""
from __future__ import annotations
from pathlib import Path
import json
import threading
from .reasoning import extract_math, solve
from .language import LanguageRuntime

class Poseidon:
    def __init__(self, root=".", adapter=None):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.language = LanguageRuntime(self.root/"models/language", adapter=adapter)
        self._core = None
        self._core_stamp = None
        self.memory_path = self.root/"outputs/memory.json"

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
        return {"version": "0.1.0", "name": "Supermix Poseidon", "language_ready": (self.root/"models/language/model.safetensors").exists(), "core_ready": (self.root/"runs/tidal/core.pt").exists(), "reports": reports, "limits": "Experimental composite system. Controlled geometric media. Tool-assisted maths. Synthetic survival."}

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

    def respond(self, prompt, mode="chat", history=None, seed=42, disabled_carriers=None, planner="policy"):
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
                from .world import rollout
                core = self.core()
                if planner == "mpc":
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
                episode = rollout(controller, seed=seed, max_steps=256)
                return {"text": f"Episode ended after {episode['steps']} steps. " + ("The agent reached the 256-step horizon alive." if episode["survived"] else f"The agent died: {episode['death_reason']}."), "backend": backend_desc, "episode": episode, "verified": True}
            if mode == "memory":
                retrieved = self._memory().retrieve(prompt, disabled_carriers=disabled_carriers or [])
                return {"text": json.dumps(retrieved, ensure_ascii=False, indent=2), "backend": "explicit carrier memory retrieval", "memories": retrieved, "verified": False}
            if mode != "chat":
                raise ValueError("Unknown mode.")
            memories = self._memory().retrieve(prompt, disabled_carriers=disabled_carriers or []) if self.memory_path.exists() else []
            result = self.language.generate(prompt, history, memories)
            result["memories"] = memories
            return result
