import argparse
import json
from .runtime import Poseidon

def main():
    p = argparse.ArgumentParser(description="Supermix Poseidon & Beyond local research system")
    p.add_argument("command", choices=[
        "serve", "chat", "math", "image", "video", "mesh", "world", "status",
        "beyond-benchmark", "beyond-circuits", "beyond-memory", "beyond-moe", "beyond-scene",
        "atlas-fit", "experiment", "verify-experiment", "contrast-fit", "contrast-experiment", "verify-contrast",
        "horizon-fit", "horizon-experiment", "verify-horizon",
        "odyssey-fit", "odyssey-experiment", "verify-odyssey",
        "verify-evidence", "audit-trajectory", "verify-trajectory",
        "fit-helm", "helm-experiment", "verify-helm",
        "fit-odysseus", "odysseus-experiment", "verify-odysseus",
        "mco-experiment", "verify-mco",
        "aura-experiment", "verify-aura",
        "tessera-experiment", "verify-tessera",
        "mnemorph-experiment", "verify-mnemorph",
    ])
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--scarcity", type=float, default=None)
    p.add_argument("--adapter")
    p.add_argument("--planner", choices=["policy", "mpc", "hybrid", "risk_aware", "uncertainty", "atlas", "contrast", "horizon", "odyssey", "helm", "odysseus", "aura"], default="policy", help="world simulation planner")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--anchors", type=int, default=24)
    p.add_argument("--train-episodes", type=int, default=12)
    p.add_argument("--selection-episodes", type=int, default=8)
    p.add_argument("--calibration-episodes", type=int, default=None)
    p.add_argument("--parent", help="parent experiment receipt for verify-trajectory")
    p.add_argument("--stride", type=int, default=32, help="regular trajectory-audit anchor interval")
    p.add_argument("--horizon", type=int, default=16, help="trajectory-audit return horizon")
    p.add_argument("--max-anchors", type=int, default=256, help="trajectory-audit anchor budget")
    args = p.parse_args()
    default_seeds = {"atlas-fit": 81000001, "experiment": 93000001,
                     "contrast-fit": 101000001, "contrast-experiment": 104000001,
                     "horizon-fit": 107000001, "horizon-experiment": 108000001,
                     "odyssey-fit": 109000001, "odyssey-experiment": 110000001,
                     "fit-helm": 120000001, "helm-experiment": 123000001,
                     "fit-odysseus": 131000001, "odysseus-experiment": 133000001,
                     "mco-experiment": 140000001, "aura-experiment": 144000001,
                     "tessera-experiment": 155000001, "mnemorph-experiment": 160000001}
    args.seed = args.seed if args.seed is not None else default_seeds.get(args.command, 42)
    args.episodes = args.episodes if args.episodes is not None else (4 if args.command in ("experiment", "contrast-experiment", "horizon-experiment", "odyssey-experiment", "helm-experiment", "odysseus-experiment", "aura-experiment") else 100)
    args.scarcity = args.scarcity if args.scarcity is not None else (1.0 if args.command == "world" else 2.5)
    args.calibration_episodes = args.calibration_episodes if args.calibration_episodes is not None else (8 if args.command in ("contrast-fit", "horizon-fit", "odyssey-fit", "fit-helm", "fit-odysseus") else 6)

    if args.command == "verify-helm":
        from pathlib import Path
        from .helm import stable_json
        from .helm_experiment import verify_helm_receipt
        if not args.prompt:
            p.error("verify-helm requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(verify_helm_receipt(stable_json(path, max_bytes=128 * 1024 * 1024)), indent=2))
        return

    if args.command == "fit-helm":
        from .helm import HelmCritic
        from pathlib import Path
        from .provenance import assert_source_current
        import os
        import tempfile
        if any(not 2 <= value <= 64 for value in (args.train_episodes, args.selection_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("fit-helm requires 2–64 episodes per partition and disjoint seed namespaces")
        runtime = Poseidon(args.root)
        sources = assert_source_current()
        def progress(event):
            assert_source_current()
            if event.get("phase") and event.get("episode") is not None:
                import sys
                print(json.dumps(event), file=sys.stderr, flush=True)
        critic, receipt = HelmCritic.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            selection_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.selection_episodes)),
            calibration_seeds=list(range(args.seed + 2000000, args.seed + 2000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 96, progress=progress)
        assert_source_current()
        directory = runtime.root / "outputs/helm"
        critic.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps(receipt, indent=2))
        return

    if args.command in ("verify-evidence", "audit-trajectory", "verify-trajectory"):
        from pathlib import Path
        from .trajectory_audit import load_json, load_artifacts, verify_parent, collect, verify_audit, save_audit
        if not args.prompt:
            p.error(f"{args.command} requires an evidence path")
        root = Path(args.root)
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else root / path
        evidence = load_json(resolve(args.prompt))
        artifacts = load_artifacts(root)
        if args.command == "verify-evidence":
            result = verify_parent(evidence, artifacts)
        elif args.command == "verify-trajectory":
            if not args.parent:
                p.error("verify-trajectory requires --parent with its experiment receipt")
            result = verify_audit(evidence, load_json(resolve(args.parent)), artifacts)
        else:
            bundle = collect(Poseidon(root).core(), evidence, artifacts,
                             stride=args.stride, horizon=args.horizon, max_anchors=args.max_anchors)
            verification = verify_audit(bundle, evidence, artifacts)
            path = save_audit(bundle, root / "outputs/trajectory_audits")
            result = {"path": str(path), "experiment_id": bundle["experiment_id"],
                      "summary": bundle["summary"], "parent_verification": bundle["parent_verification"],
                      "verification": verification, "limits": bundle["limits"]}
        print(json.dumps(result, indent=2))
        return

    if args.command in ("verify-experiment", "verify-contrast", "verify-horizon", "verify-odyssey"):
        from pathlib import Path
        if args.command == "verify-odyssey":
            from .odyssey_experiments import load_and_verify
        elif args.command == "verify-horizon":
            from .horizon_experiments import load_and_verify
        elif args.command == "verify-contrast":
            from .contrast_experiments import load_and_verify
        else:
            from .experiments import load_and_verify
        if not args.prompt:
            p.error(f"{args.command} requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(load_and_verify(path), indent=2))
        return

    if args.command == "contrast-fit":
        from .contrast import ContrastAtlas
        from pathlib import Path
        import os
        import tempfile
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.selection_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("contrast-fit needs 1–64 episodes per partition and room for disjoint selection/calibration seeds")
        runtime = Poseidon(args.root)
        atlas, receipt = ContrastAtlas.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            selection_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.selection_episodes)),
            calibration_seeds=list(range(args.seed + 2000000, args.seed + 2000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128)
        directory = Path(args.root) / "outputs/contrast"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps({key: value for key, value in receipt.items() if key != "rows" and not key.endswith("_rows")}, indent=2))
        return

    if args.command == "atlas-fit":
        from .atlas import CounterfactualAtlas
        from pathlib import Path
        import os
        import tempfile
        if not 1 <= args.train_episodes <= 64 or not 1 <= args.calibration_episodes <= 64 or not 0 <= args.seed <= 2**63 - 1000100:
            p.error("atlas-fit needs 1–64 episodes per partition and a seed with room for a disjoint calibration namespace")
        runtime = Poseidon(args.root)
        atlas, receipt = CounterfactualAtlas.fit(runtime.core(), train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            calibration_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128)
        directory = runtime.root / "outputs/atlas"
        atlas.save(directory / "atlas.json")
        fd, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        os.replace(temporary, directory / "fit_receipt.json")
        print(json.dumps({key: value for key, value in receipt.items() if key != "calibration_rows"}, indent=2))
        return

    if args.command == "horizon-fit":
        from .horizon import HorizonAtlas
        from pathlib import Path
        import os
        import tempfile
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("horizon-fit needs 1–64 episodes per partition and room for disjoint calibration seeds")
        runtime = Poseidon(args.root)
        atlas, receipt = HorizonAtlas.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            calibration_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128)
        directory = Path(args.root) / "outputs/horizon"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps({key: value for key, value in receipt.items() if key != "calibration_anchor_rows"}, indent=2))
        return

    if args.command == "odyssey-fit":
        from .odyssey import OdysseyAtlas, OdysseyConfig
        from pathlib import Path
        import os
        import tempfile
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("odyssey-fit needs 1–64 episodes per partition and room for disjoint calibration seeds")
        runtime = Poseidon(args.root)
        config = OdysseyConfig(scarcity=args.scarcity)
        atlas, receipt = OdysseyAtlas.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            calibration_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128,
            config=config)
        directory = Path(args.root) / "outputs/odyssey"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps({key: value for key, value in receipt.items() if key != "calibration_anchor_rows"}, indent=2))
        return

    if args.command == "verify-mco":
        from pathlib import Path
        from .mco import verify_mco_receipt
        target = Path(args.prompt) if args.prompt else (Path(args.root) / "outputs/mco_experiments/RECEIPT.json")
        if not target.is_absolute():
            target = Path(args.root) / target
        data = json.loads(target.read_text(encoding="utf-8"))
        print(json.dumps(verify_mco_receipt(data), indent=2))
        return

    if args.command == "mco-experiment":
        runtime = Poseidon(args.root)
        result = runtime.mco_experiment()
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-odysseus":
        from pathlib import Path
        from .odysseus_experiment import verify_odysseus_receipt
        if not args.prompt:
            p.error("verify-odysseus requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_odysseus_receipt(data), indent=2))
        return

    if args.command == "fit-odysseus":
        from .odysseus import fit_odysseus
        from pathlib import Path
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("fit-odysseus needs 1–64 episodes per partition and room for disjoint calibration seeds")
        runtime = Poseidon(args.root)
        train_seeds = list(range(args.seed, args.seed + args.train_episodes))
        cal_seeds = list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes))
        atlas, receipt = fit_odysseus(
            runtime.core(),
            train_seeds=train_seeds,
            calibration_seeds=cal_seeds,
            max_steps=args.max_steps if args.max_steps is not None else 64,
            scarcity=args.scarcity,
        )
        directory = Path(args.root) / "outputs/odysseus"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        (directory / "fit_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return

    if args.command == "verify-aura":
        from pathlib import Path
        from .aura_experiment import verify_aura_receipt
        if not args.prompt:
            p.error("verify-aura requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_aura_receipt(data), indent=2))
        return

    if args.command == "tessera-experiment":
        runtime = Poseidon(args.root)
        result = runtime.tessera_experiment()
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-tessera":
        from pathlib import Path
        from .tessera_experiment import verify_tessera_receipt
        if not args.prompt:
            p.error("verify-tessera requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_tessera_receipt(data), indent=2))
        return

    if args.command == "mnemorph-experiment":
        runtime = Poseidon(args.root)
        result = runtime.mnemorph_experiment()
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-mnemorph":
        from pathlib import Path
        from .mnemorph import verify_mnemorph_receipt
        if not args.prompt:
            p.error("verify-mnemorph requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_mnemorph_receipt(data), indent=2))
        return

    if args.command in ("experiment", "contrast-experiment", "horizon-experiment", "odyssey-experiment", "helm-experiment", "odysseus-experiment", "aura-experiment"):
        runtime = Poseidon(args.root)
        if args.command == "aura-experiment":
            experiment = runtime.aura_experiment
        elif args.command == "odysseus-experiment":
            experiment = runtime.odysseus_experiment
        elif args.command == "helm-experiment":
            experiment = runtime.helm_experiment
        elif args.command == "odyssey-experiment":
            experiment = runtime.odyssey_experiment
        elif args.command == "horizon-experiment":
            experiment = runtime.horizon_experiment
        elif args.command == "contrast-experiment":
            experiment = runtime.contrast_experiment
        else:
            experiment = runtime.experiment
        result = experiment(args.seed, args.episodes, args.max_steps if args.max_steps is not None else 64, args.scarcity)
        print(json.dumps({key: value for key, value in result.items() if key != "replay"}, indent=2))
        return

    if args.command == "serve":
        from .server import serve
        serve(args.root, args.port, args.adapter)
        return

    if args.command == "beyond-benchmark":
        from .beyond_benchmark import run_paired_benchmark
        results = run_paired_benchmark(
            core_path="runs/tidal_dagger/core.pt",
            n_episodes=args.episodes,
            base_seed=args.seed,
            scarcity=args.scarcity
        )
        output = {name: res.__dict__ for name, res in results.items()}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-circuits":
        from .connectome import compare_circuit_topologies
        results = compare_circuit_topologies(seed=args.seed)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-memory":
        from .persistent_memory import run_delayed_recall_benchmark
        results = run_delayed_recall_benchmark(seed=args.seed, include_receipts=True)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-moe":
        from .sparse_moe import benchmark_sparse_vs_dense
        results = benchmark_sparse_vs_dense()
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-scene":
        from .scene_graph import create_procedural_beyond_world
        world = create_procedural_beyond_world()
        obj_file = world.export_wavefront_obj("outputs/media/beyond_world.obj")
        world.step_simulation(dt=0.2)
        world.apply_agent_impulse("agent", "explore")
        world.step_simulation(dt=0.2)
        print(json.dumps({
            "exported_obj": obj_file,
            "scene": world.to_dict()
        }, indent=2, ensure_ascii=False))
        return

    runtime = Poseidon(args.root, args.adapter)
    if args.command == "status":
        result = runtime.status()
    else:
        result = runtime.respond(args.prompt or "Survive", args.command, seed=args.seed, planner=args.planner,
                                 scarcity=args.scarcity, max_steps=args.max_steps if args.max_steps is not None else 256)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
