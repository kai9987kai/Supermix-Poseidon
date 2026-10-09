import argparse
import json
from .runtime import Poseidon

def main():
    p = argparse.ArgumentParser(description="Supermix Poseidon & Beyond local research system")
    p.add_argument("command", choices=[
        "serve", "chat", "math", "image", "video", "mesh", "world", "status",
        "beyond-benchmark", "beyond-circuits", "beyond-memory", "beyond-moe", "beyond-scene",
        "atlas-fit", "experiment", "verify-experiment"
    ])
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--scarcity", type=float, default=2.5)
    p.add_argument("--adapter")
    p.add_argument("--planner", choices=["policy", "mpc", "hybrid", "risk_aware", "uncertainty", "atlas"], default="policy", help="world simulation planner")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--anchors", type=int, default=24)
    p.add_argument("--train-episodes", type=int, default=12)
    p.add_argument("--calibration-episodes", type=int, default=6)
    args = p.parse_args()
    args.seed = args.seed if args.seed is not None else (81000001 if args.command == "atlas-fit" else 93000001 if args.command == "experiment" else 42)
    args.episodes = args.episodes if args.episodes is not None else (4 if args.command == "experiment" else 100)

    if args.command == "verify-experiment":
        from pathlib import Path
        from .experiments import load_and_verify
        if not args.prompt:
            p.error("verify-experiment requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(load_and_verify(path), indent=2))
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

    if args.command == "experiment":
        result = Poseidon(args.root).experiment(args.seed, args.episodes, args.max_steps if args.max_steps is not None else 64, args.scarcity)
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
        result = runtime.respond(args.prompt or "Survive", args.command, seed=args.seed, planner=args.planner)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
