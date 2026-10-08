import argparse
import json
from .runtime import Poseidon

def main():
    p = argparse.ArgumentParser(description="Supermix Poseidon & Beyond local research system")
    p.add_argument("command", choices=[
        "serve", "chat", "math", "image", "video", "mesh", "world", "status",
        "beyond-benchmark", "beyond-circuits", "beyond-memory", "beyond-moe", "beyond-scene"
    ])
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--scarcity", type=float, default=2.5)
    p.add_argument("--adapter")
    p.add_argument("--planner", choices=["policy", "mpc", "hybrid", "risk_aware", "uncertainty"], default="policy", help="world simulation planner")
    args = p.parse_args()

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
        results = run_delayed_recall_benchmark(seed=args.seed)
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
