import argparse
import json
from .runtime import Poseidon

def main():
    p = argparse.ArgumentParser(description="Supermix Poseidon local research system")
    p.add_argument("command", choices=["serve", "chat", "math", "image", "video", "mesh", "world", "status"])
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter")
    p.add_argument("--planner", choices=["policy", "mpc", "hybrid"], default="policy", help="world simulation planner")
    args = p.parse_args()
    if args.command == "serve":
        from .server import serve
        serve(args.root, args.port, args.adapter)
        return
    runtime = Poseidon(args.root, args.adapter)
    if args.command == "status":
        result = runtime.status()
    else:
        result = runtime.respond(args.prompt or "Survive", args.command, seed=args.seed, planner=args.planner)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
