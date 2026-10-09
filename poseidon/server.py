"""Loopback-only workbench with bounded requests and explicit artifact paths."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse
import json
import mimetypes
import threading
from .runtime import Poseidon
from .provenance import StaleSourceError, assert_source_current, source_status

WEB = Path(__file__).parent/"web"

def make_handler(runtime, port):
    busy = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/status":
                try:
                    self.send_json(200, runtime.status())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error), "type": type(error).__name__})
                except Exception as error:
                    self.send_json(500, {"error": str(error), "type": type(error).__name__})
                return
            if path in ("/", "/index.html", "/style.css", "/app.js"):
                file = WEB/("index.html" if path in ("/", "/index.html") else path[1:])
            elif path.startswith("/artifacts/"):
                output_root = (runtime.root/"outputs").resolve()
                file = (output_root/path[len("/artifacts/"):]).resolve()
                if not file.is_relative_to(output_root) or file.suffix.lower() not in (".png", ".gif", ".mp4", ".obj", ".mtl", ".gltf", ".bin", ".json") or file.name == "memory.json":
                    self.send_json(403, {"error": "Artifact path is not public."}); return
            else:
                self.send_json(404, {"error": "Not found"}); return
            if not file.is_file():
                self.send_json(404, {"error": "Not found"}); return
            content = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if self.headers.get("Origin") not in (None, *allowed):
                self.send_json(403, {"error": "Cross-origin requests are disabled."}); return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_json(415, {"error": "Use application/json."}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be 1–65536 bytes.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
            except (ValueError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)}); return
            if self.path not in ("/api/respond", "/api/remember", "/api/experiment", "/api/contrast-experiment", "/api/horizon-experiment"):
                self.send_json(404, {"error": "Not found"}); return
            if not busy.acquire(blocking=False):
                self.send_json(409, {"error": "Poseidon is processing another request. Try again shortly."}); return
            try:
                if self.path in ("/api/experiment", "/api/contrast-experiment", "/api/horizon-experiment"):
                    if set(payload) - {"seed", "episodes", "max_steps", "scarcity"}:
                        raise ValueError("Unknown experiment setting.")
                    episodes, max_steps = payload.get("episodes", 4), payload.get("max_steps", 64)
                    if type(episodes) is not int or not 1 <= episodes <= 8 or type(max_steps) is not int or not 32 <= max_steps <= 256:
                        raise ValueError("Workbench experiments require 1–8 paired episodes and 32–256 steps.")
                    assert_source_current()
                    if self.path == "/api/horizon-experiment":
                        result = runtime.horizon_experiment(payload.get("seed", 108000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/contrast-experiment":
                        result = runtime.contrast_experiment(payload.get("seed", 104000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    else:
                        result = runtime.experiment(payload.get("seed", 93000001), episodes, max_steps, payload.get("scarcity", 2.5))
                elif self.path == "/api/remember":
                    result = runtime.remember(payload.get("text"), payload.get("carrier", "episodic"))
                else:
                    max_steps = payload.get("max_steps", 256)
                    if payload.get("mode", "chat") == "world" and (type(max_steps) is not int or not 1 <= max_steps <= 512):
                        raise ValueError("Workbench worlds require a 1–512 step horizon.")
                    result = runtime.respond(payload.get("prompt"), payload.get("mode", "chat"), payload.get("history"), payload.get("seed", 42), payload.get("disabled_carriers"), planner=payload.get("planner", "policy"), scarcity=payload.get("scarcity", 1.0), max_steps=max_steps)
                    if "paths" in result:
                        result["links"] = {key: "/artifacts/"+Path(value).resolve().relative_to(runtime.root/"outputs").as_posix() for key,value in result["paths"].items()}
                self.send_json(200, result)
            except StaleSourceError as error:
                self.send_json(409, {"error": str(error)} | source_status())
            except (ValueError, TypeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(500, {"error": str(error), "type": type(error).__name__})
            finally:
                busy.release()
    return Handler

def serve(root=".", port=8787, adapter=None):
    runtime = Poseidon(root, adapter)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(runtime, port))
    server.daemon_threads = True
    print(f"Supermix Poseidon: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--adapter")
    args = p.parse_args()
    serve(args.root, args.port, args.adapter)
