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
            if path == "/api/jobs" or path.startswith("/api/jobs/"):
                try:
                    if path == "/api/jobs":
                        result = {"jobs": runtime.jobs().list()}
                    elif path.count("/") == 3:
                        result = runtime.jobs().get(path.rsplit("/", 1)[1])
                    else:
                        raise KeyError(path)
                    self.send_json(200, result)
                except KeyError:
                    self.send_json(404, {"error": "Unknown experiment job."})
                except StaleSourceError as error:
                    self.send_json(409, {"error": str(error)} | source_status())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                return
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
            path = unquote(urlsplit(self.path).path)
            if path == "/api/jobs" or (path.startswith("/api/jobs/") and path.endswith("/cancel") and path.count("/") == 4):
                try:
                    if path == "/api/jobs":
                        assert_source_current()
                        kind = payload.get("kind")
                        if kind == "chimera-experiment":
                            self.send_json(202, runtime.submit_chimera_experiment(payload))
                        elif kind == "metamorph-experiment":
                            self.send_json(202, runtime.submit_metamorph_experiment(payload))
                        elif kind == "aura-experiment":
                            self.send_json(202, runtime.submit_aura_experiment(payload))
                        elif kind == "tessera-experiment":
                            self.send_json(202, runtime.submit_tessera_experiment(payload))
                        elif kind == "mnemorph-experiment":
                            self.send_json(202, runtime.submit_mnemorph_experiment(payload))
                        elif kind == "odysseus-experiment":
                            self.send_json(202, runtime.submit_odysseus_experiment(payload))
                        elif kind == "mco-experiment":
                            self.send_json(202, runtime.submit_mco_experiment(payload))
                        elif kind == "helm-experiment":
                            self.send_json(202, runtime.submit_helm_experiment(payload))
                        else:
                            raise ValueError(f"Unknown experiment job kind: {kind}")
                    else:
                        if payload:
                            raise ValueError("Cancellation requires an empty object.")
                        self.send_json(200, runtime.jobs().cancel(path.split("/")[3]))
                except KeyError:
                    self.send_json(404, {"error": "Unknown experiment job."})
                except StaleSourceError as error:
                    self.send_json(409, {"error": str(error)} | source_status())
                except (ValueError, TypeError) as error:
                    self.send_json(400, {"error": str(error)})
                except RuntimeError as error:
                    self.send_json(409, {"error": str(error)})
                return
            if self.path not in ("/api/respond", "/api/remember", "/api/experiment", "/api/contrast-experiment", "/api/horizon-experiment", "/api/odyssey-experiment", "/api/helm-experiment", "/api/odysseus-experiment", "/api/mco-experiment", "/api/aura-experiment", "/api/tessera-experiment", "/api/mnemorph-experiment", "/api/metamorph-experiment", "/api/chimera-experiment"):
                self.send_json(404, {"error": "Not found"}); return
            if not busy.acquire(blocking=False):
                self.send_json(409, {"error": "Poseidon is processing another request. Try again shortly."}); return
            try:
                if self.path == "/api/mco-experiment":
                    assert_source_current()
                    result = runtime.mco_experiment()
                elif self.path == "/api/tessera-experiment":
                    assert_source_current()
                    result = runtime.tessera_experiment()
                elif self.path == "/api/mnemorph-experiment":
                    assert_source_current()
                    result = runtime.mnemorph_experiment()
                elif self.path in ("/api/experiment", "/api/contrast-experiment", "/api/horizon-experiment", "/api/odyssey-experiment", "/api/helm-experiment", "/api/odysseus-experiment", "/api/aura-experiment", "/api/metamorph-experiment", "/api/chimera-experiment"):
                    if set(payload) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
                        raise ValueError("Unknown experiment setting.")
                    episodes, max_steps = payload.get("episodes", 4), payload.get("max_steps", 64)
                    if type(episodes) is not int or not 1 <= episodes <= 8 or type(max_steps) is not int or not 32 <= max_steps <= 256:
                        raise ValueError("Workbench experiments require 1–8 paired episodes and 32–256 steps.")
                    assert_source_current()
                    if self.path == "/api/chimera-experiment":
                        result = runtime.chimera_experiment(payload.get("seed", 202000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/metamorph-experiment":
                        result = runtime.metamorph_experiment(payload.get("seed", 199000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/aura-experiment":
                        result = runtime.aura_experiment(payload.get("seed", 144000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/odysseus-experiment":
                        result = runtime.odysseus_experiment(payload.get("seed", 133000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/helm-experiment":
                        result = runtime.helm_experiment(payload.get("seed", 112000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/odyssey-experiment":
                        result = runtime.odyssey_experiment(payload.get("seed", 110000001), episodes, max_steps, payload.get("scarcity", 2.5))
                    elif self.path == "/api/horizon-experiment":
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
        while True:
            try:
                server.serve_forever()
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                print(f"Server error recovered: {e}", flush=True)
    finally:
        server.server_close()
        runtime.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--adapter")
    args = p.parse_args()
    serve(args.root, args.port, args.adapter)
