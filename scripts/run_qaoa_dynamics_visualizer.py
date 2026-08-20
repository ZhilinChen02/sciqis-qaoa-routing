#!/usr/bin/env python3
"""Serve the read-only layer-by-layer QAOA Evolution Microscope."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
WEB = PROJECT_ROOT / "web" / "qaoa_dynamics_visualizer"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.qaoa_evolution_microscope import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    EvolutionMicroscopeDataError,
    QAOAEvolutionMicroscopeRepository,
)


ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/math.js": ("math.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
VENDOR_PREFIX = "/vendor/katex/"
VENDOR_ROOT = (WEB / "vendor" / "katex").resolve()
VENDOR_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


def build_handler(
    repository: object,
) -> type[BaseHTTPRequestHandler]:
    """Bind one validated, immutable repository to a standard-library server."""

    class DynamicsVisualizerHandler(BaseHTTPRequestHandler):
        server_version = "QAOAEvolutionMicroscope/2.0"

        def do_GET(self) -> None:  # noqa: N802 - standard-library API
            parsed = urlparse(self.path)
            if parsed.path == "/api/catalog":
                self._send_json(repository.catalog())
                return
            if parsed.path == "/api/validation":
                report = getattr(repository, "static_validation", None)
                if report is None:
                    report = repository.validation_report
                self._send_json(report)
                return
            if parsed.path == "/api/health":
                report = getattr(repository, "static_validation", None)
                if report is None:
                    report = repository.validation_report
                self._send_json(
                    {
                        "status": "ok",
                        "scientific_validation": report["all_checks_passed"],
                    }
                )
                return
            evolution_prefix = "/api/evolution/"
            if parsed.path.startswith(evolution_prefix):
                parts = [
                    unquote(part)
                    for part in parsed.path[len(evolution_prefix) :].split("/")
                    if part
                ]
                try:
                    if len(parts) == 3 and parts[2] == "summary":
                        self._send_json(
                            repository.evolution_summary(parts[0], int(parts[1]))
                        )
                        return
                    if len(parts) == 3 and parts[2] == "checkpoint":
                        query = parse_qs(parsed.query)
                        self._send_json(
                            repository.checkpoint(
                                parts[0],
                                int(parts[1]),
                                int(query.get("layer", [0])[0]),
                                query.get("stage", ["before_cost"])[0],
                                top_n=int(query.get("top", [20])[0]),
                                state_filter=query.get("filter", ["all"])[0],
                                query=query.get("q", [None])[0],
                                selected_state=(
                                    None
                                    if "selected" not in query
                                    else int(query["selected"][0])
                                ),
                            )
                        )
                        return
                    if len(parts) == 4 and parts[2] == "track":
                        self._send_json(
                            repository.track_state(parts[0], int(parts[1]), int(parts[3]))
                        )
                        return
                except (KeyError, ValueError) as error:
                    self._send_json(
                        {"error": "invalid_evolution_selector", "detail": str(error)},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_json(
                    {"error": "invalid_evolution_endpoint"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            prefix = "/api/run/"
            if parsed.path.startswith(prefix):
                parts = [unquote(part) for part in parsed.path[len(prefix) :].split("/")]
                if len(parts) != 2:
                    self._send_json(
                        {"error": "invalid_run_selector"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                algorithm, depth_text = parts
                try:
                    depth = int(depth_text)
                    run_config = getattr(repository, "run_config", None)
                    payload = (
                        run_config(algorithm, depth)
                        if run_config is not None
                        else repository.load_run(algorithm, depth)
                    )
                except (KeyError, ValueError):
                    self._send_json(
                        {
                            "error": "unknown_run",
                            "algorithm": algorithm,
                            "depth": depth_text,
                        },
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_json(payload)
                return
            asset = ASSETS.get(parsed.path)
            if asset is not None:
                filename, content_type = asset
                path = WEB / filename
            elif parsed.path.startswith(VENDOR_PREFIX):
                relative = unquote(parsed.path[len(VENDOR_PREFIX) :])
                path = (VENDOR_ROOT / relative).resolve()
                if not path.is_relative_to(VENDOR_ROOT) or not path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = VENDOR_CONTENT_TYPES.get(
                    path.suffix.lower(), "application/octet-stream"
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self, payload: object, *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            # Keep presentation terminals quiet while retaining explicit errors.
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(format, *args)

    return DynamicsVisualizerHandler


def export_payloads(
    repository: QAOAEvolutionMicroscopeRepository, output: Path
) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "catalog.json").write_text(
        json.dumps(repository.catalog(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "validation.json").write_text(
        json.dumps(repository.full_validation(110), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for algorithm in ("penalty_x", "global_grover"):
        for depth in (21, 110):
            (output / f"{algorithm}_p{depth}_config.json").write_text(
                json.dumps(
                    repository.run_config(algorithm, depth),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (output / f"{algorithm}_p{depth}_summary.json").write_text(
                json.dumps(
                    repository.evolution_summary(algorithm, depth),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--export-payloads", type=Path)
    arguments = parser.parse_args()
    try:
        repository = QAOAEvolutionMicroscopeRepository(arguments.result_root)
    except EvolutionMicroscopeDataError as exc:
        raise SystemExit(f"Visualization data validation failed: {exc}") from exc
    if arguments.check:
        print(json.dumps(repository.full_validation(110), indent=2))
        return
    if arguments.export_payloads is not None:
        export_payloads(repository, arguments.export_payloads)
        print(f"exported browser payloads to {arguments.export_payloads}")
        return
    server = ThreadingHTTPServer(
        (arguments.host, int(arguments.port)), build_handler(repository)
    )
    url = f"http://{arguments.host}:{arguments.port}/"
    print(
        f"QAOA Evolution Microscope ready at {url} "
        f"(artifact checks: {repository.static_validation['all_checks_passed']}; "
        "checkpoint states replay lazily from frozen parameters)"
    )
    if not arguments.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
