#!/usr/bin/env python3
"""Serve the read-only interactive QAOA Dynamics Demo."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
WEB = PROJECT_ROOT / "web" / "qaoa_dynamics_visualizer"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qaoa_dynamics_visualization import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    DynamicsVisualizationDataError,
    QAOADynamicsVisualizationRepository,
)


ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def build_handler(
    repository: QAOADynamicsVisualizationRepository,
) -> type[BaseHTTPRequestHandler]:
    """Bind one validated, immutable repository to a standard-library server."""

    class DynamicsVisualizerHandler(BaseHTTPRequestHandler):
        server_version = "QAOADynamicsVisualizer/1.0"

        def do_GET(self) -> None:  # noqa: N802 - standard-library API
            parsed = urlparse(self.path)
            if parsed.path == "/api/catalog":
                self._send_json(repository.catalog())
                return
            if parsed.path == "/api/validation":
                self._send_json(repository.validation_report)
                return
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "scientific_validation": repository.validation_report[
                            "all_checks_passed"
                        ],
                    }
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
                    payload = repository.load_run(algorithm, depth)
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
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            filename, content_type = asset
            path = WEB / filename
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
    repository: QAOADynamicsVisualizationRepository, output: Path
) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "catalog.json").write_text(
        json.dumps(repository.catalog(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "validation.json").write_text(
        json.dumps(repository.validation_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for algorithm in ("penalty_x", "grover_global", "grover_feasible"):
        for depth in (1, 2):
            (output / f"{algorithm}_p{depth}.json").write_text(
                json.dumps(
                    repository.load_run(algorithm, depth),
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
        repository = QAOADynamicsVisualizationRepository(arguments.result_root)
    except DynamicsVisualizationDataError as exc:
        raise SystemExit(f"Visualization data validation failed: {exc}") from exc
    if arguments.check:
        print(json.dumps(repository.validation_report, indent=2))
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
        f"QAOA Dynamics Demo ready at {url} "
        f"(scientific checks: {repository.validation_report['all_checks_passed']})"
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
