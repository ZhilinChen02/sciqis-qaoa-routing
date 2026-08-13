#!/usr/bin/env python3
"""Serve the read-only interactive QAOA optimizer/circuit visualizer."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
import webbrowser


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
WEB = PROJECT / "web" / "qaoa_visualizer"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.qaoa_visualization import (  # noqa: E402
    QAOAVisualizationRepository,
    VisualizationDataError,
)
from support.sealed_results import Q2F_FINAL, verify_sealed_result  # noqa: E402


ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def build_handler(repository: QAOAVisualizationRepository) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP handler bound to one immutable repository instance."""

    class VisualizerHandler(BaseHTTPRequestHandler):
        server_version = "QAOAVisualizer/1.0"

        def do_GET(self) -> None:  # noqa: N802 - standard-library API
            parsed = urlparse(self.path)
            if parsed.path == "/api/runs":
                self._send_json(repository.catalog())
                return
            prefix = "/api/runs/"
            if parsed.path.startswith(prefix):
                run_id = unquote(parsed.path[len(prefix) :])
                try:
                    payload = repository.load_run(run_id)
                except KeyError:
                    self._send_json(
                        {"error": "unknown_run", "run_id": run_id},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                except VisualizationDataError as exc:
                    self._send_json(
                        {"error": "invalid_visualization_data", "detail": str(exc)},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
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
            self,
            payload: dict[str, object],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            if getattr(self.server, "quiet", False):
                return
            super().log_message(format, *args)

    return VisualizerHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay retained QAOA gamma, beta, and energy trajectories in a browser."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate inputs and print the catalog without starting a server",
    )
    arguments = parser.parse_args()

    verification = verify_sealed_result(Q2F_FINAL)
    repository = QAOAVisualizationRepository()
    if arguments.check:
        catalog = repository.catalog()
        print(
            f"visualizer inputs: {verification['status']} integrity, "
            f"{catalog['method_count']} methods, {catalog['run_count']} retained runs"
        )
        return
    if not 0 <= arguments.port <= 65535:
        parser.error("port must be between 0 and 65535")
    if not WEB.is_dir():
        raise FileNotFoundError(f"visualizer assets missing: {WEB}")

    server = ThreadingHTTPServer(
        (arguments.host, arguments.port), build_handler(repository)
    )
    server.quiet = bool(arguments.quiet)  # type: ignore[attr-defined]
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in ("0.0.0.0", "::") else actual_host
    url = f"http://{display_host}:{actual_port}/"
    print(f"QAOA visualizer: {url}")
    print("Press Ctrl+C to stop.")
    if not arguments.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
