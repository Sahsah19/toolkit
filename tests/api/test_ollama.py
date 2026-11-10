"""query_ollama and its parsing, against a fake server."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from corallab.api.ollama import http_json, ollama_model_name, query_ollama


def _fetch(show: dict[str, Any] | None, ps: dict[str, Any] | None = None) -> Any:
    seen: list[tuple[str, Any]] = []

    def fetch(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any] | None:
        seen.append((url, body))
        if url.endswith("/api/show"):
            return show
        return ps

    fetch.seen = seen  # type: ignore[attr-defined]
    return fetch


def test_model_name_drops_the_litellm_prefix() -> None:
    assert ollama_model_name("ollama_chat/llama3.2") == "llama3.2"
    assert ollama_model_name("llama3.1:8b") == "llama3.1:8b"


def test_reads_modelfile_num_ctx_and_trained_maximum() -> None:
    fetch = _fetch(
        {
            "parameters": 'stop "<|eot_id|>"\nnum_ctx                        8192',
            "model_info": {"llama.context_length": 131072},
        }
    )
    info = query_ollama("http://host:11434/", "ollama_chat/llama3.2", fetch=fetch)
    assert info.reachable
    assert info.modelfile_num_ctx == 8192
    assert info.max_context_length == 131072
    assert fetch.seen[0] == ("http://host:11434/api/show", {"model": "llama3.2"})


def test_reads_the_loaded_context_from_ps() -> None:
    fetch = _fetch(
        {"parameters": "", "model_info": {}},
        {"models": [{"name": "llama3.2:latest", "context_length": 4096}, "junk"]},
    )
    info = query_ollama(None, "llama3.2", fetch=fetch)
    assert info.loaded_context_length == 4096
    assert info.modelfile_num_ctx is None
    assert info.max_context_length is None
    assert fetch.seen[0][0] == "http://localhost:11434/api/show"


def test_bad_num_ctx_is_ignored() -> None:
    info = query_ollama(None, "m", fetch=_fetch({"parameters": "num_ctx lots", "model_info": None}))
    assert info.modelfile_num_ctx is None


def test_unreachable_server() -> None:
    info = query_ollama(None, "llama3.2", fetch=_fetch(None))
    assert not info.reachable


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self._reply({"echo": body})

    def do_GET(self) -> None:
        self._reply(["not", "an", "object"])

    def _reply(self, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_http_json_posts_and_decodes(server: str) -> None:
    assert http_json(f"{server}/api/show", {"model": "x"}, 2.0) == {"echo": {"model": "x"}}


def test_http_json_rejects_a_non_object(server: str) -> None:
    assert http_json(f"{server}/api/ps", None, 2.0) is None


def test_http_json_returns_none_when_nothing_listens() -> None:
    assert http_json("http://127.0.0.1:9/api/show", {"model": "x"}, 0.5) is None
