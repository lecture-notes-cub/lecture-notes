"""Ollama's local HTTP API; no cloud SDK or API key required.

API reference: https://docs.ollama.com/api/chat (Ollama: MIT license).
"""

import json
import os
import time
import urllib.error
import urllib.request


class ModelError(RuntimeError):
    pass


class Ollama:
    def __init__(self, model="qwen3:4b", url=None, timeout=180):
        self.model = model
        self.url = (url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout

    def complete(self, messages, schema, trace, step):
        if sum(len(m["content"]) for m in messages) > 48000:
            raise ModelError("Visible context exceeds the MVP budget. Use a shorter lecture interval.")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 16384, "num_predict": 3000},
            "keep_alive": "30m",
        }
        trace.emit("model_request", step=step, endpoint=self.url + "/api/chat", payload=payload)
        started = time.monotonic()
        try:
            req = urllib.request.Request(
                self.url + "/api/chat", json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
            trace.emit("model_response", step=step, latency_s=time.monotonic() - started, response=result)
            if result.get("done_reason") == "length":
                raise ModelError("Model response reached output limit; shorten the response.")
            return json.loads(result["message"]["content"])
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            trace.emit("model_error", step=step, latency_s=time.monotonic() - started, error=str(exc))
            raise ModelError(str(exc)) from exc
