"""Stage: Model Inference.  Prompt -> str (raw model text).

Only local models are used. OllamaBackend talks to the Ollama server that the
drive launcher starts; FakeBackend / FaultyBackend exist for offline tests and
the failure-injection flag.
"""
from __future__ import annotations

import logging
from typing import Callable

import requests

from ..config import GEN_MODEL, ollama_base_url
from ..contracts import Prompt

log = logging.getLogger("clinicallingo.model")


class ModelError(RuntimeError):
    pass


class ModelTimeout(ModelError):
    pass


class ModelBackend:
    name: str = "base"

    def info(self) -> dict:
        return {"backend": self.name}

    def generate(self, prompt: Prompt) -> str:  # pragma: no cover
        raise NotImplementedError


class OllamaBackend(ModelBackend):
    def __init__(self, model: str = GEN_MODEL, base_url: str | None = None, temperature: float = 0.2):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url or ollama_base_url()
        self.temperature = temperature
        self._info: dict | None = None
        self.last_stats: dict = {}

    def info(self) -> dict:
        if self._info is None:
            d = {"backend": "ollama", "model": self.model}
            try:
                r = requests.post(f"{self.base_url}/api/show", json={"model": self.model}, timeout=10)
                r.raise_for_status()
                det = r.json().get("details", {})
                d.update({
                    "family": det.get("family"),
                    "parameter_size": det.get("parameter_size"),
                    "quantization": det.get("quantization_level"),
                })
            except requests.RequestException as e:
                d["info_error"] = str(e)
            self._info = d
        return dict(self._info)

    def generate(self, prompt: Prompt) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            # repeat_penalty: phi3:mini occasionally loops in JSON mode until Ollama aborts with
            # "token repeat limit reached" (observed on a sentence rewrite); a mild penalty reduces this.
            "options": {"temperature": self.temperature, "num_predict": prompt.max_tokens, "repeat_penalty": 1.15},
        }
        if prompt.json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=prompt.timeout_s)
        except requests.Timeout as e:
            raise ModelTimeout(f"{self.name} timed out after {prompt.timeout_s}s") from e
        except requests.RequestException as e:
            raise ModelError(f"{self.name} request failed: {e}") from e
        if r.status_code != 200:
            raise ModelError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        self.last_stats = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "eval_ms": round((data.get("eval_duration") or 0) / 1e6, 1),
            "load_ms": round((data.get("load_duration") or 0) / 1e6, 1),
            "total_ms": round((data.get("total_duration") or 0) / 1e6, 1),
        }
        return (data.get("message") or {}).get("content", "") or ""

    def is_available(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            if not any(n == self.model or n.split(":")[0] == self.model.split(":")[0] and n.startswith(self.model) for n in names):
                return False, f"model {self.model} not pulled; available: {names}"
            return True, "ok"
        except requests.RequestException as e:
            return False, f"Ollama not reachable at {self.base_url}: {e}"


class FakeBackend(ModelBackend):
    """Offline stand-in: returns a fixed string or whatever `responder(prompt)` gives."""

    name = "fake"

    def __init__(self, responder: Callable[[Prompt], str] | str = "{}"):
        self._responder = responder

    def generate(self, prompt: Prompt) -> str:
        if callable(self._responder):
            return self._responder(prompt)
        return self._responder


class FaultyBackend(ModelBackend):
    """Failure injection wrapper used by `--inject-failure`."""

    def __init__(self, inner: ModelBackend, failure: str):
        self.inner = inner
        self.failure = failure
        self.name = f"faulty({failure})<{inner.name}>"

    def info(self) -> dict:
        d = self.inner.info()
        d["injected_failure"] = self.failure
        return d

    def generate(self, prompt: Prompt) -> str:
        if self.failure == "malformed_json":
            # Simulates the model ignoring the JSON instruction and answering in prose.
            return ("Sure! Echogenic means that a structure reflects ultrasound waves strongly and "
                    "therefore appears bright on the image. Let me know if you want more detail.")
        if self.failure == "model_timeout":
            raise ModelTimeout(f"injected timeout after {prompt.timeout_s}s")
        if self.failure == "empty_output":
            return ""
        return self.inner.generate(prompt)
