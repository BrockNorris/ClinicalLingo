"""Logging alongside every stage: per-stage timers + one JSON line per run.

logs/runs.jsonl is the observability surface for this prototype. Each line is a
RunLog (see contracts.py): input size, which stages ran, latency per stage,
retrieval scores, model info, status, and the output.
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .config import RUN_LOG_PATH
from .contracts import RunLog


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("urllib3", "httpx", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageTimer:
    """Records stage order and per-stage latency in milliseconds."""

    def __init__(self):
        self.stages: list[str] = []
        self.latency_ms: dict[str, float] = {}
        self._t0 = perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            self.latency_ms[name] = round((perf_counter() - start) * 1000, 1)
            self.stages.append(name)

    def add(self, name: str, ms: float) -> None:
        self.latency_ms[name] = round(ms, 1)
        if name not in self.stages:
            self.stages.append(name)

    def finish(self) -> dict[str, float]:
        self.latency_ms["total"] = round((perf_counter() - self._t0) * 1000, 1)
        return dict(self.latency_ms)


def write_run(entry: RunLog, path: Path = RUN_LOG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json(exclude_none=True) + "\n")
    logging.getLogger("clinicallingo.runlog").info(
        "run %s %s status=%s total=%.0fms stages=%s",
        entry.run_id, entry.command, entry.status, entry.latency_ms.get("total", 0), ",".join(entry.stages))


def read_runs(path: Path = RUN_LOG_PATH, last: int | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-last:] if last else rows
