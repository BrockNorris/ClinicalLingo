"""Paths and settings. Everything resolves relative to the repo root so the
project runs unchanged from any drive letter or mount point.

Overrides (all optional, via environment variables):
    CLINICALLINGO_HOME        repo root (default: the folder containing this package)
    OLLAMA_HOST               Ollama server, default 127.0.0.1:11434
    CLINICALLINGO_GEN_MODEL   generation model tag, default phi3:mini
    CLINICALLINGO_EMBED_MODEL embedding model tag, default all-minilm
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("CLINICALLINGO_HOME") or Path(__file__).resolve().parents[1])
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
WEB_CACHE_DIR = DATA_DIR / "web_cache"
LOG_DIR = REPO_ROOT / "logs"
RUN_LOG_PATH = LOG_DIR / "runs.jsonl"

MEDICAL_DICT_PATH = DATA_DIR / "medical_dictionary.json"
GENERAL_DICT_PATH = DATA_DIR / "general_dictionary.json"


def ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/")


GEN_MODEL = os.environ.get("CLINICALLINGO_GEN_MODEL", "phi3:mini")
EMBED_MODEL = os.environ.get("CLINICALLINGO_EMBED_MODEL", "all-minilm")

# Input limits per mode (characters). These are the Validation contract.
INPUT_LIMITS = {"word": 64, "sentence": 512, "page": 4000}

# Retrieval contract
TOP_K_BY_MODE = {"word": 5, "sentence": 8, "page": 12}
MIN_SCORE = 0.35           # chunks below this cosine similarity are dropped
WEB_TRIGGER_SCORE = 0.55   # word mode: if best medical-dictionary score is below this, call the web tool
WEB_RELEVANCE_SCORE = 0.35 # web chunks below this similarity to the query are discarded (relevance double-check); must be >= MIN_SCORE

# Model contract
GEN_TIMEOUT_S = 120.0
GEN_MAX_TOKENS = 400
EMBED_TIMEOUT_S = 60.0

# Feedback-loop contract (cosine similarity between student answer and reference answer)
QUIZ_DEEPER_BELOW = 0.55
QUIZ_RELATED_ABOVE = 0.75

# Web tool: only these domains are ever fetched
ALLOWED_DOMAINS = (
    "medlineplus.gov",
    "webmd.com",
    "radiopaedia.org",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "mayoclinic.org",
    "clevelandclinic.org",
)
WEB_FETCH_TIMEOUT_S = 10.0
WEB_SEARCH_TIMEOUT_S = 10.0
WEB_TOOL_BUDGET_S = 30.0   # hard wall-clock cap for one tool call (search + fetches); exceeded -> error, no passages
WEB_MAX_RESULTS = 4
WEB_CHUNK_CHARS = 500
