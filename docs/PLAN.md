# ClinicalLingo Prototype 1 — Build Plan

Target: a small system that runs end to end (Input → Validation → Retrieval → Model Inference → Postprocessing → Output, with Logging alongside), satisfies every graded component of Project Assignment 2, and folds in the professor's feedback on the Part 1 proposal.

## 1. What the prototype will do

A command-line tool. The student types a word, sentence, or page of textbook text and picks a mode. The tool:

1. Validates the input (mode, length, non-empty, printable text).
2. Embeds it and retrieves matching entries from a **medical dictionary** (priority) and a **general dictionary**.
3. If dictionary coverage is weak, calls a **web search tool** restricted to trusted medical domains (MedlinePlus, WebMD, Radiopaedia, NCBI). Fetched pages are chunked, embedded, and stored in a **second RAG index (web cache)** so repeat lookups never hit the web again.
4. Prompts a small instruction-tuned LLM with the retrieved entries and generates a definition card (word mode) or a plain-language rewrite (sentence/page mode).
5. Postprocesses the raw model text into a typed result, runs a **grounding check** (did the answer actually use the retrieved definitions?), and attaches a confidence flag.
6. Runs an optional **feedback loop**: generates 1–2 comprehension questions from the retrieved content, scores the student's free-response answer with embedding cosine similarity, and either goes deeper (low score) or offers related terms (high score). This is the "simple agent loop" and directly implements the professor's first three feedback points.
7. Writes a JSONL log line per run: run id, input size, stages executed, per-stage latency, retrieval scores, model used, output.

Out of scope for prototype 1 (shown dashed on the diagram): photo input / on-device OCR, mobile app, server deployment, rate limiting, illustrations, caching across users.

## 2. Key decisions (made, with reasons)

| Decision | Choice | Why |
|---|---|---|
| Generation model | **Phi-3-mini (3.8B) via Ollama, Q4_K_M quantized** (`phi3:mini`) | Matches the proposal. Runs on this laptop's 8 GB RTX PRO 2000 GPU. Gives a real quantization observation for the write-up. |
| Second generation model (optional, for comparison) | A higher-precision Phi-3-mini tag (`phi3:3.8b-mini-4k-instruct-q8_0`, ~4 GB) behind the same `ModelBackend` interface, selected with `--model` | Gives the quantized vs. reduced-precision observation for section 4 without any hosted API. Fully local, free, lives on F:\. |
| No hosted APIs | **No Claude, Anthropic, OpenAI, or any paid/hosted model anywhere in the project** | User constraint. Everything runs locally from the flash drive. The `anthropic` package on this machine is not used and not listed in `requirements.txt`. |
| Embedding model | **all-MiniLM-L6-v2 via Ollama** (`all-minilm`) | Matches the proposal, no torch install (~2.5 GB avoided). One runtime for both models. |
| Vector store | **numpy cosine similarity, persisted to `.npy` + JSON** | Dictionary is a few hundred entries. No ChromaDB/FAISS dependency. Easy to explain and to log scores from. |
| Dictionary data | **Curated seed JSON** of ~150–250 sonography/anatomy/ultrasound-physics terms + ~50 general terms, each with `term`, `definition`, `synonyms`, `source` | A licensed medical dictionary is a later-semester concern. The seed is clearly labelled as a stub in the gap analysis. |
| Web search tool | `ddgs` (DuckDuckGo) with a **domain allowlist**, `requests` + `beautifulsoup4` for page text | No API key. Allowlist is the safety boundary. Failures degrade gracefully to "web unavailable", never crash. |
| Contracts | **Pydantic v2 models at every stage boundary** | Already installed. Gives the assignment's "stage interfaces" section for free and makes the failure test meaningful. |
| Interface | CLI via `argparse` (`clinicallingo explain ...`, `clinicallingo quiz ...`) | Assignment allows a CLI. No web framework in prototype 1. |
| Logging | Python `logging` to console + a **JSONL run log** (`logs/runs.jsonl`) | Structured log is what the write-up snippet and later observability modules need. |

## 2a. Portability: everything on the flash drive, nothing installed on any PC

Requirement: the finished prototype must run on a different Windows PC by plugging in the flash drive. So nothing may depend on an installer, a registry entry, a user profile, or an absolute path on C:\. Every component is a plain folder under `F:\ClinicalLingo`, and every script resolves paths relative to its own location, so the drive letter can change (E:, G:, ...) without edits.

```
F:\ClinicalLingo\                 (drive is exFAT: no 4 GB file limit, mounts on Windows/Linux/macOS)
├── ClinicalLingo\                # the git repo (moved here; remote stays github.com/BrockNorris/ClinicalLingo)
├── models\                       # OLLAMA_MODELS — shared by both OSes: phi3:mini (~2.3 GB), all-minilm (~50 MB), optional q8_0 (~4 GB)
├── home\                         # redirected HOME/USERPROFILE for Ollama so its identity key stays on the drive
├── win\
│   ├── python\                   # Python 3.13 embeddable build + pip + all requirements (~250 MB)
│   └── ollama\                   # Ollama Windows zip release, no installer (~1.5 GB)
├── linux\
│   ├── python\                   # python-build-standalone 3.13 x86_64 (relocatable CPython, ~60 MB)
│   ├── wheels\                   # manylinux wheels for requirements.txt, downloaded from Windows now (~40 MB)
│   ├── ollama\                   # Ollama linux-amd64 tgz release, extracted (~1.6 GB)
│   └── setup.sh                  # first run on Linux: installs wheels into linux\python offline, checks exec bits
├── env.cmd / env.sh              # sets PATH, OLLAMA_MODELS, HOME/USERPROFILE relative to the script's own folder
├── start-ollama.cmd / .sh        # runs `ollama serve` with that env (leave open in its own window)
└── run.cmd / run.sh              # `run explain "..." --mode word` → <os>/python -m clinicallingo
```

| Component | Why this form | Portability notes |
|---|---|---|
| **Python embeddable package** (`python-3.13.x-embed-amd64.zip`) instead of a venv | A venv hard-codes the path of the Python that created it and breaks on another PC. The embeddable build is self-contained. | Needs a one-line edit to `python313._pth` to enable `site-packages`, then `get-pip.py` once. All requirements (numpy, pydantic, requests, beautifulsoup4, ddgs) have Windows x64 wheels. Works on any 64-bit Windows 10/11. |
| **Ollama zip release** (`ollama-windows-amd64.zip` from GitHub releases) instead of `OllamaSetup.exe` | The installer writes to `%LOCALAPPDATA%`, the registry, and startup. The zip is just the binary plus GPU libraries. | Run via `start-ollama.cmd`. Uses NVIDIA GPU if present; falls back to CPU automatically on a PC without one (phi3:mini Q4 runs ~5–10 tokens/s on a modern CPU, so still usable). |
| **`OLLAMA_MODELS`** env var | Keeps weights on the drive. | Model blobs are content-addressed files; they copy/move freely with the folder. |
| **`USERPROFILE` redirected to `F:\ClinicalLingo\home`** inside the Ollama launcher only | Ollama writes `~/.ollama/id_ed25519` on first run. Redirecting the profile var for that one process keeps it on the drive. | Scoped to the launcher's process; does not affect anything else. |
| **Repo moved to the drive** | The launchers call `python -m clinicallingo` from a sibling folder. Keeping repo and runtime together means one folder to carry. | Git remote unchanged. The repo is still the source of truth; the drive adds the runtime next to it. The repo itself contains no installs, and `.gitignore` excludes logs and web cache. |

Linux is covered now rather than later because only the runtime folders differ: the model blobs, the repo, the index, and the logs are identical on both OSes. Linux specifics:

- There is no "embeddable" CPython for Linux; **python-build-standalone** (the relocatable CPython builds used by `uv`) is the equivalent and runs from any folder.
- Linux wheels can be downloaded from Windows today with `pip download --platform manylinux2014_x86_64 --only-binary=:all:`; `linux/setup.sh` installs them offline on first run, so the Linux PC needs no internet for setup.
- exFAT stores no POSIX execute bits. Most Linux automounters expose files as executable anyway; if not, `setup.sh` detects it and prints the one `mount -o remount,fmask=0022` or `bash run.sh` workaround.
- The Linux Ollama release bundles its own CUDA libraries; it needs only the NVIDIA driver on the host, or falls back to CPU.

Things that will still be true on the other PC and are documented in the README: 64-bit x86 only (no ARM); first model load from USB takes several seconds; USB 3.0 strongly recommended; web search features need internet, everything else runs offline.

Nothing from this project is installed to C:\ on this PC or any other.

## 3. Repository layout

```
ClinicalLingo/
├── README.md                     # how to install, run, reproduce examples
├── requirements.txt
├── scripts/
│   ├── setup_drive.ps1           # one-time: downloads Python embeddable + Ollama zip + pulls models into ../
│   ├── env.cmd                   # copied to the drive root; relative paths only
│   ├── start-ollama.cmd
│   └── run.cmd
├── .gitignore                    # logs/, data/web_cache/, __pycache__
├── clinicallingo/
│   ├── __init__.py
│   ├── cli.py                    # explain / quiz / build-index / examples / failure-test
│   ├── contracts.py              # Pydantic models for every boundary (see §5)
│   ├── validation.py             # Input → ValidatedRequest
│   ├── retrieval/
│   │   ├── embedder.py           # Ollama all-minilm; returns list[float]
│   │   ├── index.py              # VectorIndex: build / load / query(top_k, min_score)
│   │   └── build_index.py        # data/*.json → data/index/
│   ├── tools/
│   │   └── web_search.py         # allowlisted search + fetch + chunk + cache into web index
│   ├── inference/
│   │   ├── backend.py            # ModelBackend protocol: generate(prompt, timeout) -> str
│   │   └── ollama_backend.py
│   ├── prompts.py                # word / sentence / page / quiz / follow-up templates
│   ├── postprocess.py            # parse model JSON → DefinitionCard | Rewrite; grounding check
│   ├── feedback.py               # quiz generation, similarity scoring, adaptive follow-up
│   ├── pipeline.py               # orchestrates stages, times them, emits RunLog
│   └── runlog.py                 # JSONL writer + console logging setup
├── data/
│   ├── medical_dictionary.json
│   ├── general_dictionary.json
│   ├── index/                    # generated embeddings (committed: small, reproducible)
│   └── web_cache/                # generated at runtime (gitignored)
├── examples/
│   └── run_examples.py           # the three write-up runs, prints outputs + log lines
├── tests/
│   ├── test_validation.py
│   ├── test_retrieval.py
│   ├── test_postprocess.py
│   └── failure_test.py           # the intentional break for section 5
└── docs/
    ├── PLAN.md                   # this file
    ├── architecture.md           # Mermaid diagram (built vs dashed)
    └── writeup.md                # the 2–4 page submission (exported to PDF)
```

## 4. Pipeline stages and the diagram

```
[Input: CLI text + mode]
        │
        ▼
[Validation] ──invalid──▶ ValidationError (logged, exit code 2)
        │ ValidatedRequest
        ▼
[Retrieval: embed → medical dict (priority) → general dict → web cache]
        │ RetrievalResult (≤ k chunks, scores)
        ├── low coverage? ──▶ [Tool: web search (allowlist)] ──▶ [web cache index] ──┐
        │◀────────────────────────────────────────────────────────────────────────────┘
        ▼
[Prompt Builder] ──▶ [Model Inference: phi3:mini (Ollama, local)]
        │ raw text
        ▼
[Postprocessing: JSON parse → typed result → grounding check → confidence]
        │ ExplainResult
        ▼
[Output: definition card / rewrite]  ──optional──▶ [Feedback loop: quiz → score → deeper/related]
        
[Logging] runs alongside every stage: stage name, latency ms, sizes, scores, model, output.

Dashed (not built): photo input, on-device OCR, mobile client, server API, rate limiting, illustrations.
```

The Mermaid source goes in `docs/architecture.md`; it is rendered to PNG for the PDF.

## 5. Stage contracts (Pydantic)

These are the data contracts. Two of them are written up in detail for the assignment (marked ★).

| Boundary | Input | Output | Missing/invalid data |
|---|---|---|---|
| ★ CLI → Validation | `RawInput{text: str, mode: str}` | `ValidatedRequest{text, mode: Literal[word,sentence,page], char_count}` | Empty → reject. Length caps: word ≤ 64, sentence ≤ 512, page ≤ 4000 chars. Non-printable/control chars stripped; if nothing left → reject. Unknown mode → reject. All rejections raise `ValidationError`, are logged with reason, never reach the model. |
| ★ Retriever → Prompt Builder | `ValidatedRequest` | `RetrievalResult{chunks: list[Chunk{text, term, source, score, index_name}] (≤ k=5), coverage: float, used_web: bool}` | Chunks below `min_score=0.35` are dropped. If zero medical chunks and coverage < 0.5, the web tool is invoked once. If still empty, `chunks=[]` and the prompt tells the model to answer "not found in dictionary" and the result is flagged `low_confidence=True`. |
| Prompt Builder → Model | `Prompt{system, user, max_tokens, timeout_s}` | `str` (raw) | Timeout → `ModelError`, logged, run ends with `status=model_timeout`. Empty string → treated as parse failure below. |
| Model → Postprocess | raw `str` | `DefinitionCard{term, plain_definition, clinical_definition, synonyms, used_sources}` or `Rewrite{original, rewritten, key_terms[]}` | JSON parse fails → one retry with a stricter "JSON only" prompt; second failure → `status=parse_failed`, raw text returned in `fallback_text`, `low_confidence=True`. Grounding check: if none of the retrieved `term`s appear in the output, `grounded=False`. |
| Postprocess → Feedback loop | `ExplainResult + RetrievalResult` | `QuizRound{question, reference_answer, user_answer, similarity, verdict, follow_up}` | No chunks → quiz skipped (nothing to ground questions in). Similarity < 0.55 → "deeper" follow-up; ≥ 0.75 → "related terms"; between → "clarify". |

## 6. Logging schema (one JSON object per run, `logs/runs.jsonl`)

```json
{"run_id": "...", "ts": "...", "mode": "word", "input_chars": 11,
 "stages": ["validate","retrieve","web_search","infer","postprocess","output"],
 "latency_ms": {"validate": 1, "retrieve": 42, "web_search": 1830, "infer": 2410, "postprocess": 3, "total": 4290},
 "retrieval": {"k": 5, "top_score": 0.81, "coverage": 0.9, "used_web": true, "sources": ["medical_dictionary","medlineplus.gov"]},
 "model": {"backend": "ollama", "name": "phi3:mini", "quant": "Q4_K_M", "retries": 0},
 "status": "ok", "grounded": true, "low_confidence": false,
 "output": {...}}
```

## 7. Build order (each phase ends in a runnable state and a commit)

| Phase | Deliverable | Verify by |
|---|---|---|
| 0. Setup | Move repo to `F:\ClinicalLingo\ClinicalLingo`; `scripts/setup_drive.ps1` downloads the Windows Python embeddable build + Ollama zip into `win\`, the Linux python-build-standalone + Ollama tgz + manylinux wheels into `linux\`, bootstraps pip on Windows, installs `requirements.txt`, pulls `phi3:mini` and `all-minilm` into `models\`; `.cmd` and `.sh` launchers; skeleton; `.gitignore` | `start-ollama.cmd` then `run.cmd --help` works; `ollama run phi3:mini "hi"` answers; nothing new under `%LOCALAPPDATA%` or `%USERPROFILE%\.ollama` on C:\ |
| 1. Data + contracts + validation + retrieval | Seed dictionaries; `contracts.py`; `validation.py`; embedder; `VectorIndex`; `build-index` command | `tests/test_validation.py`, `tests/test_retrieval.py` pass; `clinicallingo retrieve "hepatic"` prints top-5 with scores |
| 2. Inference + postprocess + logging (**minimum submittable**) | Ollama backend; prompts; `postprocess.py`; `pipeline.py`; JSONL log | `clinicallingo explain "echogenic" --mode word` returns a definition card and one log line |
| 3. Web search tool + web cache RAG | `tools/web_search.py`; second index; trigger logic; cache hit on second call | Explain a term not in the seed dictionary; log shows `web_search` stage; second run shows cache hit, no `web_search` stage |
| 4. Feedback loop | `feedback.py`; `clinicallingo quiz` and `explain --quiz` | Answer a question well and poorly; see the two different follow-ups and the similarity score in the log |
| 5. Submission artifacts | `examples/run_examples.py` (3 runs); `tests/failure_test.py`; `docs/architecture.md`; `docs/writeup.md` → PDF; README | Fresh clone + README steps reproduce the three example runs |

Phases 1–2 are the priority. If anything slips, phases 3 and 4 are each independently droppable without breaking the end-to-end run, but both are planned because they are what the professor asked for.

## 8. Failure test (section 5 of the write-up)

Primary candidate: **malformed model output**. Run with `--inject-failure malformed_json` so the backend returns prose instead of JSON. Expected: the Model → Postprocess boundary catches it, retries once, then returns `status=parse_failed` with `fallback_text` rather than crashing. Write-up answers: what changed, crash vs. silent wrong answer, which boundary caught it, what to add (schema-constrained decoding / JSON mode).

Backup candidates, also wired to `--inject-failure`: `oversized_input` (caught at Validation), `empty_retrieval` (caught at Retriever → Prompt Builder, triggers web tool then "not found"), `model_timeout` (caught at Prompt Builder → Model).

## 9. Write-up mapping to grading rubric

| Rubric item | Source in repo |
|---|---|
| Architecture diagram (8) | `docs/architecture.md` rendered PNG, dashed boxes for OCR/mobile/server |
| Working prototype (20) | `clinicallingo explain`, RAG + tool call + agent loop all present; 3 runs from `examples/run_examples.py` with log snippets |
| Stage interfaces (8) | §5 rows ★, copied from `contracts.py` |
| Model choice and assumptions (8) | phi3:mini (Q4_0 as served by Ollama) on RTX PRO 2000 8 GB; observed tokens/s and failure modes from the run log; two model-card assumptions (English only; instruction-following on JSON is unreliable below ~4B) |
| Failure test (8) | `clinicallingo failure-test` (CLI) + §8 |
| Gap analysis (4) | Built: text pipeline, RAG, tool, loop, logging. Stubbed: dictionary coverage, grounding check (string match only). Missing: OCR, mobile, server, rate limiting. Most fragile: JSON parsing of a 3.8B model. Monitoring needs: per-stage latency percentiles, parse-failure rate, grounded rate, web-tool error rate, retrieval score drift. |
| Communication (4) | README + write-up |

## 10. Build log (what actually happened, 24-25 Oct 2026)

- Phases 0-5 are built. 59 offline unit tests pass (`python -m pytest`), and all live paths (word, sentence+quiz, web tool, failure injection) ran against the real model.
- The `phi3:mini` Ollama tag is **Q4_0**, not Q4_K_M as planned; the write-up reports the observed tag. The optional q8_0 comparison tag was skipped: the flash drive writes at ~3 MB/s, so 4 GB more would cost over 20 minutes of copying for a side observation.
- The flash drive (exFAT, USB) measured 20 MB/s read / 3 MB/s write. Downloads and provisioning therefore happen in a short-path staging folder on the local disk and are copied to the drive once (`scripts/setup_drive.ps1` encodes this). pip also fails on paths over 260 characters, which is the other reason for a short staging path.
- The Python embeddable build ignores `PYTHONPATH` and the working directory, so the CLI entry is `cl.py` at the repo root rather than `python -m clinicallingo`; pytest gets `pythonpath = ["."]`.
- Web tool findings: Radiopaedia rejects user agents that contain a URL (HTTP 406) but serves `/articles/<slug>` directly, so a direct probe is step one; the `ddgs` "auto" backend found nothing for rare terms while "duckduckgo" and "yahoo" worked; a 30 s wall-clock budget wraps the whole tool call after one run spent ~2 hours inside it (most likely the laptop slept, but the pipeline must not depend on that).
- Model findings: one sentence rewrite hit Ollama's "token repeat limit reached" (HTTP 500); the pipeline now retries once and sets `repeat_penalty=1.15`. Phi-3-mini also produced one typo inside an otherwise correct rewrite ("gallblad0der"), a useful write-up example of why a grounding check is not a correctness check.
- Linux runtime is staged (relocatable CPython tarball, Ollama tar.zst, manylinux wheels, `linux/setup.sh`) and verified only as far as checksums and wheel completeness; it has not been executed on a Linux host.

## 11. Assumptions to confirm

- The Ollama zip release runs from a plain folder with `OLLAMA_MODELS` set (about 2.3 GB for phi3:mini Q4 + 50 MB for all-minilm; +4 GB if the q8_0 comparison tag is pulled). If the zip release turns out not to be portable enough, the fallback is `llama-cpp-python` with GGUF weights in `models\` and `fastembed` for MiniLM, both pip-only inside the embeddable Python.
- The other PC is 64-bit x86 Windows now and likely a Linux distro later. Both runtimes are placed on the drive in phase 0; the Linux side is verified as far as possible from Windows (archive integrity, wheel set completeness) and gets a real smoke test the first time the drive is plugged into a Linux machine.
- The repo is moved to the flash drive before phase 0 so the launchers and the runtime sit side by side.
- No hosted or paid model API is used at any point. Confirmed.
- The seed dictionary is authored by us with cited sources; no licensed dictionary content is required for prototype 1.
- Write-up is produced as Markdown and exported to PDF for Canvas.
