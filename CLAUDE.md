# CLAUDE.md — handoff for the next Claude Code session on ClinicalLingo

Read this first. It is the project memory: the machine you are on may have no other record of the
decisions below (the previous sessions' memory files live on a different laptop).

## What this is

ClinicalLingo is Brock Norris's ECE 6540 (AI Architecture) course project: a textbook-term translator for
sonography students. Prototype 1 (Project Assignment 2) is complete and submitted-ready as of 2026-10-25.
Pipeline: Input → Validation → hybrid dictionary RAG (+ allow-listed web tool cached into a second index)
→ Phi-3-mini via local Ollama → typed postprocessing with grounding check → Output, plus a quiz feedback
loop scored by embedding similarity, and a JSONL run log per stage. Details: `docs/PLAN.md` (plan + build
log), `docs/architecture.md` (Mermaid diagram + contract table), `docs/writeup.md` / `.pdf` (the submission),
`docs/example_runs.md`, `README.md`.

## Hard constraints (from Brock — do not relax them)

1. **No hosted or paid model APIs, ever.** No Claude/Anthropic, OpenAI, or similar anywhere in the project.
   Generation = `phi3:mini` (Q4_0), embedding = `all-minilm` (all-MiniLM-L6-v2), both via a portable Ollama.
   Do not add the `anthropic` SDK or any API key to the repo.
2. **Nothing gets installed on any PC.** Everything lives on the flash drive next to this repo:
   ```
   <drive>\ClinicalLingo\          this repo
   <drive>\win\python\             Python 3.13 embeddable build + all requirements (Windows)
   <drive>\win\ollama\             Ollama zip release, no installer (Windows)
   <drive>\linux\                  relocatable CPython tarball, Ollama tar.zst, manylinux wheels, setup.sh (Linux; never yet executed on Linux)
   <drive>\models\                 OLLAMA_MODELS store (phi3:mini, all-minilm)
   <drive>\home\                   redirected HOME/USERPROFILE so Ollama's key stays on the drive
   <drive>\env.cmd/.sh, start-ollama.cmd/.sh, run.cmd/.sh, ollama.cmd   launchers (copies of scripts/)
   ```
   Temporary staging in the host's temp folder is fine if it is deleted afterwards (see `scripts/setup_drive.ps1`).
3. **The drive letter / mount point may change.** All launchers resolve paths relative to themselves; keep it that way.

## How to run and verify

- Windows: `start-ollama.cmd` (leave open), then from the drive root in PowerShell `.\run doctor`,
  `.\run explain "echogenic" --mode word`, `.\run explain "hypoechoic" --mode word --quiz`.
  (`cmd.exe` accepts plain `run`; PowerShell needs `.\`.)
- Linux (first time): `bash linux/setup.sh` (needs `zstd` on the host), then `bash start-ollama.sh`, `bash run.sh doctor`.
- Tests (offline, no Ollama needed): `<drive>\win\python\python.exe -m pytest` from the repo root — 59 tests.
- Example runs for the write-up: `python examples/run_examples.py` → `docs/example_runs.md`.
- Rebuild diagram PNG + write-up PDF: `python scripts/build_writeup.py` (headless Edge + the `markdown`
  package; install `markdown` with `pip install --target <temp dir> markdown` and pass `--mdlib <temp dir>`).
  Use `--skip-diagram` when only the text changed. The PDF must stay at 2–4 pages (currently 4, nearly full).
- Drive-root copies for submission: `..\ClinicalLingo-writeup.pdf` and `..\ClinicalLingo-architecture.png`.
  They are plain copies; re-copy after rebuilding.

## Gotchas learned the hard way

- The embeddable Python ignores `PYTHONPATH` and the cwd, so the entry point is `python cl.py …`, not
  `python -m clinicallingo`. pytest works because `pyproject.toml` sets `pythonpath = ["."]`.
- The flash drive is exFAT and slow (~20 MB/s read, ~3 MB/s write). Provision in a short local temp path and
  copy once; pip also fails on Windows paths over 260 chars. First model load from USB takes 1–3 minutes.
- Git on exFAT needs `git config --global --add safe.directory <path to repo>` on each new machine.
- `README.md` was once UTF-16; keep all text files UTF-8.
- Claude Code's Bash tool on this Windows setup mangles backslashes inside heredocs and turns robocopy's `/E`
  into a path. Write scripts to a file with the Write tool and run them, or use the PowerShell tool.
- Mermaid: `cdnjs` 404s for mermaid 11; use `https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js`.
  Subgraph titles wrap at a fixed width, so keep them short.
- Web tool: Radiopaedia returns 406 to user agents containing a URL but serves `/articles/<slug>` directly;
  `ddgs` backends that work are `duckduckgo` and `yahoo` (`auto` found nothing for rare terms). The whole tool
  call runs under a 30 s wall-clock budget after one run spent ~2 h in it (most likely the laptop slept).
- Phi-3-mini failure modes seen live: repetition loop (Ollama 500 "token repeat limit reached" → retry +
  `repeat_penalty 1.15`), prompt placeholder copied as a JSON key (→ tolerant `key_terms` parsing), typos.
- Quiz thresholds (0.40 deeper / 0.58 related) were calibrated with `examples/calibrate_quiz.py`; do not
  change them without re-running it.

## State and open work (as of 2026-10-25)

- GitHub `main` (https://github.com/BrockNorris/ClinicalLingo) and the drive repo are in sync. Brock edits
  `docs/writeup.md` on GitHub directly; after such edits pull, rebuild the PDF, push.
- Repo is private; Brock was deciding between adding the professor/TA as read collaborators or making it public.
- Brock chose to keep the failure-test section about the injected malformed-JSON case only (the quiz
  miscalibration is covered in the gap analysis, not as a second failure test).
- Not built (dashed in the diagram): photo input/OCR, mobile client, server API, rate limiting, content
  filter, illustrations, shared cache. Dictionary is a 248-entry seed written by the previous session, not a
  licensed source; every entry should be reviewed against course textbooks before students rely on it.
- Natural next steps when the course moves on: Ollama structured outputs (JSON schema) to replace the lenient
  parser; a semantic faithfulness check at Model → Postprocess; a real dictionary source; Linux smoke test;
  a `.gitattributes` with `* text=auto eol=lf` to stop the CRLF warnings.

## Working style Brock has asked for

- Say what you are about to do in a line, keep updates brief, finish with a recap that stands alone.
- Ask before anything hard to reverse; moving/deleting on the drive, pushing, and making the repo public all
  count. Commit only when the work is verified; he asks for pushes explicitly.
