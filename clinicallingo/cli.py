"""Command-line interface.

    clinicallingo explain "echogenic" --mode word
    clinicallingo explain "The liver is hypoechoic relative to the renal cortex." --mode sentence
    clinicallingo explain "<paragraph>" --mode page --quiz
    clinicallingo retrieve "hepatic" --mode word
    clinicallingo build-index
    clinicallingo doctor
    clinicallingo failure-test --failure malformed_json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__
from .config import (EMBED_MODEL, GEN_MODEL, INDEX_DIR, MEDICAL_DICT_PATH, RUN_LOG_PATH, WEB_CACHE_DIR,
                     ollama_base_url)
from .output import render, render_quiz
from .runlog import setup_logging

log = logging.getLogger("clinicallingo.cli")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--model", default=None, help=f"Ollama generation model tag (default {GEN_MODEL})")
    p.add_argument("--json", action="store_true", help="print the full result/log as JSON instead of text")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clinicallingo", description="Textbook term translator for sonography students")
    p.add_argument("--version", action="version", version=f"clinicallingo {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("explain", help="define a term or rewrite a sentence/page in plain language")
    e.add_argument("text")
    e.add_argument("--mode", choices=["word", "sentence", "page"], default="word")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--no-web", action="store_true", help="never call the web search tool")
    g.add_argument("--web", action="store_true", help="force the web search tool even if the dictionary matched")
    e.add_argument("--quiz", action="store_true", help="run the comprehension feedback loop afterwards")
    e.add_argument("--answer", action="append", default=None,
                   help="pre-supplied quiz answer (repeatable) for non-interactive runs")
    e.add_argument("--questions", type=int, default=2)
    e.add_argument("--inject-failure", choices=["oversized_input", "empty_retrieval", "malformed_json",
                                                "empty_output", "model_timeout"], default=None)
    _add_common(e)

    r = sub.add_parser("retrieve", help="show what the retriever returns, without calling the model")
    r.add_argument("text")
    r.add_argument("--mode", choices=["word", "sentence", "page"], default="word")
    r.add_argument("--no-web", action="store_true")
    _add_common(r)

    b = sub.add_parser("build-index", help="embed the dictionaries into data/index/")
    _add_common(b)

    d = sub.add_parser("doctor", help="check Ollama, models, and indexes")
    _add_common(d)

    f = sub.add_parser("failure-test", help="run the explain pipeline with an injected fault and report what caught it")
    f.add_argument("--failure", choices=["oversized_input", "empty_retrieval", "malformed_json",
                                         "empty_output", "model_timeout"], default="malformed_json")
    f.add_argument("--text", default="echogenic")
    f.add_argument("--mode", choices=["word", "sentence", "page"], default="word")
    _add_common(f)

    s = sub.add_parser("logs", help="print the last N run-log entries")
    s.add_argument("-n", type=int, default=5)
    _add_common(s)
    return p


def _backend(args):
    from .inference.backend import OllamaBackend
    return OllamaBackend(model=args.model) if args.model else None


def cmd_explain(args) -> int:
    from .pipeline import Pipeline
    pipe = Pipeline(backend=_backend(args), allow_web=not args.no_web, inject=args.inject_failure)
    result, retrieval, entry = pipe.explain(args.text, args.mode, force_web=args.web)
    if result is None:
        print(f"Input rejected: {entry.error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(entry.model_dump(exclude_none=True), indent=1))
    else:
        print(render(result, retrieval))
        print(f"\n[run {entry.run_id} | {entry.latency_ms.get('total', 0):.0f} ms | stages: {', '.join(entry.stages)}]")
    if args.quiz and result.status in ("ok", "not_found") and retrieval and retrieval.chunks:
        answers = list(args.answer or [])

        def ask(question: str) -> str:
            if answers:
                a = answers.pop(0)
                print(f"\nQ: {question}\nA (pre-supplied): {a}")
                return a
            print(f"\nQ: {question}")
            try:
                return input("Your answer: ")
            except EOFError:
                return ""

        rounds, qlog = pipe.quiz(result, retrieval, ask, n=args.questions)
        print()
        print(render_quiz(rounds) if rounds else "No quiz questions could be generated.")
        print(f"[quiz run {qlog.run_id} | {qlog.latency_ms.get('total', 0):.0f} ms]")
    return 0 if result.status == "ok" else 1


def cmd_retrieve(args) -> int:
    from .contracts import RawInput
    from .retrieval.build_index import load_all
    from .retrieval.embedder import OllamaEmbedder
    from .retrieval.index import VectorIndex
    from .retrieval.retriever import Retriever
    from .tools.web_search import WebSearchTool
    from .validation import InputValidationError, validate
    try:
        req = validate(RawInput(text=args.text, mode=args.mode))
    except InputValidationError as e:
        print(f"Input rejected: {e.reason}", file=sys.stderr)
        return 2
    med, gen = load_all(INDEX_DIR)
    ret = Retriever(OllamaEmbedder(), med, gen, VectorIndex.load(WEB_CACHE_DIR, "web_cache"),
                    web_tool=None if args.no_web else WebSearchTool())
    res = ret.retrieve(req, allow_web=not args.no_web)
    if args.json:
        print(res.model_dump_json(indent=1))
    else:
        print(f"coverage={res.coverage:.2f} used_web={res.used_web} cache_hit={res.web_cache_hit} lexical={res.lexical_hits}")
        for c in res.chunks:
            print(f"  {c.score:.3f} [{c.index_name}] {c.term}: {c.text[:110]}...")
        if res.web_error:
            print(f"  web: {res.web_error}")
    return 0


def cmd_build_index(args) -> int:
    from .retrieval.build_index import build_all
    from .retrieval.embedder import OllamaEmbedder
    counts = build_all(OllamaEmbedder())
    print(f"built indexes in {INDEX_DIR}: {counts}")
    return 0


def cmd_doctor(args) -> int:
    import requests
    from .inference.backend import OllamaBackend
    ok = True
    url = ollama_base_url()
    print(f"Ollama URL      : {url}")
    try:
        tags = requests.get(f"{url}/api/tags", timeout=5).json().get("models", [])
        names = [m["name"] for m in tags]
        print(f"Ollama reachable: yes ({len(names)} models)")
        for want in (args.model or GEN_MODEL, EMBED_MODEL):
            have = any(n == want or n.startswith(want + ":") or n.split(":")[0] == want for n in names)
            print(f"  model {want:<22}: {'ok' if have else 'MISSING -> ollama pull ' + want}")
            ok &= have
    except Exception as e:
        print(f"Ollama reachable: NO ({e}) -> run start-ollama.cmd / start-ollama.sh")
        ok = False
    print(f"Dictionary      : {'ok' if MEDICAL_DICT_PATH.exists() else 'MISSING'} ({MEDICAL_DICT_PATH})")
    built = (INDEX_DIR / "medical_dictionary" / "vectors.npy").exists()
    print(f"Index           : {'ok' if built else 'not built -> clinicallingo build-index'} ({INDEX_DIR})")
    ok &= built
    print(f"Web cache       : {'present' if (WEB_CACHE_DIR / 'entries.json').exists() else 'empty'} ({WEB_CACHE_DIR})")
    print(f"Run log         : {RUN_LOG_PATH}")
    if ok:
        info = OllamaBackend(model=args.model or GEN_MODEL).info()
        print(f"Model details   : {info}")
    print("STATUS          :", "READY" if ok else "NOT READY")
    return 0 if ok else 1


def cmd_failure_test(args) -> int:
    from .pipeline import Pipeline
    boundary = {
        "oversized_input": "CLI -> Validation (length cap)",
        "empty_retrieval": "Retriever -> Prompt Builder (empty chunk list is a legal value; prompt says 'no entries')",
        "malformed_json": "Model -> Postprocess (lenient JSON parse + schema validation, one retry)",
        "empty_output": "Model -> Postprocess (empty string treated as parse failure)",
        "model_timeout": "Prompt Builder -> Model (timeout_s in the Prompt contract)",
    }
    print(f"=== failure test: {args.failure} ===")
    pipe = Pipeline(backend=_backend(args), inject=args.failure, allow_web=False)
    result, retrieval, entry = pipe.explain(args.text, args.mode)
    print(f"crashed            : no (pipeline returned normally)")
    print(f"status             : {entry.status}")
    print(f"error              : {entry.error}")
    print(f"stages that ran    : {', '.join(entry.stages)}")
    print(f"boundary that caught it: {boundary[args.failure]}")
    if result is not None:
        print(f"grounded/low_conf  : {result.grounded} / {result.low_confidence}")
        print("--- rendered output ---")
        print(render(result, retrieval))
    print("--- log line ---")
    print(entry.model_dump_json(exclude_none=True)[:1200])
    return 0


def cmd_logs(args) -> int:
    from .runlog import read_runs
    for row in read_runs(RUN_LOG_PATH, last=args.n):
        if args.json:
            print(json.dumps(row))
        else:
            print(f"{row['ts']} {row['run_id']} {row['command']:<7} mode={row.get('mode')} status={row['status']} "
                  f"total={row['latency_ms'].get('total')}ms stages={','.join(row['stages'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    handlers = {"explain": cmd_explain, "retrieve": cmd_retrieve, "build-index": cmd_build_index,
                "doctor": cmd_doctor, "failure-test": cmd_failure_test, "logs": cmd_logs}
    try:
        return handlers[args.cmd](args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130
