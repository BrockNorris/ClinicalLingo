# ClinicalLingo — Prototype 1 Architecture

Solid boxes are implemented in this prototype. Dashed boxes are parts of the Part 1 design that are not built yet. Thresholds and limits are in `clinicallingo/config.py`; the contracts are in `clinicallingo/contracts.py`.

```mermaid
flowchart TB
    classDef built fill:#e8f4ff,stroke:#2b6cb0,stroke-width:1.5px,color:#000
    classDef notbuilt fill:#fff,stroke:#999,stroke-width:1.5px,stroke-dasharray:6 4,color:#555
    classDef log fill:#fff8e1,stroke:#b7791f,stroke-width:1.5px,color:#000
    classDef store fill:#eef7ee,stroke:#2f855a,stroke-width:1.5px,color:#000

    CLI["Input: CLI text + mode<br/>(word | sentence | page)"]:::built
    PHOTO["Input: photo + on-device OCR"]:::notbuilt
    VAL["Validation<br/>mode, control chars, non-empty<br/>length ≤ 64 / 512 / 4000"]:::built
    RET["Retrieval (RAG)<br/>all-MiniLM-L6-v2 embedding<br/>lexical match + cosine top-k ≥ 0.35"]:::built
    MED[("medical_dictionary<br/>248 entries (priority)")]:::store
    GEN[("general_dictionary<br/>54 entries")]:::store
    WEBC[("web_cache<br/>grown at runtime")]:::store
    WEB["Tool call: web search<br/>direct Radiopaedia probe + DuckDuckGo/Yahoo<br/>domain allow-list, 30 s budget<br/>relevance check ≥ 0.35"]:::built
    PB["Prompt Builder<br/>numbered entries + JSON schema"]:::built
    LLM["Model Inference<br/>Phi-3-mini 3.8B Q4_0 (Ollama, local)<br/>JSON mode, 120 s timeout"]:::built
    POST["Postprocessing<br/>lenient JSON → Pydantic (DefinitionCard | Rewrite)<br/>grounding check, confidence flag, 1 retry"]:::built
    OUT["Output<br/>definition card / plain rewrite + sources"]:::built
    LOOP["Feedback loop (simple agent loop)<br/>quiz from retrieved entries → cosine similarity<br/>< 0.55 deeper · 0.55–0.75 clarify · ≥ 0.75 related"]:::built
    RV["Response validation<br/>content filter, per-user rate limit"]:::notbuilt
    APP["Mobile client / server API<br/>illustrations, shared cache"]:::notbuilt
    LOG["Logging — logs/runs.jsonl<br/>run_id, input size, stages, latency per stage,<br/>retrieval scores, model info, status, output"]:::log

    CLI --> VAL
    PHOTO -.-> VAL
    VAL -- "ValidatedRequest" --> RET
    MED --> RET
    GEN --> RET
    WEBC --> RET
    RET -- "word mode, coverage < 0.55" --> WEB
    WEB -- "relevant passages" --> WEBC
    RET -- "RetrievalResult (≤ k chunks)" --> PB
    PB -- "Prompt" --> LLM
    LLM -- "raw str" --> POST
    POST -- "ExplainResult" --> OUT
    OUT -- "--quiz" --> LOOP
    LOOP -. "deeper explanation" .-> LLM
    OUT -.-> RV -.-> APP
    VAL -. "every stage writes a timed entry" .-> LOG
```

## Stage boundaries (data contracts)

All objects are Pydantic models in `clinicallingo/contracts.py`.

| Boundary | Input | Output | Missing / invalid data |
|---|---|---|---|
| CLI → Validation | `RawInput{text: str, mode: str}` | `ValidatedRequest{text, mode, char_count}` | Empty text, unknown mode, control-only text, over-limit length (64/512/4000 chars), or more than 3 words in word mode → `InputValidationError`; logged as `status=rejected_input`; nothing reaches retrieval or the model. |
| Validation → Retriever | `ValidatedRequest` | `RetrievalResult{chunks: list[Chunk] (≤ 5/8/12 by mode), coverage, used_web, web_cache_hit, web_error}` | Every chunk has `score ≥ 0.35`. Word mode with best medical score `< 0.55` and no cached passage → web tool called once (30 s hard budget); its passages must score `≥ 0.35` against the query or are dropped. Web failures never raise; recorded in `web_error`. Nothing found → `chunks=[]`. |
| Retriever → Prompt Builder | `RetrievalResult` | `Prompt{system, user, max_tokens, timeout_s, json_mode}` | Empty chunk list is legal: the prompt states "no dictionary entries were found" and asks for the `NOT_FOUND_IN_DICTIONARY` marker. |
| Prompt Builder → Model | `Prompt` | raw `str` | Timeout → `ModelTimeout` → `status=model_timeout`. HTTP/connection error (e.g. Ollama 500 "token repeat limit reached") → one retry, then `status=model_error`. Empty string → treated as parse failure downstream. |
| Model → Postprocess | raw `str` | `ExplainResult{status, card|rewrite, grounded, low_confidence, sources, fallback_text}` | Non-JSON or schema-invalid → one retry with a stricter instruction → `status=parse_failed` with `fallback_text`. Malformed `key_terms` items are dropped, not fatal. Empty retrieval or `NOT_FOUND` marker → `status=not_found`, citations discarded. Output that cites or mentions no retrieved term → `grounded=False, low_confidence=True`. |
| Postprocess → Feedback loop | `ExplainResult + RetrievalResult` | `list[QuizRound{question, reference_answer, user_answer, similarity, verdict, follow_up}]` | No chunks → no questions. Model fails to produce questions → deterministic fallback questions from the entries. Empty answer → similarity 0 → "deeper". Similarity is the max against the reference answer and the dictionary entry. |
