# ClinicalLingo — Prototype 1 Architecture

Solid boxes are implemented in this prototype. Dashed boxes are parts of the Part 1 design that are not built yet. Thresholds and limits are in `clinicallingo/config.py`; the contracts are in `clinicallingo/contracts.py`.

```mermaid
flowchart TB
    classDef built fill:#e8f4ff,stroke:#2b6cb0,stroke-width:1.5px,color:#000
    classDef notbuilt fill:#fff,stroke:#999,stroke-width:1.5px,stroke-dasharray:6 4,color:#555
    classDef log fill:#fff8e1,stroke:#b7791f,stroke-width:1.5px,color:#000
    classDef store fill:#eef7ee,stroke:#2f855a,stroke-width:1.5px,color:#000
    classDef lane fill:#fafafa,stroke:#bbb,stroke-width:1px,color:#333

    subgraph L1["Stages 1-3"]
        direction LR
        CLI["Input<br/>CLI text + mode<br/>word | sentence | page"]:::built
        PHOTO["Input<br/>photo + on-device OCR"]:::notbuilt
        VAL["Validation<br/>mode, control chars,<br/>non-empty, length<br/>64 / 512 / 4000"]:::built
        MED[("medical_dictionary<br/>248 entries, priority")]:::store
        GEN[("general_dictionary<br/>54 entries")]:::store
        WEBC[("web_cache<br/>grown at runtime")]:::store
        RET["Retrieval<br/>MiniLM embedding<br/>lexical + cosine top-k<br/>score >= 0.35"]:::built
        WEB["Tool: web search<br/>Radiopaedia probe +<br/>DuckDuckGo/Yahoo<br/>allow-list, 30 s budget<br/>relevance >= 0.35"]:::built
        LOG["Logging: logs/runs.jsonl<br/>run_id, input size, stages,<br/>latency per stage, retrieval<br/>scores, model, status, output"]:::log
        CLI --> VAL
        PHOTO -.-> VAL
        VAL -- "ValidatedRequest" --> RET
        MED --> RET
        GEN --> RET
        WEBC --> RET
        RET -- "coverage < 0.55<br/>(word mode)" --> WEB
        WEB -- "relevant passages" --> WEBC
        VAL -. "every stage writes<br/>a timed entry" .-> LOG
    end

    subgraph L2["Stages 4-6"]
        direction LR
        PB["Prompt Builder<br/>numbered entries<br/>+ JSON schema"]:::built
        LLM["Model Inference<br/>Phi-3-mini 3.8B Q4_0<br/>Ollama, local<br/>JSON mode, 120 s"]:::built
        POST["Postprocessing<br/>JSON -> Pydantic<br/>Card | Rewrite<br/>grounding, confidence<br/>1 retry"]:::built
        OUT["Output<br/>definition card /<br/>plain rewrite<br/>+ sources"]:::built
        LOOP["Feedback loop<br/>quiz from entries, cosine similarity<br/>deeper (model re-explains) /<br/>clarify / related terms"]:::built
        RV["Response validation<br/>content filter,<br/>rate limit"]:::notbuilt
        APP["Mobile client /<br/>server API"]:::notbuilt
        PB -- "Prompt" --> LLM
        LLM -- "raw str" --> POST
        POST -- "ExplainResult" --> OUT
        OUT -- "--quiz" --> LOOP
        LOOP -.-> LLM
        OUT -.-> RV -.-> APP
    end

    L1 == "RetrievalResult (<= k chunks)" ==> L2
    class L1,L2 lane
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
