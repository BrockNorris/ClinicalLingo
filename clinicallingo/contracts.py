"""Stage contracts: the typed objects that cross every boundary in the pipeline.

Boundary map (see docs/architecture.md):
    CLI            -> Validation      : RawInput          -> ValidatedRequest
    Validation     -> Retriever       : ValidatedRequest
    Retriever      -> Prompt Builder  : RetrievalResult
    Prompt Builder -> Model           : Prompt            -> str (raw model text)
    Model          -> Postprocess     : str               -> ExplainResult
    Postprocess    -> Feedback loop   : ExplainResult + RetrievalResult -> QuizRound
    every stage    -> Logging         : RunLog
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["word", "sentence", "page"]
Status = Literal["ok", "parse_failed", "model_timeout", "model_error", "not_found"]
Verdict = Literal["deeper", "clarify", "related"]


# --- Input / Validation -------------------------------------------------------

class RawInput(BaseModel):
    """What the CLI hands to Validation. Untrusted."""
    text: str
    mode: str


class ValidatedRequest(BaseModel):
    """Output of Validation. Guaranteed non-empty, printable, within the mode's length cap."""
    text: str = Field(min_length=1)
    mode: Mode
    char_count: int = Field(ge=1)


# --- Retrieval ------------------------------------------------------------------

class Chunk(BaseModel):
    """One retrieved dictionary entry or web-cache passage."""
    term: str
    text: str
    source: str
    score: float = Field(ge=-1.0, le=1.0)
    index_name: Literal["medical_dictionary", "general_dictionary", "web_cache"]
    synonyms: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Output of the Retriever. At most TOP_K chunks, all with score >= MIN_SCORE.
    `coverage` is the best medical-dictionary score (0.0 when nothing matched)."""
    chunks: list[Chunk] = Field(default_factory=list)
    coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    used_web: bool = False
    web_error: Optional[str] = None
    web_cache_hit: bool = False
    lexical_hits: list[str] = Field(default_factory=list)


# --- Prompt / Model ----------------------------------------------------------------

class Prompt(BaseModel):
    system: str
    user: str
    max_tokens: int = Field(gt=0)
    timeout_s: float = Field(gt=0)
    json_mode: bool = True


# --- Postprocessing ---------------------------------------------------------------

class DefinitionCard(BaseModel):
    term: str
    plain_definition: str
    clinical_definition: str
    synonyms: list[str] = Field(default_factory=list)
    used_sources: list[str] = Field(default_factory=list)


class KeyTerm(BaseModel):
    term: str
    plain_meaning: str


class Rewrite(BaseModel):
    original: str
    rewritten: str
    key_terms: list[KeyTerm] = Field(default_factory=list)


class ExplainResult(BaseModel):
    """Final typed output of one explain run."""
    mode: Mode
    status: Status
    card: Optional[DefinitionCard] = None
    rewrite: Optional[Rewrite] = None
    fallback_text: Optional[str] = None
    grounded: bool = False
    low_confidence: bool = True
    sources: list[str] = Field(default_factory=list)
    message: Optional[str] = None  # human-readable note (e.g. "term not found in dictionary")


# --- Feedback loop ----------------------------------------------------------------------

class QuizQuestion(BaseModel):
    question: str
    reference_answer: str
    based_on_term: str


class QuizRound(BaseModel):
    question: str
    reference_answer: str
    user_answer: str
    similarity: float = Field(ge=-1.0, le=1.0)
    verdict: Verdict
    follow_up: str


# --- Logging ------------------------------------------------------------------------------

class RunLog(BaseModel):
    run_id: str
    ts: str
    command: str
    mode: Optional[str] = None
    input_chars: int
    stages: list[str]
    latency_ms: dict[str, float]
    retrieval: dict = Field(default_factory=dict)
    model: dict = Field(default_factory=dict)
    status: str
    grounded: Optional[bool] = None
    low_confidence: Optional[bool] = None
    output: dict = Field(default_factory=dict)
    error: Optional[str] = None
    injected_failure: Optional[str] = None
