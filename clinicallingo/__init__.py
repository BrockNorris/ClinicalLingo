"""ClinicalLingo: a textbook-term translator for sonography students.

Prototype 1 pipeline:
    Input -> Validation -> Retrieval (dictionary RAG + web tool) -> Model Inference
          -> Postprocessing -> Output, with a JSONL run log alongside every stage.
"""

__version__ = "0.1.0"
