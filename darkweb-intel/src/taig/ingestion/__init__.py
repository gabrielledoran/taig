"""
taig.ingestion
--------------
Corpus ingestion layer for the Threat Actor Intelligence Graph.

Quick start
-----------
    from taig.ingestion import get_ingestor

    ingestor = get_ingestor("conti")          # or "babuk", "black_basta", "lockbit"
    result = ingestor.run()                   # writes data/processed/conti_intel.parquet

    # Run all enabled corpora at once
    from taig.ingestion import run_all_enabled
    results = run_all_enabled()

CLI
---
    python -m taig.ingestion --list
    python -m taig.ingestion --corpus conti --output data/processed/
    python -m taig.ingestion --corpus all --dry-run
"""

from taig.ingestion.registry import get_ingestor, list_corpora, run_all_enabled
from taig.ingestion.schemas import IntelMessage, IngestorResult, PARQUET_COLUMNS

__all__ = [
    "get_ingestor",
    "list_corpora",
    "run_all_enabled",
    "IntelMessage",
    "IngestorResult",
    "PARQUET_COLUMNS",
]
