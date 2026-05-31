"""
registry.py
-----------
Ingestor registry: maps corpus names to their ingestor classes and
provides factory functions used by the CLI and pipeline orchestrator.

Usage
-----
    from taig.ingestion.registry import get_ingestor, list_corpora

    ingestor = get_ingestor("conti")
    result = ingestor.run(output_dir=Path("data/processed"))

    # List all registered corpus names
    for name, enabled in list_corpora():
        print(name, "enabled" if enabled else "disabled")
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

from taig.ingestion.babuk import BabukIngestor
from taig.ingestion.base import BaseIngestor
from taig.ingestion.blackbasta import BlackBastaIngestor
from taig.ingestion.conti import ContiIngestor
from taig.ingestion.lockbit import LockBitIngestor
from taig.utils.config import CorpusConfig, DatasetsConfig, IngestionConfig, load_config
from taig.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry mapping: corpus name → ingestor class
# ---------------------------------------------------------------------------
# To add a new ingestor:
#  1. Create src/taig/ingestion/<name>.py with a class that extends BaseIngestor
#  2. Add an entry to config/datasets.yaml
#  3. Add the mapping below

_INGESTOR_REGISTRY: dict[str, Type[BaseIngestor]] = {
    "conti": ContiIngestor,
    "babuk": BabukIngestor,
    "black_basta": BlackBastaIngestor,
    "lockbit": LockBitIngestor,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def registered_names() -> list[str]:
    """Return all corpus names that have a registered ingestor class."""
    return list(_INGESTOR_REGISTRY)


def list_corpora(config_dir: Path | str | None = None) -> list[tuple[str, bool, str]]:
    """Return a list of (name, enabled, display_name) tuples.

    Combines the registry (which corpora have code) with the datasets config
    (which are enabled and what their human-readable names are).

    Parameters
    ----------
    config_dir:
        Override the config directory. Passed through to load_config().

    Returns
    -------
    List of (corpus_name, is_enabled, display_name) triples, sorted by name.
    """
    try:
        datasets: DatasetsConfig = load_config("datasets", config_dir=config_dir)
    except FileNotFoundError as exc:
        log.error("Could not load datasets config: %s", exc)
        return [(name, False, name) for name in sorted(registered_names())]

    result = []
    for name in sorted(_INGESTOR_REGISTRY):
        corpus = datasets.corpora.get(name)
        if corpus:
            result.append((name, corpus.enabled, corpus.display_name))
        else:
            result.append((name, False, name))
    return result


def get_ingestor(
    corpus_name: str,
    config_dir: Path | str | None = None,
) -> BaseIngestor:
    """Instantiate and return the ingestor for *corpus_name*.

    Parameters
    ----------
    corpus_name:
        Must be a key in the ingestor registry (see registered_names()).
    config_dir:
        Override the config directory. Useful in tests.

    Raises
    ------
    ValueError
        If *corpus_name* is not in the registry.
    KeyError
        If *corpus_name* is registered but missing from datasets.yaml.
    """
    if corpus_name not in _INGESTOR_REGISTRY:
        available = ", ".join(sorted(_INGESTOR_REGISTRY))
        raise ValueError(
            f"Unknown corpus: {corpus_name!r}. Available: {available}"
        )

    datasets: DatasetsConfig = load_config("datasets", config_dir=config_dir)
    pipeline_cfg = load_config("pipeline", config_dir=config_dir)

    if corpus_name not in datasets.corpora:
        raise KeyError(
            f"Corpus '{corpus_name}' is registered but not found in datasets.yaml. "
            "Add an entry for it."
        )

    corpus_cfg: CorpusConfig = datasets.corpora[corpus_name]
    ingestion_cfg: IngestionConfig = pipeline_cfg.ingestion
    ingestor_cls = _INGESTOR_REGISTRY[corpus_name]

    log.debug("Instantiating %s for corpus '%s'", ingestor_cls.__name__, corpus_name)
    return ingestor_cls(corpus_cfg, ingestion_cfg)


def run_all_enabled(
    output_dir: Path | None = None,
    config_dir: Path | str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run ingestion for all corpora that have `enabled: true` in datasets.yaml.

    Returns a dict mapping corpus_name → IngestorResult.
    """
    datasets: DatasetsConfig = load_config("datasets", config_dir=config_dir)
    enabled = [c for c in datasets.enabled_corpora() if c.name in _INGESTOR_REGISTRY]

    if not enabled:
        log.warning(
            "No enabled corpora found. "
            "Set enabled: true for at least one corpus in config/datasets.yaml."
        )
        return {}

    results = {}
    for corpus in enabled:
        try:
            ingestor = get_ingestor(corpus.name, config_dir=config_dir)
            result = ingestor.run(output_dir=output_dir, dry_run=dry_run)
            results[corpus.name] = result
        except Exception as exc:
            log.error("Ingestion failed for corpus '%s': %s", corpus.name, exc, exc_info=True)

    return results
