"""
config.py
---------
YAML config loader with typed dataclasses.

Usage
-----
    from taig.utils.config import load_config, PipelineConfig

    cfg = load_config("pipeline")          # loads config/pipeline.yaml
    umap_neighbors = cfg.umap.n_neighbors  # typed access
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_config_dir() -> Path:
    """Locate the config/ directory by walking up from this file's location."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config"
        if candidate.is_dir():
            return candidate
    # Fallback: assume CWD/config
    return Path.cwd() / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Pipeline config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingsConfig:
    batch_size: int = 512
    normalize: bool = True
    max_docs_per_language: int = 20_000
    cache_dir: str = "models/embedding_cache"


@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"
    random_state: int = 42
    low_memory: bool = False


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 30
    min_samples: int | None = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True


@dataclass
class VectorizerConfig:
    stop_words: str | None = None
    min_df: int = 5
    ngram_range: tuple[int, int] = (1, 2)


@dataclass
class BERTopicConfig:
    top_n_words: int = 10
    verbose: bool = True
    vectorizer: dict[str, VectorizerConfig] = field(default_factory=dict)


@dataclass
class IngestionConfig:
    min_body_length: int = 5
    dedup_column: str | None = "msg_id"


@dataclass
class NERConfig:
    confidence_threshold: float = 0.7
    max_sequence_length: int = 512


@dataclass
class ATTACKMapperConfig:
    similarity_threshold: float = 0.65
    top_k: int = 3


@dataclass
class RetrievalConfig:
    top_k: int = 10
    vector_weight: float = 0.7


@dataclass
class PipelineConfig:
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    bertopic: BERTopicConfig = field(default_factory=BERTopicConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    attack_mapper: ATTACKMapperConfig = field(default_factory=ATTACKMapperConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# ---------------------------------------------------------------------------
# Models config
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingModels:
    en: str = "all-MiniLM-L6-v2"
    multilingual: str = "paraphrase-multilingual-MiniLM-L12-v2"
    en_large: str = "all-mpnet-base-v2"


@dataclass
class NERModels:
    cybersecurity: str = "CyberPeace-Institute/CyNER"
    general: str = "urchade/gliner_medium-v2.1"
    spacy_fallback: str = "en_core_web_lg"


@dataclass
class LLMModels:
    anthropic: str = "claude-sonnet-4-6"
    ollama: str = "mistral:7b-instruct"


@dataclass
class ModelsConfig:
    embeddings: EmbeddingModels = field(default_factory=EmbeddingModels)
    ner: NERModels = field(default_factory=NERModels)
    llm: LLMModels = field(default_factory=LLMModels)
    attack_mapper_encoder: str = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Datasets config
# ---------------------------------------------------------------------------

@dataclass
class CorpusConfig:
    name: str
    enabled: bool
    display_name: str
    description: str
    raw_dir: Path
    formats_supported: list[str]
    column_aliases: dict[str, list[str]]
    languages: list[str]
    date_range: list[str] | None
    source: str
    notes: str


@dataclass
class DatasetsConfig:
    corpora: dict[str, CorpusConfig] = field(default_factory=dict)

    def enabled_corpora(self) -> list[CorpusConfig]:
        return [c for c in self.corpora.values() if c.enabled]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _build_pipeline_config(raw: dict) -> PipelineConfig:
    def _vcfg(d: dict) -> VectorizerConfig:
        ngram = d.get("ngram_range", [1, 2])
        return VectorizerConfig(
            stop_words=d.get("stop_words"),
            min_df=d.get("min_df", 5),
            ngram_range=(ngram[0], ngram[1]),
        )

    emb = raw.get("embeddings", {})
    umap = raw.get("umap", {})
    hdb = raw.get("hdbscan", {})
    bert = raw.get("bertopic", {})
    ing = raw.get("ingestion", {})
    ner = raw.get("ner", {})
    atk = raw.get("attack_mapper", {})
    ret = raw.get("retrieval", {})

    vectorizers = {k: _vcfg(v) for k, v in bert.get("vectorizer", {}).items()}

    return PipelineConfig(
        embeddings=EmbeddingsConfig(**{k: v for k, v in emb.items() if v is not None}),
        umap=UMAPConfig(**{k: v for k, v in umap.items() if v is not None}),
        hdbscan=HDBSCANConfig(**{k: v for k, v in hdb.items() if v is not None}),
        bertopic=BERTopicConfig(
            top_n_words=bert.get("top_n_words", 10),
            verbose=bert.get("verbose", True),
            vectorizer=vectorizers,
        ),
        ingestion=IngestionConfig(**{k: v for k, v in ing.items() if v is not None}),
        ner=NERConfig(**{k: v for k, v in ner.items() if v is not None}),
        attack_mapper=ATTACKMapperConfig(**{k: v for k, v in atk.items() if v is not None}),
        retrieval=RetrievalConfig(**{k: v for k, v in ret.items() if v is not None}),
    )


def _build_models_config(raw: dict) -> ModelsConfig:
    emb = raw.get("embeddings", {})
    ner = raw.get("ner", {})
    llm = raw.get("llm", {})
    atk_enc = raw.get("attack_mapper", {}).get("technique_encoder", "all-MiniLM-L6-v2")

    return ModelsConfig(
        embeddings=EmbeddingModels(
            en=emb.get("en", "all-MiniLM-L6-v2"),
            multilingual=emb.get("multilingual", "paraphrase-multilingual-MiniLM-L12-v2"),
            en_large=emb.get("en_large", "all-mpnet-base-v2"),
        ),
        ner=NERModels(
            cybersecurity=ner.get("cybersecurity", "CyberPeace-Institute/CyNER"),
            general=ner.get("general", "urchade/gliner_medium-v2.1"),
            spacy_fallback=ner.get("spacy_fallback", "en_core_web_lg"),
        ),
        llm=LLMModels(
            anthropic=llm.get("anthropic", "claude-sonnet-4-6"),
            ollama=llm.get("ollama", "mistral:7b-instruct"),
        ),
        attack_mapper_encoder=atk_enc,
    )


def _build_datasets_config(raw: dict) -> DatasetsConfig:
    corpora: dict[str, CorpusConfig] = {}
    for name, entry in raw.get("corpora", {}).items():
        corpora[name] = CorpusConfig(
            name=name,
            enabled=entry.get("enabled", False),
            display_name=entry.get("display_name", name),
            description=entry.get("description", ""),
            raw_dir=Path(entry.get("raw_dir", f"data/raw/{name}")),
            formats_supported=entry.get("formats_supported", ["json", "csv"]),
            column_aliases=entry.get("column_aliases", {}),
            languages=entry.get("languages", ["en"]),
            date_range=entry.get("date_range"),
            source=entry.get("source", ""),
            notes=entry.get("notes", ""),
        )
    return DatasetsConfig(corpora=corpora)


_CONFIG_BUILDERS = {
    "pipeline": (_build_pipeline_config, "pipeline.yaml"),
    "models": (_build_models_config, "models.yaml"),
    "datasets": (_build_datasets_config, "datasets.yaml"),
}


def load_config(
    name: str,
    config_dir: Path | str | None = None,
) -> PipelineConfig | ModelsConfig | DatasetsConfig:
    """Load a named config from the config/ directory.

    Parameters
    ----------
    name:
        One of "pipeline", "models", or "datasets".
    config_dir:
        Override the config directory. If None, auto-detected by walking up
        from this file.

    Returns
    -------
    Typed config dataclass corresponding to *name*.
    """
    if name not in _CONFIG_BUILDERS:
        raise ValueError(f"Unknown config name {name!r}. Choose from: {list(_CONFIG_BUILDERS)}")

    builder, filename = _CONFIG_BUILDERS[name]

    if config_dir is None:
        # Also honour an env var for CI/testing flexibility
        env_override = os.environ.get("TAIG_CONFIG_DIR")
        cfg_dir = Path(env_override) if env_override else _find_config_dir()
    else:
        cfg_dir = Path(config_dir)

    path = cfg_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Searched config_dir: {cfg_dir}"
        )

    raw = _load_yaml(path)
    return builder(raw)
