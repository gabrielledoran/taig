"""
base.py
-------
Abstract base class for all TAIG corpus ingestors.

Subclasses must implement:
  - normalize(df) → map source columns to canonical schema

Everything else — file loading, cleaning, language detection, Parquet
output — is handled here so dataset-specific code stays minimal.

Pipeline contract
-----------------
run() calls the following steps in order:

  load_raw()         Read files from corpus.raw_dir → raw DataFrame
      ↓
  normalize()        Map source column names → canonical names
      ↓
  clean()            Dedup, filter short messages, parse timestamps
      ↓
  detect_languages() Add/fill `language` column via langdetect
      ↓
  to_parquet()       Validate schema, serialize, write to disk
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from taig.ingestion.schemas import (
    PARQUET_COLUMNS,
    IngestorResult,
    IntelMessage,
    messages_to_dataframe,
)
from taig.utils.config import CorpusConfig, IngestionConfig
from taig.utils.logging import get_logger
from taig.utils.storage import ensure_dir, write_parquet

# ---------------------------------------------------------------------------
# Language detection — inline here to keep the ingestion package self-contained.
# A richer implementation lives in taig.preprocessing.language (Phase 3).
# ---------------------------------------------------------------------------
try:
    from langdetect import LangDetectException, detect as _ld_detect

    def _detect_lang(text: str) -> str:
        try:
            return _ld_detect(text[:500])  # cap for speed
        except LangDetectException:
            return "unknown"
        except Exception:
            return "unknown"

except ImportError:  # graceful degradation in lean environments
    def _detect_lang(text: str) -> str:  # type: ignore[misc]
        return "unknown"


# ---------------------------------------------------------------------------
# BaseIngestor
# ---------------------------------------------------------------------------

class BaseIngestor(ABC):
    """Abstract base for all TAIG corpus ingestors.

    Parameters
    ----------
    corpus:
        CorpusConfig loaded from config/datasets.yaml for this dataset.
    ingestion_cfg:
        IngestionConfig from config/pipeline.yaml (thresholds, dedup column).
    """

    #: Canonical column names that normalize() must produce.
    CANONICAL_COLUMNS = ["msg_id", "timestamp", "sender", "recipient", "body", "source_file"]

    def __init__(self, corpus: CorpusConfig, ingestion_cfg: IngestionConfig) -> None:
        self.corpus = corpus
        self.cfg = ingestion_cfg
        self.log = get_logger(f"taig.ingestion.{corpus.name}")

    # ------------------------------------------------------------------
    # File loading helpers (shared, overridable)
    # ------------------------------------------------------------------

    def _load_json_files(self, directory: Path) -> pd.DataFrame:
        """Load all .json files in *directory* into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        json_files = sorted(directory.glob("*.json"))

        if not json_files:
            self.log.debug("No .json files found in %s", directory)
            return pd.DataFrame()

        for fpath in json_files:
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    raw = json.load(fh)

                # Normalize: accept list-of-dicts or {"messages": [...]}
                if isinstance(raw, list):
                    records = raw
                elif isinstance(raw, dict):
                    # Try common envelope keys
                    for key in ("messages", "data", "records", "items"):
                        if key in raw and isinstance(raw[key], list):
                            records = raw[key]
                            break
                    else:
                        records = [raw]
                else:
                    self.log.warning("Unexpected JSON structure in %s — skipping", fpath.name)
                    continue

                df = pd.DataFrame(records)
                df["_source_file"] = fpath.name
                frames.append(df)
                self.log.debug("Loaded %d records from %s", len(df), fpath.name)

            except Exception as exc:
                self.log.warning("Failed to load %s: %s", fpath.name, exc)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_csv_files(self, directory: Path) -> pd.DataFrame:
        """Load all .csv files in *directory* into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        csv_files = sorted(directory.glob("*.csv"))

        if not csv_files:
            self.log.debug("No .csv files found in %s", directory)
            return pd.DataFrame()

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath, encoding="utf-8", encoding_errors="replace", low_memory=False)
                df["_source_file"] = fpath.name
                frames.append(df)
                self.log.debug("Loaded %d records from %s", len(df), fpath.name)
            except Exception as exc:
                self.log.warning("Failed to load %s: %s", fpath.name, exc)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_raw(self) -> pd.DataFrame:
        """Load all supported raw files from corpus.raw_dir.

        Tries JSON first, then CSV. Subclasses may override for dataset-
        specific loading (e.g. Matrix JSON, SQL dumps).

        Returns an empty DataFrame (not an exception) when no files are
        present — callers check for empty output via len(df) == 0.
        """
        raw_dir = Path(self.corpus.raw_dir)

        if not raw_dir.exists():
            self.log.warning(
                "Raw data directory does not exist: %s\n"
                "Place %s files there and set enabled: true in config/datasets.yaml.",
                raw_dir,
                self.corpus.name,
            )
            return pd.DataFrame()

        formats = self.corpus.formats_supported

        df_json = self._load_json_files(raw_dir) if "json" in formats else pd.DataFrame()
        df_csv = self._load_csv_files(raw_dir) if "csv" in formats else pd.DataFrame()

        parts = [d for d in (df_json, df_csv) if not d.empty]
        if not parts:
            self.log.warning(
                "No files found in %s for corpus '%s'.",
                raw_dir,
                self.corpus.name,
            )
            return pd.DataFrame()

        combined = pd.concat(parts, ignore_index=True)
        self.log.info("Loaded %d raw records from %s", len(combined), raw_dir)
        return combined

    # ------------------------------------------------------------------
    # Column alias normalization (shared)
    # ------------------------------------------------------------------

    def _apply_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename source columns to canonical names using corpus.column_aliases.

        For each canonical name (msg_id, timestamp, sender, recipient, body),
        look through the alias list and rename the first matching column.
        Columns that are already named canonically are left alone.
        Missing columns are added as None.
        """
        df = df.copy()
        aliases: dict[str, list[str]] = self.corpus.column_aliases

        for canonical, alias_list in aliases.items():
            if canonical in df.columns:
                continue  # already correct name
            for alias in alias_list:
                if alias in df.columns:
                    df = df.rename(columns={alias: canonical})
                    break

        # Propagate _source_file → source_file
        if "source_file" not in df.columns:
            df["source_file"] = df.get("_source_file", "unknown")

        # Guarantee all canonical columns exist
        for col in self.CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df

    # ------------------------------------------------------------------
    # normalize() — subclasses must implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map source columns to canonical schema.

        Must call self._apply_aliases(df) as a first step, then apply any
        dataset-specific transforms (timestamp format, nested JSON fields, etc).

        Returns a DataFrame with at minimum the columns in CANONICAL_COLUMNS.
        """

    # ------------------------------------------------------------------
    # Cleaning (shared, concrete)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: Any) -> str:
        """Normalize whitespace and line endings in a message body."""
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _make_message_id(dataset: str, source_file: str, index: int, existing_id: Any) -> str:
        """Return a stable string message ID.

        Uses existing_id when non-null and non-empty; otherwise generates a
        deterministic ID from dataset + source_file + row index.
        """
        if existing_id is not None and not pd.isna(existing_id):
            candidate = str(existing_id).strip()
            if candidate:
                return candidate

        raw = f"{dataset}::{source_file}::{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared cleaning applied after normalize().

        Steps:
        1. Drop rows with null/empty body.
        2. Clean body text (whitespace normalization).
        3. Drop messages shorter than cfg.min_body_length.
        4. Parse timestamp column to UTC datetime.
        5. Deduplicate on cfg.dedup_column (if set).
        """
        if df.empty:
            return df

        before = len(df)

        # 1. Drop null / empty body
        df = df[df["body"].notna()]
        df = df[df["body"].astype(str).str.strip() != ""]

        # 2. Clean body text
        df = df.copy()
        df["body"] = df["body"].apply(self._clean_text)

        # 3. Length filter
        min_len = self.cfg.min_body_length
        df = df[df["body"].str.len() >= min_len]

        # 4. Parse timestamps (skip if already datetime)
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # 5. Deduplicate
        dedup_col = self.cfg.dedup_column
        if dedup_col and dedup_col in df.columns:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=[dedup_col], keep="first")
            dropped = before_dedup - len(df)
            if dropped:
                self.log.debug("Deduplication on '%s' removed %d rows", dedup_col, dropped)

        self.log.info(
            "clean(): %d → %d records (dropped %d)",
            before,
            len(df),
            before - len(df),
        )
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Language detection (shared, concrete)
    # ------------------------------------------------------------------

    def detect_languages(self, df: pd.DataFrame, text_col: str = "body") -> pd.DataFrame:
        """Add a `language` column to *df* using langdetect.

        Existing values are preserved; only rows with missing/unknown language
        are re-detected to avoid redundant computation on repeated runs.
        """
        if df.empty:
            df["language"] = pd.Series(dtype=str)
            return df

        df = df.copy()
        if "language" not in df.columns:
            df["language"] = None

        # Only detect where language is missing or unknown
        needs_detection = df["language"].isna() | (df["language"] == "unknown")
        n_detect = needs_detection.sum()

        if n_detect == 0:
            return df

        self.log.info("Detecting language for %d messages...", n_detect)

        df.loc[needs_detection, "language"] = (
            df.loc[needs_detection, text_col].apply(_detect_lang)
        )

        dist = df["language"].value_counts().to_dict()
        self.log.info("Language distribution: %s", dist)
        return df

    # ------------------------------------------------------------------
    # Schema validation and Parquet output
    # ------------------------------------------------------------------

    def _to_intel_messages(self, df: pd.DataFrame) -> tuple[list[IntelMessage], list[str]]:
        """Convert cleaned DataFrame to validated IntelMessage objects.

        Returns (valid_messages, error_strings) — errors are logged but do
        not abort the run.
        """
        messages: list[IntelMessage] = []
        errors: list[str] = []

        # Columns that don't belong in canonical schema go into metadata
        extra_cols = [
            c for c in df.columns
            if c not in PARQUET_COLUMNS
            and c not in ("msg_id", "sender", "recipient", "body", "_source_file")
        ]

        for idx, row in df.iterrows():
            # Build metadata dict from extra columns + recipient
            meta: dict[str, Any] = {}
            if "recipient" in df.columns and pd.notna(row.get("recipient")):
                meta["recipient"] = str(row["recipient"])
            for col in extra_cols:
                val = row.get(col)
                if pd.notna(val):
                    meta[col] = val

            try:
                msg = IntelMessage(
                    message_id=self._make_message_id(
                        self.corpus.name,
                        str(row.get("source_file", "unknown")),
                        int(idx),
                        row.get("msg_id"),
                    ),
                    dataset=self.corpus.name,
                    actor=str(row["sender"]) if pd.notna(row.get("sender")) else None,
                    timestamp=row.get("timestamp") if pd.notna(row.get("timestamp")) else None,
                    language=str(row.get("language", "unknown")),
                    raw_text=str(row["body"]),
                    source_file=str(row.get("source_file", "unknown")),
                    metadata=meta,
                )
                messages.append(msg)
            except Exception as exc:
                errors.append(f"Row {idx}: {exc}")

        if errors:
            self.log.warning("%d rows failed schema validation", len(errors))
            for e in errors[:5]:
                self.log.debug("  %s", e)

        return messages, errors

    def to_parquet(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> tuple[Path, list[str]]:
        """Validate schema and write output Parquet file.

        Returns (output_path, validation_errors).
        """
        messages, errors = self._to_intel_messages(df)
        out_df = messages_to_dataframe(messages)

        ensure_dir(output_dir)
        out_path = output_dir / f"{self.corpus.name}_intel.parquet"
        write_parquet(out_df, out_path)

        return out_path, errors

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self, output_dir: Path | None = None, dry_run: bool = False) -> IngestorResult:
        """Execute the full ingestion pipeline for this corpus.

        Steps: load_raw → normalize → clean → detect_languages → to_parquet

        Parameters
        ----------
        output_dir:
            Where to write the output Parquet. Defaults to data/processed/
            relative to the project root (located by walking up from this file).
        dry_run:
            If True, run all steps but skip writing the output file.

        Returns
        -------
        IngestorResult with statistics about the run.
        """
        result = IngestorResult(corpus=self.corpus.name)

        if output_dir is None:
            output_dir = _find_processed_dir()

        self.log.info("=== Ingesting corpus: %s ===", self.corpus.name)

        # Step 1: Load
        raw = self.load_raw()
        result.records_raw = len(raw)

        if raw.empty:
            self.log.warning(
                "No records loaded for corpus '%s'. "
                "Is the data in %s and enabled: true in datasets.yaml?",
                self.corpus.name,
                self.corpus.raw_dir,
            )
            return result

        # Step 2: Normalize
        normalized = self.normalize(raw)
        result.records_after_normalize = len(normalized)

        # Step 3: Clean
        cleaned = self.clean(normalized)
        result.records_after_clean = len(cleaned)

        # Step 4: Language detection
        with_lang = self.detect_languages(cleaned)

        result.language_distribution = (
            with_lang["language"].value_counts().to_dict()
            if "language" in with_lang.columns
            else {}
        )

        if dry_run:
            self.log.info("dry_run=True — skipping Parquet write. %d records ready.", len(with_lang))
            result.records_written = len(with_lang)
            return result

        # Step 5: Write
        out_path, errors = self.to_parquet(with_lang, output_dir)
        result.errors = errors
        result.records_written = len(with_lang) - len(errors)
        result.output_path = str(out_path)

        self.log.info(
            "=== Done: %d records written to %s (drop rate: %.1f%%) ===",
            result.records_written,
            out_path,
            result.drop_rate * 100,
        )
        return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _find_processed_dir() -> Path:
    """Walk up from this file to find data/processed/."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "processed"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "data" / "processed"
