"""
schemas.py
----------
Canonical data contract for all ingested threat intelligence messages.

Every ingestor, regardless of source corpus, must produce records that
conform to IntelMessage. This schema is the single source of truth for
column names and types in data/processed/*.parquet files.

Parquet serialization notes
---------------------------
- `timestamp` is stored as UTC datetime64[us] (timezone-aware).
- `metadata` is serialized to a JSON string in Parquet for portability,
  and deserialized back to dict on read via `parse_metadata_column()`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Canonical DataFrame column names (in Parquet output order).
PARQUET_COLUMNS: list[str] = [
    "message_id",
    "dataset",
    "actor",
    "timestamp",
    "language",
    "raw_text",
    "source_file",
    "metadata",
]


class IntelMessage(BaseModel):
    """A single normalized threat actor communication.

    This model is the ingestion layer's output contract. Every field
    downstream code can assume exists and has the correct type.
    """

    message_id: str = Field(
        description="Unique identifier for this message. "
        "Uses the source ID when available; falls back to a deterministic hash.",
    )
    dataset: str = Field(
        description="Source corpus name, e.g. 'conti', 'babuk', 'black_basta'.",
    )
    actor: str | None = Field(
        default=None,
        description="Sender identifier as it appears in the source data. "
        "May be a username, alias, or user ID.",
    )
    timestamp: datetime | None = Field(
        default=None,
        description="Message timestamp in UTC. None if unparseable.",
    )
    language: str = Field(
        default="unknown",
        description="ISO 639-1 language code detected by langdetect, "
        "or 'unknown' if detection failed.",
    )
    raw_text: str = Field(
        description="Cleaned message body text.",
    )
    source_file: str = Field(
        description="Filename (basename) of the raw source file this record came from.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset-specific fields that do not map to the canonical schema. "
        "Examples: room_id (Black Basta), ticket_id (LockBit), recipient (Conti).",
    )

    @field_validator("message_id")
    @classmethod
    def message_id_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message_id must not be empty")
        return v.strip()

    @field_validator("raw_text")
    @classmethod
    def raw_text_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("raw_text must not be empty")
        return v

    @field_validator("dataset")
    @classmethod
    def dataset_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("dataset must not be empty")
        return v.lower().strip()


class IngestorResult(BaseModel):
    """Summary statistics produced at the end of a single ingestion run."""

    corpus: str
    records_raw: int = 0
    records_after_normalize: int = 0
    records_after_clean: int = 0
    records_written: int = 0
    output_path: str = ""
    language_distribution: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        if self.records_raw == 0:
            return 0.0
        return round(1 - self.records_written / self.records_raw, 4)


# ---------------------------------------------------------------------------
# DataFrame ↔ Parquet helpers
# ---------------------------------------------------------------------------

def messages_to_dataframe(messages: list[IntelMessage]) -> pd.DataFrame:
    """Convert a list of IntelMessage objects to a Parquet-ready DataFrame.

    The `metadata` column is serialized to a JSON string.
    """
    rows = []
    for m in messages:
        row = m.model_dump()
        row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False, default=str)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=PARQUET_COLUMNS)

    df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)

    # Ensure timestamp is timezone-aware UTC datetime64
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    return df


def dataframe_to_messages(df: pd.DataFrame) -> list[IntelMessage]:
    """Deserialize a DataFrame (read from Parquet) back to IntelMessage objects.

    The `metadata` column is deserialized from JSON string.
    """
    messages = []
    for _, row in df.iterrows():
        d = row.to_dict()
        meta = d.get("metadata", "{}")
        if isinstance(meta, str):
            try:
                d["metadata"] = json.loads(meta)
            except json.JSONDecodeError:
                d["metadata"] = {}
        messages.append(IntelMessage(**d))
    return messages


def parse_metadata_column(df: pd.DataFrame) -> pd.DataFrame:
    """Deserialize the metadata column in-place from JSON string to dict.

    Use this after read_parquet() when you need to access metadata fields.
    """
    if "metadata" in df.columns:
        df = df.copy()
        df["metadata"] = df["metadata"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else (v or {})
        )
    return df
