"""
lockbit.py
----------
Ingestor for the LockBit leak corpus (2024).

Source format
-------------
JSON or CSV files. Some distributions include SQL dumps — these must be
pre-converted to CSV before ingestion (see notes in config/datasets.yaml).

Schema variation
----------------
LockBit leaks contain heterogeneous data subsets:
- Affiliate panel communications
- Ticket/support messages
- Operator-to-affiliate messages

The `ticket_id` / `id` column is mapped to `msg_id`. The `created_at`
timestamp column uses standard ISO-8601 format across known distributions.

Corpus details
--------------
Data released following Operation Cronos (Feb 2024) and a subsequent
retaliatory dump (May 2024). Schema varies by subset.
"""

from __future__ import annotations

import pandas as pd

from taig.ingestion.base import BaseIngestor
from taig.utils.config import CorpusConfig, IngestionConfig


class LockBitIngestor(BaseIngestor):
    """Ingestor for the LockBit RaaS leak corpus."""

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize LockBit records to canonical schema.

        LockBit data may mix affiliate-panel records and operator chat
        messages in the same files. All are normalized to the same schema;
        subset type is preserved in metadata via a `record_type` column
        if present.
        """
        df = df.copy()

        # Some LockBit distributions nest messages under a `message` key
        # inside a JSON object rather than a top-level `body` field.
        # Promote it before alias mapping.
        if "message" in df.columns and "body" not in df.columns:
            df["body"] = df["message"]

        # Similarly, `description` is used in ticket-style records
        if "description" in df.columns and "body" not in df.columns:
            df["body"] = df["description"]

        # Apply alias mapping
        df = self._apply_aliases(df)

        # Coerce sender to string
        for col in ("sender", "recipient"):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", None)

        # Preserve record_type / category in a passthrough column
        # (it will end up in IntelMessage.metadata via base class logic)
        for passthrough in ("record_type", "category", "status", "priority"):
            if passthrough in df.columns:
                # Rename with prefix so it doesn't collide with canonical columns
                # but is clearly identifiable in metadata
                df[f"lk_{passthrough}"] = df[passthrough]

        return df
