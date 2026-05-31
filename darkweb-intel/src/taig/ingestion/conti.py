"""
conti.py
--------
Ingestor for the Conti ransomware leak corpus (2022).

Source format
-------------
JSON files (list-of-dicts or {"messages": [...]}) or CSV files.
Column names vary across archive mirrors — all variants are mapped in
config/datasets.yaml under conti.column_aliases.

Corpus details
--------------
~60,000 internal chat messages (Russian and English) leaked in Feb 2022.
GitHub mirror: https://github.com/TheParmak/conti-leaks-englished
Archive.org:   search "conti leaks 2022"
"""

from __future__ import annotations

import pandas as pd

from taig.ingestion.base import BaseIngestor
from taig.utils.config import CorpusConfig, IngestionConfig


class ContiIngestor(BaseIngestor):
    """Ingestor for the Conti ransomware chat leak corpus.

    No special loading or timestamp handling is required beyond the
    base class defaults — Conti timestamps are standard ISO-8601 or
    datetime strings parseable by pd.to_datetime().
    """

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Conti column aliases and ensure canonical schema.

        Conti archives use multiple column name conventions depending on
        the mirror source. The alias mapping in datasets.yaml covers all
        known variants. No further dataset-specific transforms are needed.
        """
        df = self._apply_aliases(df)

        # Some mirrors store the fromId/toId as integers — coerce to string
        # so the actor field is consistently typed.
        for col in ("sender", "recipient"):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", None)

        return df
