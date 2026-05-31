"""
babuk.py
--------
Ingestor for the Babuk ransomware leak corpus (2021).

Source format
-------------
JSON, CSV, or plain-text (.txt) files.
The corpus is small (~few thousand messages) compared to Conti.
Source code files are ignored; only communication/chat records are ingested.

Corpus details
--------------
Internal communications from the Babuk RaaS group, leaked mid-2021 by a
disgruntled member alongside source code. Publicly archived.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from taig.ingestion.base import BaseIngestor
from taig.utils.config import CorpusConfig, IngestionConfig
from taig.utils.logging import get_logger

log = get_logger(__name__)


class BabukIngestor(BaseIngestor):
    """Ingestor for the Babuk ransomware communication corpus."""

    def load_raw(self) -> pd.DataFrame:
        """Load Babuk files.

        Extends base load_raw() to also handle .txt files that may contain
        newline-delimited JSON (ndjson) or plain-text chat logs.
        """
        # Start with the base JSON + CSV loader
        df_base = super().load_raw()

        raw_dir = Path(self.corpus.raw_dir)
        if not raw_dir.exists():
            return df_base

        # Also try .txt files — may be ndjson or simple line-per-message chat
        txt_frames: list[pd.DataFrame] = []
        for fpath in sorted(raw_dir.glob("*.txt")):
            try:
                df_txt = self._load_txt_file(fpath)
                if not df_txt.empty:
                    txt_frames.append(df_txt)
            except Exception as exc:
                self.log.warning("Failed to load %s: %s", fpath.name, exc)

        parts = [d for d in ([df_base] + txt_frames) if not d.empty]
        if not parts:
            return pd.DataFrame()

        combined = pd.concat(parts, ignore_index=True)
        self.log.info("Babuk: %d records after combining all sources", len(combined))
        return combined

    @staticmethod
    def _load_txt_file(fpath: Path) -> pd.DataFrame:
        """Parse a .txt file as ndjson; fall back to one-message-per-line."""
        rows: list[dict] = []
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except json.JSONDecodeError:
                    # Plain text line — treat as a body-only record
                    rows.append({"body": line, "msg_id": f"{fpath.stem}_{line_num}"})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["_source_file"] = fpath.name
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Babuk column aliases.

        Babuk archives use simpler column names than Conti — the alias
        mapping in datasets.yaml covers the known variants.
        """
        df = self._apply_aliases(df)

        for col in ("sender", "recipient"):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", None)

        return df
