"""
blackbasta.py
-------------
Ingestor for the Black Basta ransomware chat leak corpus (2024).

Source format
-------------
Matrix protocol JSON export files.

Key differences from other corpora
-----------------------------------
1. Timestamps are Unix milliseconds (origin_server_ts), not ISO strings.
   Must divide by 1000 before pd.to_datetime().

2. Message body may be nested inside a `content` dict:
   {"type": "m.room.message", "content": {"msgtype": "m.text", "body": "hello"}}
   Only records with type == "m.room.message" and msgtype == "m.text" are kept.

3. `sender` is a Matrix user ID (@username:homeserver) — preserved as-is.

4. `room_id` (Matrix channel) is stored in metadata as a proxy for recipient.

Corpus details
--------------
~200,000 internal messages from Black Basta RaaS operations, leaked via
threat intelligence community channels in early 2024.
Date range: approximately 2022-09 to 2024-09.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from taig.ingestion.base import BaseIngestor
from taig.utils.config import CorpusConfig, IngestionConfig


class BlackBastaIngestor(BaseIngestor):
    """Ingestor for the Black Basta Matrix chat leak corpus."""

    def load_raw(self) -> pd.DataFrame:
        """Load Black Basta JSON exports.

        Matrix exports can be structured as:
        - A list of event objects at the top level
        - A dict with a "chunk" or "events" key containing the list

        Handles both.
        """
        raw_dir = Path(self.corpus.raw_dir)
        if not raw_dir.exists():
            self.log.warning(
                "Raw data directory does not exist: %s\n"
                "Place Black Basta Matrix JSON exports there.",
                raw_dir,
            )
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for fpath in sorted(raw_dir.glob("*.json")):
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    raw = json.load(fh)

                # Unwrap common Matrix export envelopes
                if isinstance(raw, dict):
                    for key in ("chunk", "events", "messages", "data"):
                        if key in raw and isinstance(raw[key], list):
                            raw = raw[key]
                            break
                    else:
                        raw = [raw]

                if not isinstance(raw, list):
                    self.log.warning("Unexpected format in %s — skipping", fpath.name)
                    continue

                df = pd.DataFrame(raw)
                df["_source_file"] = fpath.name
                frames.append(df)
                self.log.debug("Loaded %d events from %s", len(df), fpath.name)

            except Exception as exc:
                self.log.warning("Failed to load %s: %s", fpath.name, exc)

        if not frames:
            self.log.warning("No .json files found in %s", raw_dir)
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        self.log.info("Loaded %d raw events from %s", len(combined), raw_dir)
        return combined

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Black Basta Matrix events to canonical schema.

        Special handling:
        - Filter to m.room.message events with m.text msgtype
        - Extract nested body from content dict
        - Convert millisecond timestamps to UTC datetime
        """
        if df.empty:
            return df

        df = df.copy()

        # --- 1. Filter to text messages only ---
        # Keep rows where type == "m.room.message" if the column exists
        if "type" in df.columns:
            df = df[df["type"] == "m.room.message"].copy()
            self.log.debug("After type filter: %d records", len(df))

        # --- 2. Extract body from nested content dict ---
        if "content" in df.columns:
            def _extract_body(content: object) -> str | None:
                if isinstance(content, dict):
                    # Only process text messages
                    if content.get("msgtype") == "m.text":
                        return content.get("body")
                    # Some exports store body directly
                    return content.get("body")
                if isinstance(content, str):
                    # content may already be the body string
                    return content
                return None

            df["body"] = df["content"].apply(_extract_body)

        # --- 3. Apply alias mapping ---
        df = self._apply_aliases(df)

        # --- 4. Convert millisecond timestamps ---
        # origin_server_ts (aliases to `timestamp`) is Unix ms, not seconds.
        # Convert the full column at once to avoid mixed-dtype assignment issues.
        if "timestamp" in df.columns:
            numeric_vals = pd.to_numeric(df["timestamp"], errors="coerce")
            if numeric_vals.notna().any():
                median_val = numeric_vals.dropna().median()
                if median_val > 1e10:
                    # Milliseconds — divide by 1000 before passing to to_datetime
                    df["timestamp"] = pd.to_datetime(
                        numeric_vals / 1000, unit="s", utc=True, errors="coerce"
                    )
                else:
                    df["timestamp"] = pd.to_datetime(
                        numeric_vals, unit="s", utc=True, errors="coerce"
                    )

        # --- 5. Preserve room_id in metadata carrier column ---
        # room_id maps to `recipient` via alias, then into metadata in base class
        for col in ("sender",):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", None)

        return df
