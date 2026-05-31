"""
test_ingestion.py
-----------------
Unit tests for the TAIG ingestion layer.

All tests use synthetic data fixtures defined in conftest.py.
No real corpus files are required. Tests are deterministic and fast.

Test categories
---------------
- Schema validation (IntelMessage)
- Column alias normalization (BaseIngestor._apply_aliases)
- Cleaning logic (dedup, min_body_length, timestamp parsing)
- Language detection (graceful fallback)
- ContiIngestor end-to-end
- BabukIngestor end-to-end (JSON + TXT loading)
- BlackBastaIngestor (ms timestamp conversion, type filtering)
- LockBitIngestor end-to-end
- Registry (get_ingestor, list_corpora)
- Parquet output (schema, column names, row count)
- CLI argument parsing
- Empty / missing data handling
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from taig.ingestion.babuk import BabukIngestor
from taig.ingestion.base import BaseIngestor
from taig.ingestion.blackbasta import BlackBastaIngestor
from taig.ingestion.conti import ContiIngestor
from taig.ingestion.lockbit import LockBitIngestor
from taig.ingestion.schemas import (
    PARQUET_COLUMNS,
    IngestorResult,
    IntelMessage,
    dataframe_to_messages,
    messages_to_dataframe,
    parse_metadata_column,
)


# ===========================================================================
# Schema tests
# ===========================================================================

class TestIntelMessageSchema:
    def test_valid_message(self):
        msg = IntelMessage(
            message_id="abc123",
            dataset="conti",
            actor="stern",
            timestamp=datetime(2022, 2, 14, 10, 0, 0, tzinfo=timezone.utc),
            language="ru",
            raw_text="готов к работе",
            source_file="conti_leak.json",
            metadata={"recipient": "johny"},
        )
        assert msg.message_id == "abc123"
        assert msg.dataset == "conti"
        assert msg.language == "ru"

    def test_minimal_message_defaults(self):
        msg = IntelMessage(
            message_id="x",
            dataset="babuk",
            raw_text="hello world",
            source_file="f.json",
        )
        assert msg.actor is None
        assert msg.timestamp is None
        assert msg.language == "unknown"
        assert msg.metadata == {}

    def test_empty_message_id_raises(self):
        with pytest.raises(Exception):
            IntelMessage(
                message_id="   ",
                dataset="conti",
                raw_text="some text",
                source_file="f.json",
            )

    def test_empty_raw_text_raises(self):
        with pytest.raises(Exception):
            IntelMessage(
                message_id="1",
                dataset="conti",
                raw_text="",
                source_file="f.json",
            )

    def test_empty_dataset_raises(self):
        with pytest.raises(Exception):
            IntelMessage(
                message_id="1",
                dataset="",
                raw_text="hello",
                source_file="f.json",
            )

    def test_dataset_lowercased(self):
        msg = IntelMessage(
            message_id="1",
            dataset="CONTI",
            raw_text="hello",
            source_file="f.json",
        )
        assert msg.dataset == "conti"


class TestIngestorResult:
    def test_drop_rate_zero_when_no_raw(self):
        r = IngestorResult(corpus="test", records_raw=0, records_written=0)
        assert r.drop_rate == 0.0

    def test_drop_rate_calculation(self):
        r = IngestorResult(corpus="test", records_raw=100, records_written=80)
        assert r.drop_rate == pytest.approx(0.2)


class TestParquetSerialization:
    def test_round_trip(self):
        messages = [
            IntelMessage(
                message_id="1",
                dataset="conti",
                actor="stern",
                timestamp=datetime(2022, 2, 14, tzinfo=timezone.utc),
                language="ru",
                raw_text="готов к работе",
                source_file="test.json",
                metadata={"recipient": "johny", "extra": 42},
            )
        ]
        df = messages_to_dataframe(messages)
        assert list(df.columns) == PARQUET_COLUMNS
        assert len(df) == 1
        assert df["message_id"].iloc[0] == "1"
        assert df["dataset"].iloc[0] == "conti"
        assert isinstance(df["metadata"].iloc[0], str)  # serialized to JSON string

    def test_round_trip_empty(self):
        df = messages_to_dataframe([])
        assert list(df.columns) == PARQUET_COLUMNS
        assert len(df) == 0

    def test_parse_metadata_column(self):
        messages = [
            IntelMessage(
                message_id="1",
                dataset="conti",
                raw_text="hello",
                source_file="f.json",
                metadata={"recipient": "johny"},
            )
        ]
        df = messages_to_dataframe(messages)
        df2 = parse_metadata_column(df)
        assert isinstance(df2["metadata"].iloc[0], dict)
        assert df2["metadata"].iloc[0]["recipient"] == "johny"


# ===========================================================================
# BaseIngestor tests (via ContiIngestor as concrete subclass)
# ===========================================================================

class TestBaseIngestorCleaning:
    def test_clean_removes_null_body(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "body": ["hello world this is a test", None, "another valid message"],
            "msg_id": ["1", "2", "3"],
            "sender": ["a", "b", "c"],
            "recipient": ["x", "y", "z"],
            "timestamp": [None, None, None],
            "source_file": ["f.json", "f.json", "f.json"],
        })
        cleaned = ingestor.clean(df)
        assert len(cleaned) == 2
        assert "2" not in cleaned["msg_id"].values

    def test_clean_removes_short_messages(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "body": ["hi", "this is a long enough message", "ok"],
            "msg_id": ["1", "2", "3"],
            "sender": ["a", "b", "c"],
            "recipient": ["x", "y", "z"],
            "timestamp": [None, None, None],
            "source_file": ["f.json", "f.json", "f.json"],
        })
        cleaned = ingestor.clean(df)
        assert len(cleaned) == 1
        assert cleaned["msg_id"].iloc[0] == "2"

    def test_clean_deduplicates(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "body": ["first message text here", "second message text here"],
            "msg_id": ["1", "1"],
            "sender": ["a", "a"],
            "recipient": ["x", "x"],
            "timestamp": [None, None],
            "source_file": ["f.json", "f.json"],
        })
        cleaned = ingestor.clean(df)
        assert len(cleaned) == 1

    def test_clean_parses_timestamps(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "body": ["valid message body text here"],
            "msg_id": ["1"],
            "sender": ["stern"],
            "recipient": ["johny"],
            "timestamp": ["2022-02-14 10:00:00"],
            "source_file": ["f.json"],
        })
        cleaned = ingestor.clean(df)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])

    def test_clean_normalizes_whitespace(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "body": ["hello   world\r\nthis is   a test"],
            "msg_id": ["1"],
            "sender": ["a"],
            "recipient": ["b"],
            "timestamp": [None],
            "source_file": ["f.json"],
        })
        cleaned = ingestor.clean(df)
        assert "  " not in cleaned["body"].iloc[0]
        assert "\r" not in cleaned["body"].iloc[0]

    def test_clean_handles_empty_dataframe(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        result = ingestor.clean(pd.DataFrame())
        assert result.empty

    def test_make_message_id_uses_existing(self):
        result = BaseIngestor._make_message_id("conti", "file.json", 0, "abc-123")
        assert result == "abc-123"

    def test_make_message_id_generates_hash_when_null(self):
        result = BaseIngestor._make_message_id("conti", "file.json", 0, None)
        assert len(result) == 16
        # Deterministic
        assert result == BaseIngestor._make_message_id("conti", "file.json", 0, None)

    def test_make_message_id_generates_hash_when_empty(self):
        result = BaseIngestor._make_message_id("conti", "file.json", 5, "  ")
        assert len(result) == 16


class TestColumnAliases:
    def test_apply_aliases_renames_columns(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "id": ["1"],
            "date": ["2022-01-01"],
            "from": ["stern"],
            "to": ["johny"],
            "text": ["hello world from test"],
            "_source_file": ["f.json"],
        })
        result = ingestor._apply_aliases(df)
        assert "msg_id" in result.columns
        assert "timestamp" in result.columns
        assert "sender" in result.columns
        assert "recipient" in result.columns
        assert "body" in result.columns

    def test_apply_aliases_preserves_already_canonical(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "msg_id": ["1"],
            "timestamp": ["2022-01-01"],
            "sender": ["stern"],
            "recipient": ["johny"],
            "body": ["hello world from test"],
            "source_file": ["f.json"],
        })
        result = ingestor._apply_aliases(df)
        # Column names should remain the same
        assert "msg_id" in result.columns
        assert "id" not in result.columns

    def test_apply_aliases_adds_missing_columns_as_null(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({"body": ["hello world long enough"], "_source_file": ["f.json"]})
        result = ingestor._apply_aliases(df)
        for col in BaseIngestor.CANONICAL_COLUMNS:
            assert col in result.columns


# ===========================================================================
# ContiIngestor end-to-end
# ===========================================================================

class TestContiIngestor:
    def test_load_json(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = ingestor.load_raw()
        # 6 records in fixture (including 1 duplicate and 1 short message)
        assert len(df) == 6

    def test_normalize_produces_canonical_columns(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        raw = ingestor.load_raw()
        normalized = ingestor.normalize(raw)
        for col in BaseIngestor.CANONICAL_COLUMNS:
            assert col in normalized.columns

    def test_run_produces_parquet(self, conti_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        result = ingestor.run(output_dir=processed_dir)
        assert result.records_raw == 6
        # After dedup (removes 1) and short message filter (removes 1): expect 4
        assert result.records_written == 4
        assert (processed_dir / "conti_intel.parquet").exists()

    def test_run_parquet_has_correct_columns(self, conti_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        ingestor.run(output_dir=processed_dir)
        df = pd.read_parquet(processed_dir / "conti_intel.parquet")
        assert list(df.columns) == PARQUET_COLUMNS

    def test_run_all_records_have_dataset_field(self, conti_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        ingestor.run(output_dir=processed_dir)
        df = pd.read_parquet(processed_dir / "conti_intel.parquet")
        assert (df["dataset"] == "conti").all()

    def test_run_missing_dir_returns_empty_result(self, ingestion_cfg, tmp_path):
        from tests.conftest import _make_corpus_cfg
        corpus = _make_corpus_cfg("conti", tmp_path / "nonexistent")
        ingestor = ContiIngestor(corpus, ingestion_cfg)
        result = ingestor.run(output_dir=tmp_path / "out")
        assert result.records_raw == 0
        assert result.records_written == 0

    def test_dry_run_does_not_write_file(self, conti_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        result = ingestor.run(output_dir=processed_dir, dry_run=True)
        assert result.records_written > 0
        assert not (processed_dir / "conti_intel.parquet").exists()


# ===========================================================================
# BabukIngestor
# ===========================================================================

class TestBabukIngestor:
    def test_load_csv_and_txt(self, babuk_corpus_cfg, ingestion_cfg):
        ingestor = BabukIngestor(babuk_corpus_cfg, ingestion_cfg)
        df = ingestor.load_raw()
        # 4 CSV rows + 3 TXT lines (2 ndjson + 1 plain text)
        assert len(df) >= 5

    def test_run_produces_parquet(self, babuk_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = BabukIngestor(babuk_corpus_cfg, ingestion_cfg)
        result = ingestor.run(output_dir=processed_dir)
        assert result.records_raw > 0
        assert result.records_written > 0
        assert (processed_dir / "babuk_intel.parquet").exists()

    def test_run_parquet_dataset_field(self, babuk_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = BabukIngestor(babuk_corpus_cfg, ingestion_cfg)
        ingestor.run(output_dir=processed_dir)
        df = pd.read_parquet(processed_dir / "babuk_intel.parquet")
        assert (df["dataset"] == "babuk").all()


# ===========================================================================
# BlackBastaIngestor
# ===========================================================================

class TestBlackBastaIngestor:
    def test_load_matrix_json(self, blackbasta_corpus_cfg, ingestion_cfg):
        ingestor = BlackBastaIngestor(blackbasta_corpus_cfg, ingestion_cfg)
        df = ingestor.load_raw()
        # All 4 events loaded (filtering happens in normalize)
        assert len(df) == 4

    def test_normalize_filters_non_message_events(self, blackbasta_corpus_cfg, ingestion_cfg):
        ingestor = BlackBastaIngestor(blackbasta_corpus_cfg, ingestion_cfg)
        raw = ingestor.load_raw()
        normalized = ingestor.normalize(raw)
        # m.room.member event should be filtered out
        assert len(normalized) == 3

    def test_normalize_converts_ms_timestamp(self, blackbasta_corpus_cfg, ingestion_cfg):
        ingestor = BlackBastaIngestor(blackbasta_corpus_cfg, ingestion_cfg)
        raw = ingestor.load_raw()
        normalized = ingestor.normalize(raw)
        ts_col = normalized["timestamp"]
        # Timestamps should be datetime, not large integers
        assert pd.api.types.is_datetime64_any_dtype(ts_col)
        # Spot check: 1709244000000 ms = 2024-03-01 ~00:00:00 UTC
        first_ts = pd.Timestamp(ts_col.dropna().iloc[0])
        assert first_ts.year == 2024

    def test_normalize_extracts_body_from_content(self, blackbasta_corpus_cfg, ingestion_cfg):
        ingestor = BlackBastaIngestor(blackbasta_corpus_cfg, ingestion_cfg)
        raw = ingestor.load_raw()
        normalized = ingestor.normalize(raw)
        bodies = normalized["body"].dropna().tolist()
        assert any("payment" in str(b).lower() for b in bodies)

    def test_run_produces_parquet(self, blackbasta_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = BlackBastaIngestor(blackbasta_corpus_cfg, ingestion_cfg)
        result = ingestor.run(output_dir=processed_dir)
        assert result.records_written > 0
        assert (processed_dir / "black_basta_intel.parquet").exists()

    def test_room_id_in_metadata(self, blackbasta_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = BlackBastaIngestor(blackbasta_corpus_cfg, ingestion_cfg)
        ingestor.run(output_dir=processed_dir)
        df = pd.read_parquet(processed_dir / "black_basta_intel.parquet")
        df = parse_metadata_column(df)
        # room_id should be in metadata via recipient alias
        has_recipient = df["metadata"].apply(lambda m: "recipient" in m if isinstance(m, dict) else False)
        assert has_recipient.any()


# ===========================================================================
# LockBitIngestor
# ===========================================================================

class TestLockBitIngestor:
    def test_load_json(self, lockbit_corpus_cfg, ingestion_cfg):
        ingestor = LockBitIngestor(lockbit_corpus_cfg, ingestion_cfg)
        df = ingestor.load_raw()
        assert len(df) == 4

    def test_normalize_maps_message_to_body(self, lockbit_corpus_cfg, ingestion_cfg):
        ingestor = LockBitIngestor(lockbit_corpus_cfg, ingestion_cfg)
        raw = ingestor.load_raw()
        normalized = ingestor.normalize(raw)
        assert "body" in normalized.columns
        assert normalized["body"].notna().any()

    def test_run_produces_parquet(self, lockbit_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = LockBitIngestor(lockbit_corpus_cfg, ingestion_cfg)
        result = ingestor.run(output_dir=processed_dir)
        assert result.records_written == 4
        assert (processed_dir / "lockbit_intel.parquet").exists()

    def test_run_parquet_dataset_field(self, lockbit_corpus_cfg, ingestion_cfg, processed_dir):
        ingestor = LockBitIngestor(lockbit_corpus_cfg, ingestion_cfg)
        ingestor.run(output_dir=processed_dir)
        df = pd.read_parquet(processed_dir / "lockbit_intel.parquet")
        assert (df["dataset"] == "lockbit").all()


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistry:
    def test_registered_names_contains_all_corpora(self):
        from taig.ingestion.registry import registered_names
        names = registered_names()
        assert "conti" in names
        assert "babuk" in names
        assert "black_basta" in names
        assert "lockbit" in names

    def test_get_ingestor_unknown_corpus_raises(self):
        from taig.ingestion.registry import get_ingestor
        with pytest.raises(ValueError, match="Unknown corpus"):
            get_ingestor("nonexistent_corpus_xyz")

    def test_list_corpora_returns_all_registered(self):
        from taig.ingestion.registry import list_corpora
        corpora = list_corpora()
        names = [c[0] for c in corpora]
        assert "conti" in names
        assert "babuk" in names


# ===========================================================================
# Language detection
# ===========================================================================

class TestLanguageDetection:
    def test_detect_languages_adds_column(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "body": ["hello this is a message in english", "готов к работе"],
            "msg_id": ["1", "2"],
        })
        result = ingestor.detect_languages(df)
        assert "language" in result.columns
        assert result["language"].notna().all()

    def test_detect_languages_handles_empty(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        result = ingestor.detect_languages(pd.DataFrame())
        assert result.empty

    def test_detect_languages_preserves_existing(self, conti_corpus_cfg, ingestion_cfg):
        ingestor = ContiIngestor(conti_corpus_cfg, ingestion_cfg)
        df = pd.DataFrame({
            "body": ["hello world long message"],
            "msg_id": ["1"],
            "language": ["ru"],  # pre-set
        })
        result = ingestor.detect_languages(df)
        # Should not overwrite existing non-unknown values
        assert result["language"].iloc[0] == "ru"


# ===========================================================================
# CLI tests
# ===========================================================================

class TestCLI:
    def test_list_command_exits_zero(self, capsys):
        from taig.ingestion.__main__ import main
        exit_code = main(["--list"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "conti" in captured.out.lower()

    def test_no_corpus_exits_nonzero(self):
        from taig.ingestion.__main__ import main
        exit_code = main([])
        assert exit_code != 0

    def test_unknown_corpus_exits_nonzero(self):
        from taig.ingestion.__main__ import main
        exit_code = main(["--corpus", "xyz_unknown_corpus"])
        assert exit_code != 0

    def test_dry_run_flag_parsed(self, conti_corpus_cfg, ingestion_cfg, tmp_path, monkeypatch):
        """dry_run should complete without writing a file."""
        import taig.ingestion.registry as reg

        # Monkeypatch get_ingestor to use our test corpus config
        original = reg.get_ingestor
        def _mock_get(name, config_dir=None):
            if name == "conti":
                return ContiIngestor(conti_corpus_cfg, ingestion_cfg)
            return original(name, config_dir)

        monkeypatch.setattr(reg, "get_ingestor", _mock_get)

        from taig.ingestion.__main__ import main
        out_dir = tmp_path / "processed"
        exit_code = main([
            "--corpus", "conti",
            "--output", str(out_dir),
            "--dry-run",
        ])
        assert exit_code == 0
        assert not (out_dir / "conti_intel.parquet").exists()
