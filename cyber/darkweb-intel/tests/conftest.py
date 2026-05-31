"""
conftest.py
-----------
Shared pytest fixtures for TAIG tests.

All fixtures use tmp_path (pytest built-in) so tests are self-contained
and leave no artifacts. No real dataset files are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taig.utils.config import (
    CorpusConfig,
    IngestionConfig,
)


# ---------------------------------------------------------------------------
# Synthetic raw data builders
# ---------------------------------------------------------------------------

CONTI_RECORDS = [
    {"id": "1", "date": "2022-02-14 10:00:00", "from": "stern", "to": "johny",
     "text": "готов к работе сегодня утром"},
    {"id": "2", "date": "2022-02-14 10:01:00", "from": "johny", "to": "stern",
     "text": "good morning, ready to start the deployment"},
    {"id": "3", "date": "2022-02-14 10:02:00", "from": "stern", "to": "johny",
     "text": "отправь мне список жертв за последнюю неделю"},
    {"id": "4", "date": "2022-02-14 10:03:00", "from": "mango", "to": "stern",
     "text": "the locker is ready, tested on Windows Server 2019"},
    # Duplicate — should be removed by dedup
    {"id": "2", "date": "2022-02-14 10:01:00", "from": "johny", "to": "stern",
     "text": "good morning, ready to start the deployment"},
    # Too short — should be filtered
    {"id": "5", "date": "2022-02-14 10:04:00", "from": "stern", "to": "johny",
     "text": "ok"},
]

BABUK_CSV_ROWS = [
    "id,date,from,to,text",
    "101,2021-04-01 09:00:00,babuk_dev,partner1,preparing the payload for the new target",
    "102,2021-04-01 09:05:00,partner1,babuk_dev,confirmed receipt and testing environment",
    "103,2021-04-01 09:10:00,babuk_dev,partner1,use the standard ESXi locker for VMware",
    "104,2021-04-02 08:00:00,admin,babuk_dev,payment received confirm on chain",
]

BABUK_NDJSON_LINES = [
    '{"msg_id": "201", "timestamp": "2021-04-03 10:00:00", "sender": "babuk_dev", "recipient": "partner2", "body": "new target identified in healthcare sector"}',
    '{"msg_id": "202", "timestamp": "2021-04-03 10:05:00", "sender": "partner2", "recipient": "babuk_dev", "body": "encrypting now will report back"}',
    "this is a plain text line that should be treated as a body",
]

BLACK_BASTA_EVENTS = [
    {
        "event_id": "$evt001:matrix.bb",
        "type": "m.room.message",
        "origin_server_ts": 1709244000000,
        "sender": "@bb_admin:matrix.bb",
        "room_id": "!ops_room:matrix.bb",
        "content": {"msgtype": "m.text", "body": "payment received from target alpha"},
    },
    {
        "event_id": "$evt002:matrix.bb",
        "type": "m.room.message",
        "origin_server_ts": 1709244060000,
        "sender": "@bb_ops:matrix.bb",
        "room_id": "!ops_room:matrix.bb",
        "content": {"msgtype": "m.text", "body": "decryptor sent to victim confirmed delivery"},
    },
    # Non-message event — should be filtered by type
    {
        "event_id": "$evt003:matrix.bb",
        "type": "m.room.member",
        "origin_server_ts": 1709244120000,
        "sender": "@bb_admin:matrix.bb",
        "room_id": "!ops_room:matrix.bb",
        "content": {"membership": "join"},
    },
    {
        "event_id": "$evt004:matrix.bb",
        "type": "m.room.message",
        "origin_server_ts": 1709244180000,
        "sender": "@bb_dev:matrix.bb",
        "room_id": "!dev_room:matrix.bb",
        "content": {"msgtype": "m.text", "body": "CitrixBleed exploit integrated into the toolkit"},
    },
]

LOCKBIT_RECORDS = [
    {"id": "T001", "created_at": "2023-12-01 00:00:00", "sender": "lockbit_admin",
     "to": "affiliate_01", "message": "payment confirmed for target bravo"},
    {"id": "T002", "created_at": "2023-12-02 00:00:00", "sender": "affiliate_01",
     "to": "lockbit_admin", "message": "new target identified requesting access"},
    {"id": "T003", "created_at": "2023-12-03 00:00:00", "sender": "lockbit_admin",
     "to": "affiliate_02", "message": "ransom increased deadline extended by 48 hours"},
    {"id": "T004", "created_at": "2023-12-04 00:00:00", "sender": "affiliate_02",
     "to": "lockbit_admin", "message": "victim negotiating reduce demand by 30 percent"},
]


# ---------------------------------------------------------------------------
# Fixtures: synthetic raw data directories
# ---------------------------------------------------------------------------

@pytest.fixture()
def conti_raw_dir(tmp_path: Path) -> Path:
    """Temp directory containing a synthetic Conti JSON file."""
    d = tmp_path / "raw" / "conti"
    d.mkdir(parents=True)
    (d / "conti_leak_sample.json").write_text(
        json.dumps(CONTI_RECORDS, ensure_ascii=False), encoding="utf-8"
    )
    return d


@pytest.fixture()
def babuk_raw_dir(tmp_path: Path) -> Path:
    """Temp directory containing synthetic Babuk CSV and ndjson .txt files."""
    d = tmp_path / "raw" / "babuk"
    d.mkdir(parents=True)
    (d / "babuk_comms.csv").write_text(
        "\n".join(BABUK_CSV_ROWS), encoding="utf-8"
    )
    (d / "babuk_extra.txt").write_text(
        "\n".join(BABUK_NDJSON_LINES), encoding="utf-8"
    )
    return d


@pytest.fixture()
def blackbasta_raw_dir(tmp_path: Path) -> Path:
    """Temp directory containing synthetic Black Basta Matrix JSON export."""
    d = tmp_path / "raw" / "black_basta"
    d.mkdir(parents=True)
    (d / "bb_matrix_export.json").write_text(
        json.dumps(BLACK_BASTA_EVENTS, ensure_ascii=False), encoding="utf-8"
    )
    return d


@pytest.fixture()
def lockbit_raw_dir(tmp_path: Path) -> Path:
    """Temp directory containing synthetic LockBit JSON records."""
    d = tmp_path / "raw" / "lockbit"
    d.mkdir(parents=True)
    (d / "lockbit_comms.json").write_text(
        json.dumps(LOCKBIT_RECORDS, ensure_ascii=False), encoding="utf-8"
    )
    return d


@pytest.fixture()
def processed_dir(tmp_path: Path) -> Path:
    """Temp output directory for Parquet files."""
    d = tmp_path / "processed"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# Fixtures: config objects
# ---------------------------------------------------------------------------

def _make_ingestion_cfg(**overrides) -> IngestionConfig:
    defaults = dict(min_body_length=5, dedup_column="msg_id")
    defaults.update(overrides)
    return IngestionConfig(**defaults)


def _make_corpus_cfg(name: str, raw_dir: Path, **overrides) -> CorpusConfig:
    """Build a minimal CorpusConfig for a given corpus and raw directory."""
    alias_maps = {
        "conti": {
            "msg_id": ["id", "message_id", "msg_id", "ID"],
            "timestamp": ["date", "ts", "timestamp", "time", "Date"],
            "sender": ["from", "fromId", "from_id", "sender", "From", "author"],
            "recipient": ["to", "toId", "to_id", "recipient", "To"],
            "body": ["text", "body", "message", "content", "msg", "Text", "Message"],
        },
        "babuk": {
            "msg_id": ["id", "message_id"],
            "timestamp": ["date", "timestamp", "time"],
            "sender": ["from", "sender", "author"],
            "recipient": ["to", "recipient"],
            "body": ["text", "body", "message", "content"],
        },
        "black_basta": {
            "msg_id": ["event_id", "id", "message_id"],
            "timestamp": ["origin_server_ts", "timestamp", "date"],
            "sender": ["sender", "from", "user_id"],
            "recipient": ["room_id", "to", "recipient"],
            "body": ["body", "content", "message", "text"],
        },
        "lockbit": {
            "msg_id": ["id", "message_id", "ticket_id"],
            "timestamp": ["created_at", "timestamp", "date"],
            "sender": ["sender", "from", "username", "user"],
            "recipient": ["to", "recipient", "room"],
            "body": ["body", "message", "text", "content", "description"],
        },
    }

    defaults = dict(
        name=name,
        enabled=True,
        display_name=name.replace("_", " ").title(),
        description=f"Synthetic {name} corpus for testing",
        raw_dir=raw_dir,
        formats_supported=["json", "csv", "txt"],
        column_aliases=alias_maps.get(name, {}),
        languages=["en", "ru"],
        date_range=None,
        source="synthetic",
        notes="Test fixture",
    )
    defaults.update(overrides)
    return CorpusConfig(**defaults)


@pytest.fixture()
def conti_corpus_cfg(conti_raw_dir: Path) -> CorpusConfig:
    return _make_corpus_cfg("conti", conti_raw_dir)


@pytest.fixture()
def babuk_corpus_cfg(babuk_raw_dir: Path) -> CorpusConfig:
    return _make_corpus_cfg("babuk", babuk_raw_dir)


@pytest.fixture()
def blackbasta_corpus_cfg(blackbasta_raw_dir: Path) -> CorpusConfig:
    return _make_corpus_cfg("black_basta", blackbasta_raw_dir)


@pytest.fixture()
def lockbit_corpus_cfg(lockbit_raw_dir: Path) -> CorpusConfig:
    return _make_corpus_cfg("lockbit", lockbit_raw_dir)


@pytest.fixture()
def ingestion_cfg() -> IngestionConfig:
    return _make_ingestion_cfg()
