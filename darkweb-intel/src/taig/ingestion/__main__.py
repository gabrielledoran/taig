"""
__main__.py
-----------
CLI entry point for the TAIG ingestion layer.

Usage
-----
    # List all registered corpora and their status
    python -m taig.ingestion --list

    # Ingest a single corpus (must have data in data/raw/<corpus>/)
    python -m taig.ingestion --corpus conti

    # Ingest to a custom output directory
    python -m taig.ingestion --corpus conti --output /tmp/taig_out/

    # Ingest all enabled corpora
    python -m taig.ingestion --corpus all

    # Dry run — validate and count records, skip writing
    python -m taig.ingestion --corpus conti --dry-run

    # Use a custom config directory (useful in tests / CI)
    python -m taig.ingestion --corpus conti --config-dir /path/to/config/

    # Verbose logging
    python -m taig.ingestion --corpus conti -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from taig.ingestion.registry import (
    get_ingestor,
    list_corpora,
    registered_names,
    run_all_enabled,
)
from taig.utils.logging import get_logger, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m taig.ingestion",
        description="TAIG ingestion CLI — load and normalize threat intelligence corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m taig.ingestion --list
  python -m taig.ingestion --corpus conti
  python -m taig.ingestion --corpus all --dry-run
  python -m taig.ingestion --corpus black_basta --output data/processed/
        """,
    )

    parser.add_argument(
        "--corpus",
        metavar="NAME",
        help=(
            "Corpus to ingest. Use 'all' to run every enabled corpus. "
            f"Registered: {', '.join(sorted(registered_names()))}"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Output directory for Parquet files. Defaults to data/processed/.",
    )
    parser.add_argument(
        "--config-dir",
        metavar="DIR",
        default=None,
        help="Override the config/ directory path.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered corpora and their enabled status, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but skip writing output files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _cmd_list(config_dir: str | None) -> None:
    corpora = list_corpora(config_dir=config_dir)
    print(f"\n{'CORPUS':<20} {'ENABLED':<10} DISPLAY NAME")
    print("-" * 60)
    for name, enabled, display in corpora:
        status = "yes" if enabled else "no"
        print(f"{name:<20} {status:<10} {display}")
    print()


def _cmd_ingest(
    corpus: str,
    output: str | None,
    config_dir: str | None,
    dry_run: bool,
) -> int:
    """Run ingestion for one corpus or all enabled corpora.

    Returns 0 on success, 1 on any error.
    """
    log = get_logger("taig.ingestion.cli")
    output_dir = Path(output) if output else None

    if corpus == "all":
        results = run_all_enabled(
            output_dir=output_dir,
            config_dir=config_dir,
            dry_run=dry_run,
        )
        if not results:
            log.error(
                "No enabled corpora to ingest. "
                "Set enabled: true in config/datasets.yaml."
            )
            return 1
        _print_summary(results)
        return 0

    try:
        ingestor = get_ingestor(corpus, config_dir=config_dir)
    except (ValueError, KeyError) as exc:
        log.error("%s", exc)
        return 1

    result = ingestor.run(output_dir=output_dir, dry_run=dry_run)
    _print_summary({corpus: result})

    return 0 if result.records_written >= 0 else 1


def _print_summary(results: dict) -> None:
    from taig.ingestion.schemas import IngestorResult

    print("\n=== Ingestion Summary ===")
    for corpus, result in results.items():
        if not isinstance(result, IngestorResult):
            continue
        print(f"\n  Corpus     : {result.corpus}")
        print(f"  Raw records: {result.records_raw}")
        print(f"  After clean: {result.records_after_clean}")
        print(f"  Written    : {result.records_written}")
        print(f"  Drop rate  : {result.drop_rate:.1%}")
        if result.output_path:
            print(f"  Output     : {result.output_path}")
        if result.language_distribution:
            print(f"  Languages  : {result.language_distribution}")
        if result.errors:
            print(f"  Errors     : {len(result.errors)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    if args.list:
        _cmd_list(args.config_dir)
        return 0

    if not args.corpus:
        parser.print_help()
        print("\nerror: --corpus is required (or use --list)\n")
        return 1

    return _cmd_ingest(
        corpus=args.corpus,
        output=args.output,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
