# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Top-level ``mosaic`` command.

Dispatches to the per-stage subcommands. Each stage module exposes a
``configure_parser(parser)`` / ``run(args)`` pair; this module wires them
into an ``argparse`` subparser tree and invokes the selected handler.
"""

import argparse

from . import (
    aggregate,
    clean_db,
    index_db,
    ingest,
    init_db,
    maintain,
    register,
    remove,
)

# (name, module, help) for each subcommand. The module supplies
# configure_parser() and run().
_SUBCOMMANDS = [
    ("init", init_db, "Create PostgreSQL + Iceberg tables (run once)"),
    ("ingest", ingest, "Ingest one or more gVCF files"),
    (
        "register",
        register,
        "Register deferred ingest batches' files with Iceberg",
    ),
    (
        "aggregate",
        aggregate,
        "Reduce per-batch partials into the PostgreSQL aggregates",
    ),
    (
        "remove",
        remove,
        "Remove samples (a negative batch for the next aggregate)",
    ),
    ("index", index_db, "Create PostgreSQL indexes (after ingestion)"),
    (
        "maintain",
        maintain,
        "Iceberg maintenance: expire snapshots, reclaim their files",
    ),
    ("clean", clean_db, "Reset PostgreSQL + Iceberg state"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaic",
        description=(
            "MOSAIC — scalable gVCF ingestion into PostgreSQL + Apache "
            "Iceberg. Typical flow: init -> ingest -> register -> aggregate "
            "-> index + maintain (register only after `ingest "
            "--defer-register`); to remove samples: remove -> aggregate -> "
            "index + maintain."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, module, help_text in _SUBCOMMANDS:
        subparser = sub.add_parser(name, help=help_text, description=help_text)
        module.configure_parser(subparser)
        subparser.set_defaults(_run=module.run)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args._run(args)


if __name__ == "__main__":
    main()
