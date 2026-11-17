# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for ref-block banding's spec and option (`mosaic ingest
--ref-gq-bands`).

* RefBandSpec: parse / normalize, and the malformed specs.
* The CLI: --ref-gq-bands validation and normalization.

The bander's tests this file held earlier (RefBandSpec.band(), RefBlockBander on
crafted streams, collect_sample_ref_block_data with and without a bander)
tested the Python ingest worker; they are ``cargo test``s of the Rust
worker's ``banding.rs`` / ``ref_blocks.rs`` with the same inputs and
expected values, and the random-stream test runs the worker binary
(``tests/unit/test_worker_bin_ref_blocks.py``).
"""

import argparse

import pytest

from mosaic.ingest import _ref_gq_bands_text, configure_parser
from mosaic.ref_banding import RefBandSpec

pytestmark = pytest.mark.unit


# ── RefBandSpec ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,bounds,norm",
    [
        ("20,60", (20, 60), "20,60"),
        ("all", (), "all"),
        ("20", (20,), "20"),
        (" 20 , 60 ", (20, 60), "20,60"),
        ("1,99", (1, 99), "1,99"),
        ("010,20", (10, 20), "10,20"),
        ("  all ", (), "all"),
        ("10,20,30,40,50,60,70,80,90", tuple(range(10, 100, 10)), None),
    ],
)
def test_spec_parse_and_normalize(text, bounds, norm):
    spec = RefBandSpec.parse(text)
    assert spec.bounds == bounds
    if norm is not None:
        assert spec.text == str(spec) == norm
    assert RefBandSpec.parse(spec.text) == spec  # normalized round trip


@pytest.mark.parametrize(
    "text,needle",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("60,20", "ascending"),
        ("20,20", "ascending"),
        ("0", "outside"),
        ("100", "outside"),
        ("20,x", "not an integer"),
        ("20.5", "not an integer"),
        ("20,", "not an integer"),
        ("-5", "not an integer"),
        ("+20", "not an integer"),
        ("ALL", "not an integer"),
        ("all,20", "not an integer"),
    ],
)
def test_spec_errors(text, needle):
    with pytest.raises(ValueError, match=needle):
        RefBandSpec.parse(text)


def test_spec_constructor_validates():
    with pytest.raises(ValueError):
        RefBandSpec((60, 20))
    with pytest.raises(ValueError):
        RefBandSpec((0,))


# ── the CLI ─────────────────────────────────────────────────────────────────


def _parse(argv):
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    return parser.parse_args(argv)


def test_cli_default_is_off():
    args = _parse(["a.gvcf.gz"])
    assert args.ref_gq_bands is None
    assert _ref_gq_bands_text(args) is None


@pytest.mark.parametrize(
    "value,norm", [("20,60", "20,60"), (" 20, 060 ", "20,60"), ("all", "all")]
)
def test_cli_normalizes(value, norm):
    args = _parse(["a.gvcf.gz", "--ref-gq-bands", value])
    assert isinstance(args.ref_gq_bands, RefBandSpec)
    assert _ref_gq_bands_text(args) == norm


@pytest.mark.parametrize("value", ["", "60,20", "20,20", "0", "100", "x"])
def test_cli_rejects(value, capsys):
    with pytest.raises(SystemExit) as e:
        _parse(["a.gvcf.gz", "--ref-gq-bands", value])
    assert e.value.code == 2
    assert "--ref-gq-bands" in capsys.readouterr().err


def test_text_of_a_hand_built_namespace():
    assert _ref_gq_bands_text(argparse.Namespace(ref_gq_bands="20 ,60")) == (
        "20,60"
    )
