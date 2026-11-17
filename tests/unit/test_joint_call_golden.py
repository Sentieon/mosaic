# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the joint-call goldens of the test harness:
``jc_testlib.vcf_text_digest``,
``assert_golden`` (a match, a mismatch with the first differing lines, a
scenario never recorded) and its record mode (format 2: per-scenario
provenance; the reason required; an unchanged entry left as it is; a
changed digest restamped with the reason; a change of the closing lines
alone recorded beside the old provenance; a scenario's later outputs in the
same process checked against the first)."""

import gzip
import hashlib
import json

import pytest

import jc_testlib as L

pytestmark = pytest.mark.unit

HEADER = (
    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)
RECORDS = "chr21\t1\t.\tA\tC\t.\t.\tAN=0\nchr21\t2\t.\tG\tT\t.\t.\tAN=0\n"
STDERR = (
    "noise\nCompleted! Total processed: 2 variants\n"
    "Catalog cross-check: the catalog's non-ref allele counts "
    "(alleles.count) equal the emitted AC at every record (2 records)\n"
)


@pytest.fixture
def golden(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "GOLDEN_DIGESTS", str(tmp_path / "g.json"))
    monkeypatch.setattr(L, "GOLDEN_VCF_DIR", str(tmp_path / "vcfs"))
    monkeypatch.setattr(L, "_GOLDEN_RECORDED", set())
    monkeypatch.delenv(L.GOLDEN_RECORD_ENV, raising=False)
    monkeypatch.delenv(L.GOLDEN_REASON_ENV, raising=False)
    return tmp_path


def _vcf(path, text, compress=True):
    if compress:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        with open(path, "w") as fh:
            fh.write(text)
    return str(path)


def _record(monkeypatch, scenario, path, stderr=None, reason="why", **kw):
    """One record-mode call, as a fresh process would make it."""
    monkeypatch.setattr(L, "_GOLDEN_RECORDED", set())
    monkeypatch.setenv(L.GOLDEN_RECORD_ENV, "1")
    if reason is None:
        monkeypatch.delenv(L.GOLDEN_REASON_ENV, raising=False)
    else:
        monkeypatch.setenv(L.GOLDEN_REASON_ENV, reason)
    try:
        return L.assert_golden(scenario, path, stderr, **kw)
    finally:
        monkeypatch.delenv(L.GOLDEN_RECORD_ENV)
        monkeypatch.setattr(L, "_GOLDEN_RECORDED", set())


def _json():
    with open(L.GOLDEN_DIGESTS) as fh:
        return json.load(fh)


def test_digest_of_the_decompressed_text(golden):
    text = HEADER + RECORDS
    d = L.vcf_text_digest(_vcf(golden / "a.vcf.gz", text))
    assert d == {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "records": 2,
        "bytes": len(text),
    }
    # A plain file and a compressed one of the same text: the same digest.
    assert L.vcf_text_digest(_vcf(golden / "a.vcf", text, False)) == d


def test_unrecorded_scenario_fails(golden):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    with pytest.raises(AssertionError, match="no digest recorded"):
        L.assert_golden("never", path)


def test_record_then_check(golden, monkeypatch):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    d = _record(monkeypatch, "s1", path, STDERR, reference="both engines")
    data = _json()
    assert data["format"] == 2 and "recorded" not in data
    entry = data["scenarios"]["s1"]
    assert {k: entry[k] for k in d} == d
    assert entry["closing_lines"] == STDERR.splitlines()[1:]
    rec = entry["recorded"]
    assert rec["reference"] == "both engines; why"
    assert {"commit", "tracked_changes", "platform", "date"} <= set(rec)
    assert (golden / "vcfs" / "s1.vcf.gz").exists()
    # A plain-text output of the same text passes; the closing lines too.
    plain = _vcf(golden / "c.vcf", HEADER + RECORDS, False)
    assert L.assert_golden("s1", plain, STDERR) == d
    with pytest.raises(AssertionError, match="closing lines differ"):
        L.assert_golden("s1", path, STDERR.replace("(2 records)", "(3)"))


def test_later_outputs_of_a_scenario_checked_against_the_first(
    golden, monkeypatch
):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    other = _vcf(golden / "b.vcf.gz", HEADER + RECORDS.replace("C", "G"))
    monkeypatch.setenv(L.GOLDEN_RECORD_ENV, "1")
    monkeypatch.setenv(L.GOLDEN_REASON_ENV, "why")
    L.assert_golden("s1", path)
    with pytest.raises(AssertionError, match="differs from the recorded"):
        L.assert_golden("s1", other)


def test_record_mode_without_a_reason_is_refused(golden, monkeypatch):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    for reason in (None, "", "   "):
        with pytest.raises(AssertionError, match=L.GOLDEN_REASON_ENV):
            _record(monkeypatch, "s1", path, reason=reason)
    assert not (golden / "g.json").exists()
    assert not (golden / "vcfs").exists()


def test_unchanged_entry_left_as_it_is(golden, monkeypatch):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    _record(monkeypatch, "s1", path, STDERR, reason="first")
    before = (golden / "g.json").read_bytes()
    again = _vcf(golden / "b.vcf", HEADER + RECORDS, False)
    _record(monkeypatch, "s1", again, STDERR, reason="second")
    assert (golden / "g.json").read_bytes() == before
    assert _json()["scenarios"]["s1"]["recorded"]["reference"].endswith(
        "; first"
    )


def test_changed_digest_restamped_with_the_reason(golden, monkeypatch):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    _record(monkeypatch, "s1", path, STDERR, reason="first", reference="x")
    _record(monkeypatch, "s2", path, STDERR, reason="first", reference="x")
    s2 = _json()["scenarios"]["s2"]
    other = _vcf(golden / "b.vcf.gz", HEADER + RECORDS.replace("C", "G"))
    _record(monkeypatch, "s1", other, STDERR, reason="K9: changes s1")
    data = _json()
    entry = data["scenarios"]["s1"]
    assert entry["sha256"] == L.vcf_text_digest(other)["sha256"]
    assert entry["recorded"]["reference"] == (
        f"{L.RUST_ONLY_REFERENCE}; K9: changes s1"
    )
    assert "closing_lines_recorded" not in entry
    # The other scenario is not restamped.
    assert data["scenarios"]["s2"] == s2
    assert L.vcf_text_bytes(str(golden / "vcfs" / "s1.vcf.gz")) == (
        L.vcf_text_bytes(other)
    )


def test_closing_lines_only_change(golden, monkeypatch):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    _record(monkeypatch, "s1", path, STDERR, reason="first", reference="x")
    old = _json()["scenarios"]["s1"]
    new_err = STDERR.replace("(2 records)", "(two records)")
    _record(monkeypatch, "s1", path, new_err, reason="K1: label")
    entry = _json()["scenarios"]["s1"]
    assert entry["recorded"] == old["recorded"]
    assert entry["sha256"] == old["sha256"]
    assert entry["closing_lines"] == new_err.splitlines()[1:]
    clr = entry["closing_lines_recorded"]
    assert clr["reason"] == "K1: label" and {"commit", "date"} <= set(clr)
    assert L.assert_golden("s1", path, new_err)["records"] == 2


def test_mismatch_shows_the_first_differing_lines(golden, monkeypatch):
    path = _vcf(golden / "a.vcf.gz", HEADER + RECORDS)
    _record(monkeypatch, "s2", path)
    changed = RECORDS.replace("chr21\t2\t.\tG\tT", "chr21\t2\t.\tG\tA")
    other = _vcf(golden / "b.vcf.gz", HEADER + changed + "chr21\t3\n")
    with pytest.raises(AssertionError) as e:
        L.assert_golden("s2", other)
    msg = str(e.value)
    assert "line 4:" in msg and "G\tT" in msg and "G\tA" in msg
    assert "line 5:" in msg and "<end of golden>" in msg
    assert "line 3:" not in msg
    # Without the local copy: the digests only.
    (golden / "vcfs" / "s2.vcf.gz").unlink()
    with pytest.raises(AssertionError, match="no local golden copy"):
        L.assert_golden("s2", other)
