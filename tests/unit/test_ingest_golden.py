# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the ingest signatures of the test harness:
``jc_testlib.canonical`` /
``canonical_digest`` (floats as bits, tuple keys, bytes), and
``assert_ingest_signature`` (a case never recorded fails; record mode
needs a reason, records with provenance and a local copy, leaves an
unchanged entry as it is; a mismatch shows the first differences against
the local copy, or a ``compare`` callback's)."""

import json
import math

import pytest

import jc_testlib as L

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "INGEST_GOLDEN_DIGESTS", str(tmp_path / "g.json"))
    monkeypatch.setattr(L, "INGEST_GOLDEN_DIR", str(tmp_path / "copies"))
    monkeypatch.setattr(L, "_INGEST_GOLDEN_RECORDED", set())
    monkeypatch.delenv(L.INGEST_GOLDEN_RECORD_ENV, raising=False)
    monkeypatch.delenv(L.INGEST_GOLDEN_REASON_ENV, raising=False)
    return tmp_path


def _record(monkeypatch, case, detail, counts, reason="why", **kw):
    monkeypatch.setattr(L, "_INGEST_GOLDEN_RECORDED", set())
    monkeypatch.setenv(L.INGEST_GOLDEN_RECORD_ENV, "1")
    if reason is None:
        monkeypatch.delenv(L.INGEST_GOLDEN_REASON_ENV, raising=False)
    else:
        monkeypatch.setenv(L.INGEST_GOLDEN_REASON_ENV, reason)
    try:
        return L.assert_ingest_signature(case, detail, counts, **kw)
    finally:
        monkeypatch.delenv(L.INGEST_GOLDEN_RECORD_ENV)
        monkeypatch.setattr(L, "_INGEST_GOLDEN_RECORDED", set())


DETAIL = {
    "result": {"sample_qc": {(1, 0): [1, 2]}, "gtc": [0.1, -0.0]},
    "files": {"a.parquet": {"columns": {"x": "abc"}, "rows": [3, 4]}},
}


def test_canonical_floats_by_bits_and_keys():
    assert L.canonical(0.0) != L.canonical(-0.0)
    assert L.canonical(math.nan) == L.canonical(math.nan)
    assert L.canonical(0.1) != L.canonical(0.1 + 2**-56)
    assert L.canonical({(1, 0): b"\x01"}) == {"[1, 0]": "bytes:01"}
    assert L.canonical((1, [2.0])) == [1, ["f64:0000000000000040"]]
    # Key order does not matter, values do.
    assert L.canonical_digest({"a": 1, "b": 2}) == L.canonical_digest(
        {"b": 2, "a": 1}
    )
    assert L.canonical_digest([1, 2]) != L.canonical_digest([2, 1])


def test_unrecorded_case_fails(store):
    with pytest.raises(AssertionError, match="no digest recorded"):
        L.assert_ingest_signature("never", DETAIL, {"rows": 1})


def test_record_then_check(store, monkeypatch, tmp_path):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    d = _record(
        monkeypatch,
        "c1",
        DETAIL,
        {"rows": 7},
        files={"a.parquet": str(f)},
        reference="both workers",
    )
    data = json.loads((store / "g.json").read_text())
    entry = data["scenarios"]["c1"]
    assert data["format"] == L.INGEST_GOLDEN_FORMAT
    assert entry["sha256"] == d == L.canonical_digest(DETAIL)
    assert entry["counts"] == {"rows": 7}
    rec = entry["recorded"]
    assert rec["reference"] == "both workers; why"
    assert {"commit", "tracked_changes", "platform", "date"} <= set(rec)
    copy = L.load_ingest_copy("c1")
    assert copy is not None and copy[0] == L.canonical(DETAIL)
    assert (store / "copies" / "c1" / "a.parquet").read_bytes() == b"x"
    assert L.assert_ingest_signature("c1", DETAIL, {"rows": 7}) == d
    with pytest.raises(AssertionError, match="differs from the recorded"):
        L.assert_ingest_signature("c1", DETAIL, {"rows": 8})


def test_record_mode_without_a_reason_is_refused(store, monkeypatch):
    for reason in (None, "", "  "):
        with pytest.raises(AssertionError, match=L.INGEST_GOLDEN_REASON_ENV):
            _record(monkeypatch, "c1", DETAIL, {}, reason=reason)
    assert not (store / "g.json").exists()


def test_unchanged_entry_left_as_it_is(store, monkeypatch):
    _record(monkeypatch, "c1", DETAIL, {"rows": 1}, reason="first")
    before = (store / "g.json").read_bytes()
    _record(monkeypatch, "c1", DETAIL, {"rows": 1}, reason="second")
    assert (store / "g.json").read_bytes() == before


def test_case_slash_names_and_other_cases_untouched(store, monkeypatch):
    _record(monkeypatch, "row/a", DETAIL, {}, reason="first")
    _record(monkeypatch, "row/b", DETAIL, {}, reason="first")
    b = json.loads((store / "g.json").read_text())["scenarios"]["row/b"]
    other = {**DETAIL, "extra": 1}
    _record(monkeypatch, "row/a", other, {}, reason="K9: changes a")
    data = json.loads((store / "g.json").read_text())["scenarios"]
    assert data["row/a"]["sha256"] == L.canonical_digest(other)
    assert data["row/b"] == b
    assert (store / "copies" / "row__a" / "detail.json.gz").exists()


def test_mismatch_shows_the_first_differences(store, monkeypatch):
    _record(monkeypatch, "c1", DETAIL, {})
    changed = json.loads(json.dumps(L.canonical(DETAIL)))
    changed["files"]["a.parquet"]["rows"] = [3, 5]
    with pytest.raises(AssertionError) as e:
        L.assert_ingest_signature("c1", changed, {})
    msg = str(e.value)
    assert "/files/a.parquet/rows[1]: golden 4 != output 5" in msg
    # A compare callback replaces the generic differences.
    with pytest.raises(AssertionError, match="row 9 differs"):
        L.assert_ingest_signature(
            "c1", changed, {}, compare=lambda g, d: ["  row 9 differs"]
        )
    # Without the local copy: the digests only.
    import shutil

    shutil.rmtree(store / "copies")
    with pytest.raises(AssertionError, match="no local copy"):
        L.assert_ingest_signature("c1", changed, {})


def test_empty_compare_falls_back_to_the_details(store, monkeypatch):
    """A compare callback that finds nothing (the difference is in the
    result dict or a count, not in a file it reads): the canonical details
    side by side are reported."""
    _record(monkeypatch, "c1", DETAIL, {})
    changed = json.loads(json.dumps(L.canonical(DETAIL)))
    changed["result"]["sample_qc"]["[1, 0]"] = [1, 3]
    with pytest.raises(AssertionError) as e:
        L.assert_ingest_signature("c1", changed, {}, compare=lambda g, d: [])
    assert "/result/sample_qc/[1, 0][1]: golden 2 != output 3" in str(e.value)
