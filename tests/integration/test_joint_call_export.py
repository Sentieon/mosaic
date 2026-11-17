# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The cohort export of ``joint_call.py``: ``--cohort-state``,
``--export-cohort DIR``, ``--export-contig DIR --region CONTIG`` and
``--from-export DIR``, on the six chr21 samples as three 2-sample batches
(the stores the golden scenarios use). The module's store moves forward,
test by test, as test_rust_joint_call.py's does:

  1. the fresh cohort: the state line is stable; the export holds exactly
     its four files; ``--from-export`` writes the ``fresh_cohort_chr21``
     golden (two window and thread settings) and, with ``--region``, the
     ``slice_chr21_10M_12M`` golden, from the whole-contig export;
  2. no connection: every ``--from-export`` run is made with the database
     port pointed at a closed port, and a PostgreSQL-path call under that
     environment fails (the control);
  3. the export path equals the PostgreSQL path (the text byte for byte,
     the closing lines) for subsets, ``--samples-file``, the trim with
     ``--min-ac 1``, caps 0 / 3 / the default, a one-site region inside a
     spanning deletion and a region without a site; the same data files
     read for a subset;
  4. the export directory is untouched by those runs;
  5. refusals on real stores: a contig not in the cohort; a directory
     with ``cohort.json`` only;
  6. after ``mosaic remove HG00100`` (stale): the gate, ``--allow-stale``,
     the first export refused, a stale export giving the
     ``stale_after_remove_HG00100`` golden, a subset naming HG00100
     refused;
  7. after the aggregate that applies it: the stale export refused, a
     third export giving the ``removed_HG00100_aggregated`` golden;
  8. two contigs (the HG00099 chr21 + chr22 slice, when present): the
     whole genome from the export equals the PostgreSQL path; a directory
     holding chr22's files only serves ``--region chr22`` and refuses the
     whole genome.

The golden checks run at ``--max-alt-alleles 0`` and never record.
"""

import json
import os
import re
import shutil
from typing import Any, Dict, Optional

import pytest

import jc_testlib as L

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not L.joint_call_worker_available(),
        reason=L.RUST_JOINT_CALL_SKIP_REASON,
    ),
]

TD = os.path.join(L.REPO_ROOT, "test_data")
DOWNLOADS_CHR21 = [
    os.path.join(TD, f"downloads/{s}/{s}.chr21.final.hard-filtered.gvcf.gz")
    for s in ("HG00099", "HG00100", "HG00101", "HG00102")
]
SLICE_2CHR = os.path.join(
    TD,
    "downloads/HG00099/"
    "HG00099.chr21_1-15M.chr22_10.5-20M.final.hard-filtered.gvcf.gz",
)
SLICE = "chr21:10000000-12000000"
# The database port pointed at a port nothing listens on: any connection
# attempt fails the run.
NO_DB = {"MOSAIC_DB_PORT": "1"}
STATE_KEYS = {
    "format",
    "agg_run_id",
    "status",
    "aggregated_at",
    "completed_batches",
    "active_samples",
    "removed_samples",
    "samples_sha256",
    "stale",
}


def _listing(directory: str) -> Dict[str, Any]:
    """Every file of the directory with its size and mtime."""
    out = {}
    for name in sorted(os.listdir(directory)):
        st = os.stat(os.path.join(directory, name))
        out[name] = (st.st_size, st.st_mtime_ns)
    return out


def _from_export(directory: str, out: str, **kw):
    """A joint call from the export, with no reachable database."""
    return L.run_joint_call(
        out, from_export=directory, extra_env=NO_DB, threads=4, **kw
    )


def _state(proc) -> Dict[str, Any]:
    lines = proc.stdout.splitlines()
    assert len(lines) == 1, proc.stdout
    state = json.loads(lines[0])
    assert set(state) == STATE_KEYS
    return state


def _records(path: str):
    text = L.vcf_text_bytes(path).decode()
    return [x for x in text.split("\n") if x and not x.startswith("#")]


def _same_as_postgres(
    work: str, directory: str, tag: str, **kw
) -> Dict[str, Any]:
    """The same call from PostgreSQL and from the export: the text byte
    for byte and the closing lines equal. Returns both runs' stderr."""
    a = os.path.join(work, f"{tag}.pg.vcf.gz")
    b = os.path.join(work, f"{tag}.export.vcf.gz")
    pa = L.run_joint_call(a, threads=4, **kw)
    pb = _from_export(directory, b, **kw)
    assert L.vcf_text_bytes(a) == L.vcf_text_bytes(b), tag
    assert L.closing_lines(pa.stderr) == L.closing_lines(pb.stderr), tag
    assert "from the export" in pb.stderr
    return {"pg": pa.stderr, "export": pb.stderr, "out": b}


# ── the six-sample cohort ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cohort(docker_stack, tmp_path_factory):
    missing = [p for p in DOWNLOADS_CHR21 if not os.path.exists(p)]
    if missing:
        pytest.skip(f"chr21 downloads not present: {missing}")
    work = str(tmp_path_factory.mktemp("jc_export"))
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    for batch in (L.GVCFS, DOWNLOADS_CHR21[:2], DOWNLOADS_CHR21[2:]):
        L.run_ingest(batch, region="chr21", threads=2)
    L.run_aggregate()
    return {"work": work}


def test_fresh_cohort_export_and_goldens(cohort):
    work = cohort["work"]
    p1 = L.run_cohort_state()
    p2 = L.run_cohort_state(samples=["HG00099", "HG00245"])
    assert p1.stdout == p2.stdout
    state = _state(p1)
    assert state["stale"] is None and state["status"] == "complete"
    assert state["completed_batches"] == 3
    assert (state["active_samples"], state["removed_samples"]) == (6, 0)
    cohort["state"] = state

    d1 = os.path.join(work, "export1")
    L.run_export_cohort(d1)
    proc = L.run_export_contig(d1, "chr21")
    assert "chr21: exported" in proc.stderr
    doc = json.load(open(os.path.join(d1, "cohort.json")))
    assert doc["state"] == state
    (c21,) = doc["contigs"]
    assert c21["name"] == "chr21" and c21["has_sites"]
    assert [s["name"] for s in doc["samples"]] == sorted(
        s["name"] for s in doc["samples"]
    )
    i = c21["index"]
    assert sorted(os.listdir(d1)) == sorted(
        [
            "cohort.json",
            f"sites.c{i}.parquet",
            f"alleles.c{i}.parquet",
            f"export.c{i}.json",
        ]
    )
    side = json.load(open(os.path.join(d1, f"export.c{i}.json")))
    assert side["state"] == state and side["rows"]["sites"] == 132_382
    cohort["d1"] = d1
    cohort["listing"] = _listing(d1)

    for window, threads in ((10_000_000, 2), (1_000_000, 4)):
        out = os.path.join(work, f"fresh.w{window}.t{threads}.vcf.gz")
        proc = L.run_joint_call(
            out,
            from_export=d1,
            extra_env=NO_DB,
            window=window,
            threads=threads,
            max_alt_alleles=0,
        )
        L.assert_golden("fresh_cohort_chr21", out, proc.stderr)
    cohort["fresh"] = out
    out = os.path.join(work, "slice.vcf.gz")
    proc = _from_export(d1, out, region=SLICE, max_alt_alleles=0)
    L.assert_golden("slice_chr21_10M_12M", out, proc.stderr)


def test_no_connection_control(cohort):
    """The environment of the --from-export runs fails a PostgreSQL-path
    call (and --cohort-state): the runs above did not connect."""
    out = os.path.join(cohort["work"], "control.vcf.gz")
    proc = L.run_joint_call(out, extra_env=NO_DB, check=False)
    assert proc.returncode == 1, proc.stderr[-2000:]
    assert "Error:" in proc.stderr and not os.path.exists(out)
    proc = L.run_cohort_state(extra_env=NO_DB, check=False)
    assert proc.returncode == 1 and proc.stdout == ""


def test_equal_to_the_postgres_path(cohort, tmp_path):
    work, d1 = cohort["work"], cohort["d1"]
    one = _same_as_postgres(work, d1, "one", samples=["HG00099"])
    # Two samples of different batches, given out of the header's order.
    two = _same_as_postgres(
        work, d1, "two", samples=["HG00245", "HG00101"], max_alt_alleles=0
    )
    assert "Sample subset: 2 of 6 active samples: HG00101, HG00245" in (
        two["export"]
    )
    assert L.joint_call_files_read(two["pg"]) == L.joint_call_files_read(
        two["export"]
    )
    assert L.joint_call_files_read(one["export"]) == (1, 1)
    names = tmp_path / "names"
    names.write_text("HG00102\n\nHG00420\n")
    _same_as_postgres(work, d1, "file", samples_file=str(names))
    _same_as_postgres(work, d1, "trim", trim_alt_alleles=True, min_ac=1)
    for cap in (0, 3, None):
        _same_as_postgres(work, d1, f"cap{cap}", max_alt_alleles=cap)
    # A one-site region inside a spanning deletion: the first record with
    # a `*` ALT (test_joint_call_oracle.py's (f)).
    star_pos = next(
        int(f[1])
        for f in (r.split("\t", 5) for r in _records(cohort["fresh"]))
        if "*" in f[4].split(",")
    )
    r = _same_as_postgres(
        work, d1, "one_site", region=f"chr21:{star_pos}-{star_pos}"
    )
    assert len(_records(r["out"])) == 1
    # A region without a site.
    r = _same_as_postgres(work, d1, "no_site", region="chr21:1-1000")
    assert _records(r["out"]) == []


def test_the_export_directory_is_untouched(cohort):
    assert _listing(cohort["d1"]) == cohort["listing"]


def test_refusals_on_real_stores(cohort):
    work, d1 = cohort["work"], cohort["d1"]
    proc = L.run_export_contig(d1, "chr22", check=False)
    assert proc.returncode == 2, proc.stderr
    assert "contig 'chr22' is not in the cohort export" in proc.stderr
    bare = os.path.join(work, "bare")
    os.makedirs(bare)
    shutil.copy(os.path.join(d1, "cohort.json"), bare)
    out = os.path.join(work, "bare.vcf.gz")
    proc = L.run_joint_call(
        out, from_export=bare, extra_env=NO_DB, check=False
    )
    assert proc.returncode == 2, proc.stderr
    assert re.search(
        r"chr21 has sites but the export holds no sites\.c\d+\.parquet, "
        r"alleles\.c\d+\.parquet, export\.c\d+\.json \(run --export-contig",
        proc.stderr,
    ), proc.stderr
    assert not os.path.exists(out)
    assert os.listdir(bare) == ["cohort.json"]
    assert _listing(d1) == cohort["listing"]


def test_after_remove_stale(cohort):
    work, d1 = cohort["work"], cohort["d1"]
    L.run_remove(["HG00100"])
    proc = L.run_cohort_state(check=False)
    assert proc.returncode == 2 and proc.stdout == ""
    assert (
        "Refusing to joint-call stale aggregates; pass --allow-stale to "
        "override."
    ) in proc.stderr
    assert "does not cover the completed ingest batches" in proc.stderr
    state = _state(L.run_cohort_state(allow_stale=True))
    assert state["stale"] and "does not cover" in state["stale"]
    assert state["removed_samples"] == 1
    # The first export: refused, the directory unchanged.
    proc = L.run_export_contig(d1, "chr21", check=False)
    assert proc.returncode == 2, proc.stderr
    assert "the database's cohort state differs from" in proc.stderr
    assert "--export-cohort" in proc.stderr
    assert _listing(d1) == cohort["listing"]
    d2 = os.path.join(work, "export2")
    proc = L.run_export_cohort(d2, check=False)
    assert proc.returncode == 2 and not os.path.exists(d2)
    L.run_export_cohort(d2, allow_stale=True)
    L.run_export_contig(d2, "chr21")
    doc = json.load(open(os.path.join(d2, "cohort.json")))
    assert doc["state"] == state
    assert doc["removed"] == [
        {"id": doc["removed"][0]["id"], "name": "HG00100"}
    ]
    out = os.path.join(work, "stale.vcf.gz")
    proc = _from_export(d2, out, max_alt_alleles=0)
    L.assert_golden("stale_after_remove_HG00100", out, proc.stderr)
    assert "Excluding 1 tombstoned sample(s)" in proc.stderr
    assert "Warning (the export was made under --allow-stale):" in (
        proc.stderr
    )
    proc = L.run_joint_call(
        out + ".sub",
        from_export=d2,
        extra_env=NO_DB,
        samples=["HG00099", "HG00100"],
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert (
        "argument --samples: removed (samples.status = 'removed'): HG00100"
    ) in proc.stderr
    cohort["d2"] = d2


def test_after_the_aggregate(cohort):
    work, d2 = cohort["work"], cohort["d2"]
    L.run_aggregate()
    before = _listing(d2)
    proc = L.run_export_contig(d2, "chr21", check=False)
    assert proc.returncode == 2, proc.stderr
    assert _listing(d2) == before
    d3 = os.path.join(work, "export3")
    L.run_export_cohort(d3)
    L.run_export_contig(d3, "chr21")
    out = os.path.join(work, "removed.vcf.gz")
    proc = _from_export(d3, out, max_alt_alleles=0)
    L.assert_golden("removed_HG00100_aggregated", out, proc.stderr)
    assert "Excluding 1 tombstoned sample(s)" in proc.stderr
    assert "Warning" not in proc.stderr


# ── two contigs (its own store; last in the file) ───────────────────────────


def test_two_contigs(docker_stack, tmp_path):
    if not os.path.exists(SLICE_2CHR):
        pytest.skip(
            f"{SLICE_2CHR} not downloaded (bcftools view -r "
            "chr21:1-15000000,chr22:10500000-20000000 of HG00099)"
        )
    L.reset_state()
    L.run_init(SLICE_2CHR, "chr21,chr22")
    L.run_ingest([SLICE_2CHR], threads=2)
    L.run_aggregate()
    work = str(tmp_path)
    d = os.path.join(work, "export")
    L.run_export_cohort(d)
    for contig in ("chr21", "chr22"):
        L.run_export_contig(d, contig)
    doc = json.load(open(os.path.join(d, "cohort.json")))
    assert [(c["name"], c["has_sites"]) for c in doc["contigs"]] == [
        ("chr21", True),
        ("chr22", True),
    ]
    whole = _same_as_postgres(work, d, "whole")
    assert {r.split("\t", 1)[0] for r in _records(whole["out"])} == {
        "chr21",
        "chr22",
    }
    # chr22's files alone serve --region chr22 and refuse the genome.
    only22 = os.path.join(work, "only22")
    os.makedirs(only22)
    shutil.copy(os.path.join(d, "cohort.json"), only22)
    i22 = next(c["index"] for c in doc["contigs"] if c["name"] == "chr22")
    i21 = next(c["index"] for c in doc["contigs"] if c["name"] == "chr21")
    for name in (
        f"sites.c{i22}.parquet",
        f"alleles.c{i22}.parquet",
        f"export.c{i22}.json",
    ):
        shutil.copy(os.path.join(d, name), only22)
    r = _same_as_postgres(work, only22, "chr22", region="chr22")
    assert _records(r["out"])
    out: Optional[str] = os.path.join(work, "only22.vcf.gz")
    proc = L.run_joint_call(
        out, from_export=only22, extra_env=NO_DB, check=False
    )
    assert proc.returncode == 2, proc.stderr
    assert f"chr21 has sites but the export holds no sites.c{i21}" in (
        proc.stderr
    )
    assert not os.path.exists(out)
