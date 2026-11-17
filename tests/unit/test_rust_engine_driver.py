# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The Python side of the Rust ingest worker: the removed --engine option
(an argparse error), the test harness's worker rule, the worker's manifest, the result-line parsing, the delivery of results in
shard order to the checkpointer, the error paths, and the engine column's
DDL. No database, no Docker; the worker binary only where a test says so.
"""

import argparse
import json
import os
import stat
import sys

import pytest

import jc_testlib as L
from mosaic import ingest
from mosaic import init_db

pytestmark = pytest.mark.unit


def _parser():
    p = argparse.ArgumentParser()
    ingest.configure_parser(p)
    return p


# ── --engine is gone (K6) ───────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["rust", "python"])
def test_engine_option_is_an_argparse_error(capsys, value):
    """``mosaic ingest --engine`` was removed with the Python worker (K6):
    an unknown argument, exit 2; ``MOSAIC_INGEST_ENGINE`` is read nowhere."""
    with pytest.raises(SystemExit) as e:
        _parser().parse_args(["x.g.vcf.gz", "--engine", value])
    assert e.value.code == 2
    assert "unrecognized arguments: --engine" in capsys.readouterr().err
    assert not hasattr(ingest, "INGEST_ENGINE_ENV")
    assert ingest.INGEST_ENGINE == "rust"
    args = _parser().parse_args(["x.g.vcf.gz"])
    assert not hasattr(args, "engine")


def test_engine_option_is_an_error_on_the_cli():
    import subprocess

    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "mosaic.cli",
            "ingest",
            "x",
            "--engine",
            "rust",
        ],
        capture_output=True,
        text=True,
        cwd=L.REPO_ROOT,
    )
    assert p.returncode == 2
    assert "unrecognized arguments: --engine" in p.stderr


# ── the test harness's ingest worker (K6) ───────────────────────────────────


def _harness(monkeypatch, tmp_path, built, worker_env=None):
    """jc_testlib.ingest_worker_env on a clean environment: the release
    build present or not, nothing on PATH."""
    build = tmp_path / "build" / "mosaic-ingest-worker"
    if built:
        build.parent.mkdir(parents=True)
        build.write_text("#!/bin/sh\n")
        build.chmod(build.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(L, "RUST_WORKER_BUILD", str(build))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("MOSAIC_INGEST_ENGINE", raising=False)
    if worker_env is None:
        monkeypatch.delenv("MOSAIC_INGEST_WORKER", raising=False)
    else:
        monkeypatch.setenv("MOSAIC_INGEST_WORKER", worker_env)
    env = os.environ.copy()
    L.ingest_worker_env(env)
    return env, str(build)


# The tests that run the real worker read the chr21 fixture gVCFs, which
# are not distributed with the repository: skip without them.
_FIXTURES_PRESENT = all(os.path.exists(p) for p in L.GVCFS)
_needs_worker_and_fixtures = pytest.mark.skipif(
    not (L.rust_worker_available() and _FIXTURES_PRESENT),
    reason=(
        L.RUST_WORKER_SKIP_REASON
        if not L.rust_worker_available()
        else "the chr21 fixture gVCFs under test_data/ are not present"
    ),
)


def test_harness_worker_set_to_the_build(monkeypatch, tmp_path):
    env, build = _harness(monkeypatch, tmp_path, built=True)
    assert env["MOSAIC_INGEST_WORKER"] == build
    assert L.test_env()["MOSAIC_INGEST_WORKER"] == build
    assert L.rust_worker_path() == build
    # No engine switch is left in the harness's environment.
    assert "MOSAIC_INGEST_ENGINE" not in L.test_env()
    L.require_ingest_worker()  # no skip
    # An exported worker is kept.
    env, _ = _harness(
        monkeypatch, tmp_path / "w", True, worker_env="/usr/bin/true"
    )
    assert env["MOSAIC_INGEST_WORKER"] == "/usr/bin/true"


def test_harness_no_binary_skips(monkeypatch, tmp_path):
    """Without a worker binary every way a test runs ingest skips with
    the reason: run_ingest, stage_batches, popen_pipeline("ingest"), the
    worker harness."""
    import worker_harness as H

    env, _ = _harness(monkeypatch, tmp_path, built=False)
    assert "MOSAIC_INGEST_WORKER" not in env
    assert L.rust_worker_path() is None
    with pytest.raises(pytest.skip.Exception, match="not built"):
        L.require_ingest_worker()
    with pytest.raises(pytest.skip.Exception, match="not built"):
        L.run_ingest(["a.g.vcf.gz"])
    with pytest.raises(pytest.skip.Exception, match="not built"):
        L.stage_batches(["chr21:1-2"], str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="not built"):
        L.popen_pipeline("ingest", ["a.g.vcf.gz"])
    with pytest.raises(pytest.skip.Exception, match="not built"):
        H.run_worker([], [("chr21", 0, 1)], str(tmp_path))
    # An unusable exported worker is no binary either.
    _harness(monkeypatch, tmp_path / "w", True, worker_env="/nonexistent")
    with pytest.raises(pytest.skip.Exception):
        L.require_ingest_worker()


def test_harness_skip_ignores_a_tests_extra_env(monkeypatch, tmp_path):
    """The skip decision reads the real environment: an unusable worker a
    test passes in ``extra_env`` (test_missing_binary_writes_nothing) does
    not make the test skip."""
    _harness(monkeypatch, tmp_path, built=True)
    calls = []
    monkeypatch.setattr(L, "_run", lambda args, **kw: calls.append((args, kw)))
    L.run_ingest(["a.g.vcf.gz"], extra_env={"MOSAIC_INGEST_WORKER": "/nope"})
    assert calls and calls[0][0][:4] == [
        "-m",
        "mosaic.cli",
        "ingest",
        "a.g.vcf.gz",
    ]
    assert "--engine" not in calls[0][0]
    assert calls[0][1]["extra_env"] == {"MOSAIC_INGEST_WORKER": "/nope"}


# ── the test harness's joint-call binary ─────────────────────────────────────


def _jc_harness(monkeypatch, tmp_path, built, worker_env=None):
    """jc_testlib.joint_call_worker_env on a clean environment: the release
    build of mosaic-joint-call present or not, nothing on PATH."""
    build = tmp_path / "build" / "mosaic-joint-call"
    if built:
        build.parent.mkdir(parents=True)
        build.write_text("#!/bin/sh\n")
        build.chmod(build.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(L, "RUST_JOINT_CALL_BUILD", str(build))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    if worker_env is None:
        monkeypatch.delenv("MOSAIC_JOINT_CALL_WORKER", raising=False)
    else:
        monkeypatch.setenv("MOSAIC_JOINT_CALL_WORKER", worker_env)
    env = os.environ.copy()
    L.joint_call_worker_env(env)
    return env, str(build)


def test_jc_harness_worker_set_to_the_build(monkeypatch, tmp_path):
    env, build = _jc_harness(monkeypatch, tmp_path, built=True)
    assert env["MOSAIC_JOINT_CALL_WORKER"] == build
    assert L.test_env()["MOSAIC_JOINT_CALL_WORKER"] == build
    assert L.joint_call_worker_path() == build
    # No engine switch is left in the harness's environment.
    assert "MOSAIC_JOINT_CALL_ENGINE" not in L.test_env()
    L.require_joint_call_worker()  # no skip
    # An exported worker is kept.
    env, _ = _jc_harness(
        monkeypatch, tmp_path / "w", True, worker_env="/usr/bin/true"
    )
    assert env["MOSAIC_JOINT_CALL_WORKER"] == "/usr/bin/true"


def test_jc_harness_no_binary_skips(monkeypatch, tmp_path):
    env, _ = _jc_harness(monkeypatch, tmp_path, built=False)
    assert "MOSAIC_JOINT_CALL_WORKER" not in env
    assert L.joint_call_worker_path() is None
    with pytest.raises(pytest.skip.Exception, match="not built"):
        L.require_joint_call_worker()
    with pytest.raises(pytest.skip.Exception):
        L.run_joint_call(str(tmp_path / "x.vcf.gz"))
    # An unusable exported worker is no binary either.
    _jc_harness(monkeypatch, tmp_path / "w", True, worker_env="/nonexistent")
    with pytest.raises(pytest.skip.Exception):
        L.require_joint_call_worker()


def test_jc_skip_ignores_a_tests_extra_env(monkeypatch, tmp_path):
    """The skip decision reads the real environment: an unusable worker a
    test passes in ``extra_env`` (test_binary_missing_refused) does not
    make the test skip."""
    _jc_harness(monkeypatch, tmp_path, built=True)
    calls = []
    monkeypatch.setattr(L, "_run", lambda args, **kw: calls.append((args, kw)))
    L.run_joint_call(
        "out.vcf.gz", extra_env={"MOSAIC_JOINT_CALL_WORKER": "/nope"}
    )
    assert calls and calls[0][0] == ["joint_call.py", "out.vcf.gz"]
    assert calls[0][1]["extra_env"] == {"MOSAIC_JOINT_CALL_WORKER": "/nope"}


def test_find_worker(monkeypatch, tmp_path):
    exe = tmp_path / "w"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(ingest.INGEST_WORKER_ENV, str(exe))
    assert ingest.find_ingest_worker() == str(exe)
    monkeypatch.setenv(ingest.INGEST_WORKER_ENV, "/nonexistent")
    assert ingest.find_ingest_worker() is None
    monkeypatch.delenv(ingest.INGEST_WORKER_ENV)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ingest.find_ingest_worker() is None
    named = tmp_path / ingest.INGEST_WORKER_BINARY
    named.write_text("#!/bin/sh\n")
    named.chmod(named.stat().st_mode | stat.S_IEXEC)
    assert ingest.find_ingest_worker() == str(named)


# ── the manifest ────────────────────────────────────────────────────────────


class _VF:
    def __init__(self, samples):
        self.samples = samples


def _manifest(tmpdir="/tmp/b"):
    return ingest.rust_manifest(
        12,
        ["a.g.vcf.gz", "/d/b.g.vcf.gz"],
        [_VF(["HG2"]), _VF([])],
        [5, None],
        {"chr21": 20, "chr22": 21},
        [5],
        [("chr21", 0, 10), ("chr21", 10, 15), ("chr22", 0, 5)],
        10,
        "20,60",
        tmpdir,
        frozenset({("chr21", 10), ("chr21", 0)}),
        ["HG2"],
    )


def test_manifest_fields():
    m = _manifest()
    assert m == {
        "batch_id": 12,
        "inputs": [
            {"path": "a.g.vcf.gz", "sample_id": 5, "sample_name": "HG2"},
            {
                "path": "/d/b.g.vcf.gz",
                "sample_id": None,
                "sample_name": "b.g.vcf.gz",
            },
        ],
        "contig_map": {"chr21": 20, "chr22": 21},
        "slot_sample_ids": [5],
        "shards": [
            {"chrom": "chr21", "start": 0, "end": 10, "index": 0},
            {"chrom": "chr21", "start": 10, "end": 15, "index": 1},
            {"chrom": "chr22", "start": 0, "end": 5, "index": 2},
        ],
        "pos_batch_size": 10,
        "ref_gq_bands": "20,60",
        "shard_tmpdir": "/tmp/b",
        "committed_shards": [["chr21", 0], ["chr21", 10]],
        "sample_names": ["HG2"],
    }
    json.dumps(m)  # serializable as written


@_needs_worker_and_fixtures
def test_manifest_is_what_the_worker_reads(tmp_path):
    """The binary accepts the manifest (deny_unknown_fields, its
    validation): --count-records over a tiny shard of a real fixture."""
    import subprocess

    m = ingest.rust_manifest(
        1,
        [L.GVCFS[0]],
        [_VF(["HG00245"])],
        [1],
        {"chr21": 20},
        [1],
        [("chr21", 10_000_000, 10_001_000)],
        0,
        None,
        str(tmp_path),
        frozenset(),
        ["HG00245"],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    out = subprocess.run(
        [L.rust_worker_path(), "--manifest", str(path), "--count-records"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith("records=")
    m["extra"] = 1
    path.write_text(json.dumps(m))
    p = subprocess.run(
        [L.rust_worker_path(), "--manifest", str(path)],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 1
    assert "unknown field" in p.stderr


# ── result lines and their delivery ─────────────────────────────────────────


def _line(index, start=None, **over):
    obj = {
        "index": index,
        "shard_chrom": "chr21",
        "shard_start": start if start is not None else index * 10,
        "rb_paths": [f"/t/rb{index}"],
        "si_paths": [],
        "brb_paths": [],
        "bsi_paths": [],
        "bac_paths": [],
        "bgt_paths": [],
        "sample_qc": [[5, 20, [1] + [0] * 18]],
        "variant_count": 2,
        "ref_block_count": 3,
        "group_count": 4,
        "parse_stats": {},
        "foreign_del_spans": [[5, 20, 7, [9, 12]]],
    }
    obj.update(over)
    return json.dumps(obj) + "\n"


def test_parse_result_line():
    idx, rv = ingest.parse_rust_result(_line(3))
    assert idx == 3
    assert "index" not in rv
    assert rv["sample_qc"] == {(5, 20): [1] + [0] * 18}
    assert rv["foreign_del_spans"] == {(5, 20, 7): [9, 12]}
    assert rv["rb_paths"] == ["/t/rb3"]


@pytest.mark.parametrize(
    "line",
    [
        "not json\n",
        '{"index": 0}\n',
        _line(0)[:40],
        _line("x"),
        _line(0, sample_qc=[[1, 2]]),
    ],
)
def test_parse_result_line_refuses(line):
    with pytest.raises(ValueError, match="malformed mosaic-ingest-worker"):
        ingest.parse_rust_result(line)


class _Reduce:
    """Records the delivery order as the checkpointer would see it."""

    def __init__(self):
        self.order = []

    def __call__(self, results, rv=None):
        if rv is None:
            return []
        self.order.append(rv["shard_start"])
        results.append(rv)
        return results


def test_results_delivered_in_shard_order():
    r = _Reduce()
    lines = [_line(2), _line(0), "\n", _line(3), _line(1)]
    out = ingest.deliver_rust_results(lines, 4, r)
    assert r.order == [0, 10, 20, 30]
    assert [x["shard_start"] for x in out] == [0, 10, 20, 30]


def test_delivery_is_incremental():
    """A shard is delivered as soon as every earlier one has arrived (the
    checkpointer commits it then), not at the end."""
    r = _Reduce()
    seen = []

    def lines():
        yield _line(1)
        seen.append(list(r.order))
        yield _line(0)
        seen.append(list(r.order))
        yield _line(2)

    ingest.deliver_rust_results(lines(), 3, r)
    assert seen == [[], [0, 10]]


def test_delivery_errors():
    with pytest.raises(
        ingest.RustShardsMissing, match=r"missing shard index\(es\) \[1\]"
    ):
        ingest.deliver_rust_results([_line(0), _line(2)], 3, _Reduce())
    with pytest.raises(RuntimeError, match="repeated"):
        ingest.deliver_rust_results([_line(0), _line(0)], 2, _Reduce())
    with pytest.raises(RuntimeError, match="out of range"):
        ingest.deliver_rust_results([_line(5)], 2, _Reduce())
    with pytest.raises(RuntimeError, match="malformed"):
        ingest.deliver_rust_results(["{"], 1, _Reduce())


def _fake_worker(tmp_path, body):
    """An executable standing in for the binary: prints ``body``'s lines
    and exits with its status."""
    lines, status = body
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import sys\n"
        f"for line in {lines!r}:\n"
        "    sys.stdout.write(line)\n"
        "    sys.stdout.flush()\n"
        "print('worker log', file=sys.stderr)\n"
        f"sys.exit({status})\n"
    )
    exe = tmp_path / "fake_worker"
    exe.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_run_rust_shards_ok(tmp_path):
    w = _fake_worker(tmp_path, ([_line(1), _line(0)], 0))
    r = _Reduce()
    ingest.run_rust_shards(w, "/m.json", 2, 2, r)
    assert r.order == [0, 10]


def test_run_rust_shards_nonzero_exit(tmp_path):
    w = _fake_worker(tmp_path, ([_line(0)], 3))
    with pytest.raises(RuntimeError, match="exited with status 3"):
        ingest.run_rust_shards(w, "/m.json", 1, 2, _Reduce())
    # Every shard reported but a failing exit: still an error.
    w = _fake_worker(tmp_path, ([_line(0)], 1))
    with pytest.raises(RuntimeError, match="exited with status 1"):
        ingest.run_rust_shards(w, "/m.json", 1, 1, _Reduce())


def test_run_rust_shards_missing_and_bad_line(tmp_path):
    w = _fake_worker(tmp_path, ([_line(0)], 0))
    with pytest.raises(RuntimeError, match="missing"):
        ingest.run_rust_shards(w, "/m.json", 1, 2, _Reduce())
    w = _fake_worker(tmp_path, (["garbage\n", _line(0)], 0))
    with pytest.raises(RuntimeError, match="malformed"):
        ingest.run_rust_shards(w, "/m.json", 1, 1, _Reduce())


def test_run_rust_shards_reduce_failure_kills_the_worker(tmp_path):
    """MOSAIC_TEST_FAIL_SHARD raises inside the checkpointer: the error
    propagates and the worker does not outlive it."""
    w = _fake_worker(tmp_path, ([_line(0), _line(1)], 0))

    def boom(results, rv=None):
        if rv is None:
            return []
        raise RuntimeError("MOSAIC_TEST_FAIL_SHARD: injected")

    with pytest.raises(RuntimeError, match="MOSAIC_TEST_FAIL_SHARD"):
        ingest.run_rust_shards(w, "/m.json", 1, 2, boom)


# ── R3: the real binary on several threads ──────────────────────────────────

_R3_SHARDS = [
    ("chr21", 10_000_000 + i * 130_000, 10_000_000 + (i + 1) * 130_000)
    for i in range(8)
]


def _r3_manifest(tmp_path, name):
    out = tmp_path / name
    out.mkdir()
    m = ingest.rust_manifest(
        1,
        list(L.GVCFS),
        [_VF(["HG00245"]), _VF(["HG00420"])],
        [1, 2],
        {"chr21": 20},
        [1, 2],
        _R3_SHARDS,
        50_000,
        None,
        str(out),
        frozenset(),
        ["HG00245", "HG00420"],
    )
    path = out / "manifest.json"
    path.write_text(json.dumps(m))
    return out, str(path)


@_needs_worker_and_fixtures
def test_worker_threads_same_files_in_shard_order(tmp_path):
    """8 shards at --threads 1 / 2 / 4: every shard delivered once, in
    shard order, and the same local files byte for byte."""
    files = {}
    for threads in (1, 2, 4):
        out, path = _r3_manifest(tmp_path, f"t{threads}")
        r = _Reduce()
        res = ingest.run_rust_shards(
            L.rust_worker_path(), path, threads, len(_R3_SHARDS), r
        )
        assert r.order == [s for _, s, _ in _R3_SHARDS]
        assert len(res) == len(_R3_SHARDS)
        files[threads] = {
            f.name: f.read_bytes()
            for f in sorted(out.iterdir())
            if f.name != "manifest.json"
        }
        assert files[threads]
    assert files[1] == files[2] == files[4]


@_needs_worker_and_fixtures
def test_worker_fail_hook(tmp_path, monkeypatch, capfd):
    """MOSAIC_TEST_FAIL_RUST_SHARD fails the named shard: exit 1, the shard
    named on stderr; a malformed value fails before any shard."""
    out, path = _r3_manifest(tmp_path, "hook")
    monkeypatch.setenv("MOSAIC_TEST_FAIL_RUST_SHARD", "chr21:10390000")
    with pytest.raises(RuntimeError, match="exited with status 1"):
        ingest.run_rust_shards(
            L.rust_worker_path(), path, 4, len(_R3_SHARDS), _Reduce()
        )
    err = capfd.readouterr().err
    assert "Error: shard chr21:10390000-10520000" in err
    assert "injected failure of shard chr21:10390000" in err
    monkeypatch.setenv("MOSAIC_TEST_FAIL_RUST_SHARD", "10390000")
    r = _Reduce()
    with pytest.raises(RuntimeError, match="exited with status 1"):
        ingest.run_rust_shards(
            L.rust_worker_path(), path, 2, len(_R3_SHARDS), r
        )
    assert r.order == []
    assert "expected <chrom>:<start>" in capfd.readouterr().err


# ── the engine column ───────────────────────────────────────────────────────


class _Cur:
    def __init__(self, log, probe):
        self.log = log
        self.probe = probe

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append(" ".join(sql.split()))

    def fetchone(self):
        return (self.probe,)


class _Conn:
    def __init__(self, probe):
        self.log = []
        self.probe = probe
        self.commits = 0

    def cursor(self):
        return _Cur(self.log, self.probe)

    def commit(self):
        self.commits += 1


def test_ensure_ingest_schema_adds_the_engine_column():
    conn = _Conn(probe=False)
    init_db.ensure_ingest_schema(conn)
    assert any("column_name = 'engine'" in s for s in conn.log)
    assert (
        "ALTER TABLE ingestion_progress ADD COLUMN IF NOT EXISTS engine TEXT"
        in conn.log
    )
    # Everything present: the probe only, no DDL.
    conn = _Conn(probe=True)
    init_db.ensure_ingest_schema(conn)
    assert len(conn.log) == 1
    assert "column_name = 'engine'" in conn.log[0]


def test_init_creates_the_engine_column():
    assert init_db.INGESTION_ENGINE_COLUMN_DDL == "engine TEXT"
    src = open(init_db.__file__).read()
    assert src.count("_add_ingest_engine(cur)") == 2  # init and ensure


def test_manifest_sample_names_sorted(tmp_path):
    m = ingest.rust_manifest(
        1,
        ["a", "b"],
        [_VF(["Z"]), _VF(["A"])],
        [1, 2],
        {"c": 0},
        [1, 2],
        [("c", 0, 5)],
        0,
        None,
        str(tmp_path),
        frozenset(),
        ["Z", "A"],
    )
    assert m["sample_names"] == ["A", "Z"]
    assert os.path.basename(m["inputs"][1]["path"]) == "b"


# ── the combine in the worker ────────────────────────────────────────────────


def test_manifest_combine_object():
    spec = ingest.rust_combine_spec(1234)
    assert spec == {
        "max_size_bytes": 1234,
        "rows_per_group": {
            "sample_ref_blocks": ingest.PARQUET_ROW_GROUP_ROWS,
            "sample_site_info": ingest.PARQUET_SITE_ROW_GROUP_ROWS,
        },
        "pending_files": ingest.RUST_COMBINE_PENDING_FILES,
    }
    assert ingest.rust_combine_spec(5, pending_files=1)["pending_files"] == 1
    m = _manifest()
    assert "combine" not in m
    m = ingest.rust_manifest(
        12,
        ["a.g.vcf.gz"],
        [_VF(["HG2"])],
        [5],
        {"chr21": 20},
        [5],
        [("chr21", 0, 10)],
        10,
        None,
        "/tmp/b",
        frozenset(),
        ["HG2"],
        combine=spec,
    )
    assert m["combine"] == spec
    json.dumps(m)


def _file(table, index, chrom=None, path=None, rows=3, ids=None, n=None):
    whole = table in ingest.WHOLE_SAMPLE_TABLES
    return {
        "kind": "file",
        "table": table,
        "chromosome_index": chrom if whole else None,
        "index": index,
        "path": path or f"/t/{table}_{index}.parquet",
        "rows": rows,
        "sample_ids": (ids or [1, 2]) if whole else None,
        "n_samples": (n or 2) if whole else None,
    }


def _summary(files, spans=0):
    return {
        "kind": "combined",
        "files": files,
        "spans_applied": spans,
        "seconds": 0.5,
    }


def _jl(obj):
    return json.dumps(obj) + "\n"


def test_parse_file_and_summary_lines():
    kind, fl = ingest.parse_rust_line(_jl(_file("sample_site_info", 0, 20)))
    assert kind == "file"
    assert fl["table"] == "sample_site_info" and "kind" not in fl
    kind, fl = ingest.parse_rust_line(_jl(_file("batch_genotypes", 3)))
    assert (kind, fl["index"], fl["chromosome_index"]) == (
        "file",
        3,
        None,
    )
    kind, sm = ingest.parse_rust_line(_jl(_summary({"batch_site_info": 1})))
    assert kind == "combined" and sm["files"] == {"batch_site_info": 1}
    # A shard line, with or without kind (R3's lines stand).
    kind, (idx, rv) = ingest.parse_rust_line(_line(4))
    assert (kind, idx) == ("shard", 4)
    obj = json.loads(_line(2))
    obj["kind"] = "shard"
    kind, (idx, rv) = ingest.parse_rust_line(_jl(obj))
    assert (kind, idx, "kind" in rv) == ("shard", 2, False)


def _bad_files():
    out = []
    for key, val in [
        ("table", "samples"),
        ("index", -1),
        ("index", "0"),
        ("rows", None),
        ("path", ""),
        ("chromosome_index", None),
        ("sample_ids", [1]),
        ("n_samples", 0),
    ]:
        f = _file("sample_ref_blocks", 0, 20)
        f[key] = val
        out.append(f)
    f = _file("batch_ref_blocks", 0)
    f["chromosome_index"] = 20
    out.append(f)
    f = _file("batch_ref_blocks", 0)
    f["extra"] = 1
    out.append(f)
    f = _file("batch_ref_blocks", 0)
    del f["rows"]
    out.append(f)
    return out


@pytest.mark.parametrize("obj", _bad_files())
def test_parse_file_line_refuses(obj):
    with pytest.raises(
        ValueError, match="malformed mosaic-ingest-worker file"
    ):
        ingest.parse_rust_line(_jl(obj))


@pytest.mark.parametrize(
    "obj",
    [
        _summary({"samples": 1}),
        _summary({"batch_site_info": -1}),
        _summary({}, spans=-1),
        {**_summary({}), "seconds": "x"},
        {**_summary({}), "extra": 1},
        {"kind": "combined"},
        {"kind": "other"},
    ],
)
def test_parse_summary_line_refuses(obj):
    with pytest.raises(ValueError, match="malformed mosaic-ingest-worker"):
        ingest.parse_rust_line(_jl(obj))


class _FakeS3:
    """upload_file copies the file under ``root`` by its key; ``fail``
    makes the upload of a key containing it raise."""

    def __init__(self, root, fail=None):
        self.root = root
        self.fail = fail
        self.uploads = []

    def upload_file(self, path, bucket, key):
        if self.fail and self.fail in key:
            raise OSError(f"injected upload failure of {key}")
        dst = os.path.join(self.root, key.replace("/", "__"))
        with open(path, "rb") as a, open(dst, "wb") as b:
            b.write(a.read())
        self.uploads.append((bucket, key))


def _combine(tmp_path, fail=None):
    up = tmp_path / "uploaded"
    up.mkdir(exist_ok=True)
    s3 = _FakeS3(str(up), fail)
    return (
        ingest.RustCombine(s3, "bkt", ["HG2", "HG1"], 9, "chr21"),
        s3,
    )


def test_rust_combined_s3_keys():
    names = ["HG2", "HG1"]
    f = _file("sample_site_info", 2, 20)
    h = ingest._whole_sample_key_hash(
        "sample_site_info", names, 9, "chr21", 20, 2
    )
    assert ingest.rust_combined_s3_key(f, names, 9, "chr21") == (
        f"sample_site_info/{h}/data.parquet"
    )
    f = _file("batch_ref_blocks", 1)
    assert "s3://bkt/" + ingest.rust_combined_s3_key(
        f, names, 9, "chr21"
    ) == ingest.combined_s3_uri(
        "batch_ref_blocks", names, 9, "chr21", 1, bucket="bkt"
    )


def _fake_combine_worker(tmp_path, shard_lines, files, summary, status=0):
    """A stand-in for the binary with the R5 protocol: prints the shard
    lines; then, per combined file, writes it, prints its line and waits
    for its acknowledgement on stdin (logged to acks.txt); then the
    summary line."""
    script = tmp_path / "fake_combine_worker.py"
    log = tmp_path / "acks.txt"
    script.write_text(
        "import json, sys\n"
        f"shards = {shard_lines!r}\n"
        f"files = {files!r}\n"
        f"summary = {summary!r}\n"
        "for line in shards:\n"
        "    sys.stdout.write(line)\n"
        "sys.stdout.flush()\n"
        f"log = open({str(log)!r}, 'w')\n"
        "for f in files:\n"
        "    if f.get('path'):\n"
        "        open(f['path'], 'wb').write(b'PAR1' + f['path'].encode())\n"
        "    sys.stdout.write(json.dumps(f) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "    ack = sys.stdin.readline()\n"
        "    if not ack:\n"
        "        print('Error: stdin closed', file=sys.stderr)\n"
        "        sys.exit(1)\n"
        "    log.write(ack)\n"
        "    log.flush()\n"
        "if summary is not None:\n"
        "    sys.stdout.write(json.dumps(summary) + '\\n')\n"
        f"sys.exit({status})\n"
    )
    exe = tmp_path / "fake_combine_worker"
    exe.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe), log


def _two_files(tmp_path):
    return [
        _file("sample_ref_blocks", 0, 20, str(tmp_path / "rb0.parquet")),
        _file("batch_ref_blocks", 0, path=str(tmp_path / "brb0.parquet")),
    ]


def _shard_with(paths):
    return _line(0, rb_paths=paths["rb"], brb_paths=paths["brb"])


def test_run_rust_shards_with_a_combine(tmp_path, capfd):
    files = _two_files(tmp_path)
    shard = _shard_with({"rb": ["/l/rb"], "brb": ["/l/brb"]})
    w, log = _fake_combine_worker(
        tmp_path,
        [shard],
        files,
        _summary({"sample_ref_blocks": 1, "batch_ref_blocks": 1}),
    )
    comb, s3 = _combine(tmp_path)
    r = _Reduce()
    ingest.run_rust_shards(w, "/m.json", 1, 1, r, combine=comb)
    assert r.order == [0]
    # Every file uploaded to its key, deleted, and acknowledged.
    keys = sorted(k for _, k in s3.uploads)
    assert keys == sorted(
        ingest.rust_combined_s3_key(f, ["HG2", "HG1"], 9, "chr21")
        for f in files
    )
    assert all(not os.path.exists(f["path"]) for f in files)
    assert sorted(log.read_text().split()) == sorted(f["path"] for f in files)
    uris = comb.uris_by_table()
    assert len(uris["sample_ref_blocks"]) == len(uris["batch_ref_blocks"]) == 1
    assert uris["batch_genotypes"] == []
    err = capfd.readouterr().err
    assert "Uploaded 3 rows of 2 sample(s) 1..2 (chromosome_index 20)" in err
    assert "Uploaded aggregated 3 rows to s3://bkt/batch_ref_blocks/" in err


def test_uris_in_chromosome_and_index_order(tmp_path):
    comb, _ = _combine(tmp_path)
    for c, i in [(21, 0), (20, 1), (20, 0)]:
        comb.uris[("sample_site_info", c, i)] = f"u{c}.{i}"
    for i in (1, 0):
        comb.uris[("batch_site_info", None, i)] = f"b{i}"
    u = comb.uris_by_table()
    assert u["sample_site_info"] == ["u20.0", "u20.1", "u21.0"]
    assert u["batch_site_info"] == ["b0", "b1"]


@pytest.mark.parametrize(
    "case,message",
    [
        ("missing-file", r"file indices \[1\]: not 0..0"),
        ("summary-count", "summary counts 2 sample_ref_blocks"),
        ("no-local", "combined 0 local batch_ref_blocks file"),
        ("no-summary", "ended before its combine summary"),
        ("repeated", "twice"),
    ],
)
def test_combine_completeness_errors(tmp_path, case, message):
    files = _two_files(tmp_path)
    counts = {"sample_ref_blocks": 1, "batch_ref_blocks": 1}
    paths = {"rb": ["/l/rb"], "brb": ["/l/brb"]}
    summary = _summary(counts)
    if case == "missing-file":
        files[0]["index"] = 1
        summary = _summary(counts)
    elif case == "summary-count":
        summary = _summary({**counts, "sample_ref_blocks": 2})
    elif case == "no-local":
        paths = {"rb": ["/l/rb"], "brb": []}
    elif case == "no-summary":
        summary = None
    elif case == "repeated":
        files = [files[0], dict(files[0], path=str(tmp_path / "x.parquet"))]
    w, _ = _fake_combine_worker(tmp_path, [_shard_with(paths)], files, summary)
    comb, _ = _combine(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        ingest.run_rust_shards(w, "/m.json", 1, 1, _Reduce(), combine=comb)


def test_file_line_before_the_shards_or_without_a_combine(tmp_path):
    f = _file("batch_ref_blocks", 0)
    with pytest.raises(RuntimeError, match="after 0 of 2 shard"):
        comb, _ = _combine(tmp_path)
        ingest.deliver_rust_results(
            [_jl(f), _line(0), _line(1)],
            2,
            _Reduce(),
            comb,
        )
    with pytest.raises(RuntimeError, match="no combine was asked for"):
        ingest.deliver_rust_results([_line(0), _jl(f)], 1, _Reduce())
    comb, _ = _combine(tmp_path)
    with pytest.raises(RuntimeError, match="shard line after a combined"):
        ingest.deliver_rust_results(
            [_line(0), _jl(f), _line(0)], 1, _Reduce(), comb
        )
    comb.abort()


def test_an_upload_error_kills_the_worker_and_fails(tmp_path):
    """The worker waits for an acknowledgement that never comes (the
    upload failed): the uploader kills it, the call raises the upload's
    error, nothing is acknowledged."""
    files = _two_files(tmp_path)
    w, log = _fake_combine_worker(
        tmp_path,
        [_shard_with({"rb": ["/l/rb"], "brb": ["/l/brb"]})],
        files,
        _summary({"sample_ref_blocks": 1, "batch_ref_blocks": 1}),
    )
    comb, _ = _combine(tmp_path, fail="sample_ref_blocks/")
    with pytest.raises(RuntimeError, match="upload of a combined file failed"):
        ingest.run_rust_shards(w, "/m.json", 1, 1, _Reduce(), combine=comb)
    assert log.read_text() == ""


def test_a_worker_error_during_the_combine_fails(tmp_path):
    files = _two_files(tmp_path)[:1]
    w, _ = _fake_combine_worker(
        tmp_path,
        [_shard_with({"rb": ["/l/rb"], "brb": []})],
        files,
        None,
        status=1,
    )
    comb, _ = _combine(tmp_path)
    with pytest.raises(RuntimeError, match="exited with status 1"):
        ingest.run_rust_shards(w, "/m.json", 1, 1, _Reduce(), combine=comb)


# ── R5 with the real binary ─────────────────────────────────────────────────


def _r5_run(tmp_path, name, threads, combine, fail=None):
    """The 8-shard manifest of the R3 test, optionally with a combine
    object (a 1 MB target, so several files per table): the results, the
    shard directory, and the uploaded files {key: bytes}."""
    out, path = _r3_manifest(tmp_path, name)
    if combine:
        m = json.loads(open(path).read())
        m["combine"] = ingest.rust_combine_spec(1 << 20, pending_files=1)
        with open(path, "w") as fh:
            json.dump(m, fh)
    up = tmp_path / f"{name}_up"
    up.mkdir()
    s3 = _FakeS3(str(up), fail)
    comb = (
        ingest.RustCombine(s3, "bkt", ["HG00245", "HG00420"], 1, None)
        if combine
        else None
    )
    res = ingest.run_rust_shards(
        L.rust_worker_path(),
        path,
        threads,
        len(_R3_SHARDS),
        _Reduce(),
        combine=comb,
    )
    uploaded = {f.name: f.read_bytes() for f in sorted(up.iterdir())}
    return res, out, uploaded, comb


@_needs_worker_and_fixtures
def test_worker_combine_threads_and_signature(tmp_path, capfd):
    """The 8-shard manifest with a combine object at --threads 1 / 2 / 4:
    the combined files byte-identical across thread counts, every local
    file deleted, the summary's counts the files'; the combined files (rows with floats
    as bits, row groups, schema, sort declaration, statistics columns)
    against their frozen signature, recorded in K6 stage A while they
    equalled the Python combine's of the same local files."""
    up = {}
    for threads in (1, 2, 4):
        res, out, uploaded, comb = _r5_run(
            tmp_path, f"c{threads}", threads, True
        )
        assert uploaded
        assert comb.summary["files"] == {
            t: len(u) for t, u in comb.uris_by_table().items()
        }
        assert sum(comb.summary["files"].values()) == len(uploaded)
        left = [f.name for f in out.iterdir() if f.name != "manifest.json"]
        assert left == [], left
        up[threads] = uploaded
    assert up[1] == up[2] == up[4]
    capfd.readouterr()

    rdir = tmp_path / "rs_files"
    rdir.mkdir()
    for name, data in up[2].items():
        (rdir / name).write_bytes(data)
    # Several files per table at the 1 MB target.
    assert sum("sample_ref_blocks" in n for n in up[2]) >= 2
    sigs = {n: L.parquet_file_signature(str(rdir / n)) for n in sorted(up[2])}
    L.assert_ingest_signature(
        "driver_combine/r3_8shards_1mb",
        sigs,
        {
            "files": len(sigs),
            "rows": sum(x["num_rows"] for x in sigs.values()),
        },
    )


@_needs_worker_and_fixtures
def test_worker_combine_fail_hook(tmp_path, monkeypatch, capfd):
    """MOSAIC_TEST_FAIL_RUST_SHARD=combine:<table>:<index> fails that file
    in the worker: exit 1, the file named on stderr."""
    monkeypatch.setenv(
        "MOSAIC_TEST_FAIL_RUST_SHARD", "combine:batch_ref_blocks:0"
    )
    with pytest.raises(RuntimeError, match="exited with status 1"):
        _r5_run(tmp_path, "hook", 4, True)
    err = capfd.readouterr().err
    assert "Error: combine batch_ref_blocks file 0" in err
    assert "injected failure of combine batch_ref_blocks file 0" in err
    monkeypatch.setenv("MOSAIC_TEST_FAIL_RUST_SHARD", "combine:nope:0")
    with pytest.raises(RuntimeError, match="exited with status 1"):
        _r5_run(tmp_path, "hook2", 2, True)
    assert "expected combine:<table>:<index>" in capfd.readouterr().err


@_needs_worker_and_fixtures
def test_worker_stdin_closed_is_an_error(tmp_path):
    """The binary with a combine object and its stdin closed (the driver
    gone): exit 1 with the message, whatever stage it was in."""
    import subprocess

    out, path = _r3_manifest(tmp_path, "eof")
    m = json.loads(open(path).read())
    m["combine"] = ingest.rust_combine_spec(1 << 20)
    with open(path, "w") as fh:
        json.dump(m, fh)
    p = subprocess.run(
        [L.rust_worker_path(), "--manifest", path, "--threads", "2"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 1
    assert "the driver is gone" in p.stderr
    assert '"kind":"combined"' not in p.stdout
