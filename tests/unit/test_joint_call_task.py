# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""workflow/bin/joint_call_task.py, the helper of the joint-call workflow:
each
subcommand's driver command, the driver lookup, the password fetched by the
three PostgreSQL subcommands and not by ``call``, TMPDIR and --spill-dir
under the work directory, the record-count check, the publish order (the
summary last) and its size check to a local directory and to S3, the
driver's exit status passed through, no summary after any failure, the file
name of an awkward chromosome name, ``summarize``. The helper is imported by
path; the driver, bcftools and the S3 client are faked (no subprocess, no
store)."""

import importlib.util
import io
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BIN_DIR = os.path.join(REPO_ROOT, "workflow", "bin")
SCRIPT = os.path.join(BIN_DIR, "joint_call_task.py")

STATE = {"format": 1, "agg_run_id": 7, "status": "complete", "stale": None}
CONTIGS = [
    {"index": 0, "name": "chr21", "length": 100, "has_sites": True},
    {"index": 4, "name": "HLA-A*01:01", "length": 50, "has_sites": True},
]


class FakePopen:
    """subprocess.Popen standing in for the driver: ``behave(cmd, env)``
    returns (status, stdout text, stderr text) and may write files."""

    behave = None
    calls: list = []

    def __init__(self, cmd, stdout=None, stderr=None, env=None, **kw):
        FakePopen.calls.append({"cmd": list(cmd), "env": env})
        rc, out, errtext = FakePopen.behave(list(cmd), env)
        self._rc = rc
        self.stdout = io.StringIO(out) if stdout is not None else None
        self.stderr = io.StringIO(errtext)

    def wait(self):
        return self._rc


class FakeS3:
    """upload_file / head_object / delete_object over a dict, in order."""

    def __init__(self):
        self.objects = {}
        self.ops = []
        self.short = set()  # keys whose head_object reports a size off
        self.fail_upload = set()  # keys whose upload raises

    def upload_file(self, local, bucket, key, Config=None):
        self.ops.append(("upload", bucket, key))
        if key in self.fail_upload:
            raise RuntimeError(f"upload of {key} failed")
        assert Config.multipart_chunksize == 64 << 20
        with open(local, "rb") as fh:
            self.objects[(bucket, key)] = fh.read()

    def head_object(self, Bucket, Key):
        n = len(self.objects[(Bucket, Key)])
        return {"ContentLength": n - 1 if Key in self.short else n}

    def delete_object(self, Bucket, Key):
        self.ops.append(("delete", Bucket, Key))
        self.objects.pop((Bucket, Key), None)


@pytest.fixture
def jct(monkeypatch, tmp_path):
    """The helper imported by path (with_db_password.py beside it), with
    the driver, bcftools, the password fetch and S3 faked; the working
    directory a fresh temporary one."""
    monkeypatch.syspath_prepend(BIN_DIR)
    spec = importlib.util.spec_from_file_location("joint_call_task", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    driver = tmp_path / "joint_call.py"
    driver.write_text("# the driver\n")
    monkeypatch.setenv("MOSAIC_JOINT_CALL_DRIVER", str(driver))
    FakePopen.calls = []
    FakePopen.behave = lambda cmd, env: (0, "", "")
    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    mod.password_fetches = []

    def fetch():
        # The real fetch logs on stdout when it loads a password.
        print("[INFO] Loaded MOSAIC_DB_PASSWORD from Secrets Manager")
        mod.password_fetches.append(1)

    monkeypatch.setattr(mod, "fetch_db_password_into_env", fetch)
    mod.bcftools = []
    mod.index_count = None

    def fake_run(cmd, *a, **kw):
        cmd = list(cmd)
        mod.bcftools.append(cmd)
        assert cmd[:2] == ["bcftools", "index"]
        if "-t" in cmd:
            if mod.index_count == "fail":
                raise subprocess.CalledProcessError(1, cmd)
            with open(cmd[-1] + ".tbi", "wb") as fh:
                fh.write(b"TBI")
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"{mod.index_count}\n", stderr=""
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    s3 = FakeS3()
    monkeypatch.setattr(mod, "s3_client", lambda: s3)
    mod.s3 = s3
    mod.driver = str(driver)
    mod.work = str(work)
    mod.tmp = str(tmp_path)
    return mod


def _export(path, contigs=CONTIGS, state=STATE, samples=("A", "B", "C")):
    os.makedirs(path, exist_ok=True)
    doc = {
        "format": 1,
        "state": state,
        "contigs": contigs,
        "samples": [{"id": i, "name": n} for i, n in enumerate(samples)],
        "removed": [],
    }
    with open(os.path.join(path, "cohort.json"), "w") as fh:
        json.dump(doc, fh)
    return path


def _names(path, names):
    with open(path, "w") as fh:
        fh.write("".join(f"{n}\n" for n in names))
    return path


def _call_ok(records=3, extra_err=""):
    """A driver that writes its VCF and prints its closing lines."""

    def behave(cmd, env):
        with open(cmd[2], "wb") as fh:
            fh.write(b"VCF-BYTES")
        err = (
            "engine=rust threads=4\n"
            "mosaic-joint-call: 1 unit(s), 2 sample(s), 0.10 s\n"
            f"\nCompleted! Total processed: {records} variants\n"
            "Catalog cross-check: skipped under --samples / --samples-file\n"
            + extra_err
        )
        return 0, "", err

    return behave


def _run(mod, argv):
    """main(argv); the exit status (0 when it returns)."""
    try:
        mod.main(argv)
    except SystemExit as e:
        return e.code
    return 0


# ── the driver lookup ────────────────────────────────────────────────────


def test_driver_lookup_order(jct, monkeypatch, tmp_path):
    explicit = tmp_path / "explicit.py"
    explicit.write_text("")
    assert jct.find_driver(str(explicit)) == str(explicit)
    assert jct.find_driver(None) == jct.driver  # the env variable
    monkeypatch.delenv("MOSAIC_JOINT_CALL_DRIVER")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    onpath = bindir / "joint_call.py"
    onpath.write_text("")
    onpath.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    assert jct.find_driver(None) == str(onpath)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    # The repository root above the helper.
    assert jct.find_driver(None) == os.path.join(REPO_ROOT, "joint_call.py")
    assert jct.repo_driver() == os.path.join(REPO_ROOT, "joint_call.py")


def test_missing_driver_exits_1_with_the_remedy(jct, monkeypatch, capsys):
    monkeypatch.setenv("MOSAIC_JOINT_CALL_DRIVER", "/nonexistent/jc.py")
    assert _run(jct, ["state"]) == 1
    assert "MOSAIC_JOINT_CALL_DRIVER=/nonexistent/jc.py" in (
        capsys.readouterr().err
    )
    monkeypatch.delenv("MOSAIC_JOINT_CALL_DRIVER")
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(jct, "repo_driver", lambda: "/nonexistent/x.py")
    for argv in (
        ["state"],
        ["export-cohort", "--dir", "e", "--out", "o"],
        ["export-contig", "--dir", "e", "--contig", "chr21", "--out", "o"],
    ):
        assert _run(jct, argv) == 1
        err = capsys.readouterr().err
        assert "the joint-call driver is not found" in err
        assert "MOSAIC_JOINT_CALL_DRIVER" in err
    assert FakePopen.calls == []
    assert _run(jct, ["state", "--driver", "/nonexistent/d.py"]) == 1
    assert "--driver=/nonexistent/d.py" in capsys.readouterr().err


# ── state ────────────────────────────────────────────────────────────────


def test_state_prints_the_drivers_line_only(jct, capsys):
    line = json.dumps(STATE, sort_keys=True, separators=(",", ":"))
    FakePopen.behave = lambda cmd, env: (0, line + "\n", "checks passed\n")
    assert _run(jct, ["state"]) == 0
    out = capsys.readouterr()
    assert out.out == line + "\n"
    assert "[INFO] Loaded" in out.err and "checks passed" in out.err
    assert FakePopen.calls[-1]["cmd"] == [
        sys.executable,
        jct.driver,
        "--cohort-state",
    ]
    assert _run(jct, ["state", "--samples-file", "all.txt"]) == 0
    assert FakePopen.calls[-1]["cmd"][2:] == [
        "--cohort-state",
        "--samples-file",
        "all.txt",
    ]
    assert len(jct.password_fetches) == 2


@pytest.mark.parametrize("rc", [1, 2])
def test_state_passes_the_drivers_status_through(jct, capsys, rc):
    FakePopen.behave = lambda cmd, env: (rc, "", "Error: stale\n")
    assert _run(jct, ["state"]) == rc
    out = capsys.readouterr()
    assert out.out == ""
    assert "Error: stale" in out.err


# ── the exports ──────────────────────────────────────────────────────────


def test_export_cohort(jct, tmp_path):
    def behave(cmd, env):
        _export(cmd[3])
        return 0, "", "Wrote it\n"

    FakePopen.behave = behave
    out = tmp_path / "out"
    argv = ["export-cohort", "--dir", "export", "--out", str(out)]
    assert _run(jct, argv) == 0
    assert FakePopen.calls[-1]["cmd"] == [
        sys.executable,
        jct.driver,
        "--export-cohort",
        "export",
    ]
    assert jct.password_fetches == [1]
    published = out / "gen7" / "export" / "cohort.json"
    assert json.loads(published.read_text())["state"] == STATE
    assert os.path.exists("export/cohort.json")  # copied, not moved


def test_export_cohort_to_s3(jct):
    FakePopen.behave = lambda cmd, env: (0, "", _export(cmd[3]) and "")
    assert (
        _run(
            jct,
            ["export-cohort", "--dir", "export", "--out", "s3://bk/pre/fix/"],
        )
        == 0
    )
    assert jct.s3.ops == [("upload", "bk", "pre/fix/gen7/export/cohort.json")]


@pytest.mark.parametrize("rc", [1, 2])
def test_export_failure_status_and_nothing_published(jct, tmp_path, rc):
    FakePopen.behave = lambda cmd, env: (rc, "", "Error\n")
    out = tmp_path / "out"
    argv = ["export-cohort", "--dir", "export", "--out", str(out)]
    assert _run(jct, argv) == rc
    _export("e2")
    argv = ["export-contig", "--dir", "e2", "--contig", "chr21"]
    assert _run(jct, argv + ["--out", str(out)]) == rc
    assert not out.exists()


def test_export_contig_publishes_the_sidecar_last(jct):
    _export("export")

    def behave(cmd, env):
        for name in ("sites.c4.parquet", "alleles.c4.parquet"):
            with open(os.path.join("export", name), "w") as fh:
                fh.write(name)
        with open("export/export.c4.json", "w") as fh:
            fh.write("{}")
        return 0, "", ""

    FakePopen.behave = behave
    argv = ["export-contig", "--dir", "export", "--contig", "HLA-A*01:01"]
    assert _run(jct, argv + ["--out", "s3://bk"]) == 0
    assert FakePopen.calls[-1]["cmd"] == [
        sys.executable,
        jct.driver,
        "--export-contig",
        "export",
        "--region",
        "HLA-A*01:01",
    ]
    assert jct.password_fetches == [1]
    assert [op[2] for op in jct.s3.ops] == [
        "gen7/export/sites.c4.parquet",
        "gen7/export/alleles.c4.parquet",
        "gen7/export/export.c4.json",
    ]


def test_export_contig_unknown_contig(jct, capsys):
    _export("export")
    argv = ["export-contig", "--dir", "export", "--contig", "chrZ"]
    assert _run(jct, argv + ["--out", "o"]) == 1
    assert "'chrZ' is not in the export" in capsys.readouterr().err
    assert FakePopen.calls == []


# ── call ─────────────────────────────────────────────────────────────────


def _call_argv(out, *extra, contig="chr21", index=0, names=True):
    argv = [
        "call",
        "--set",
        "set_0001",
        "--contig",
        contig,
        "--contig-index",
        str(index),
        "--export",
        "export",
        "--out",
        str(out),
    ]
    if names:
        argv += ["--samples-file", "set_0001.samples.txt"]
    return argv + list(extra)


@pytest.fixture
def call_setup(jct):
    _export("export")
    _names("set_0001.samples.txt", ["A", "C"])
    FakePopen.behave = _call_ok(records=3)
    jct.index_count = 3
    return jct


def test_call_commands_per_option_set(call_setup, tmp_path):
    jct = call_setup
    out = tmp_path / "out"
    work = jct.work
    vcf = os.path.join(work, "set_0001.chr21.vcf.gz")
    base = [
        sys.executable,
        jct.driver,
        vcf,
        "--from-export",
        "export",
        "--region",
        "chr21",
        "--spill-dir",
        os.path.join(work, "spill"),
    ]
    assert _run(jct, _call_argv(out)) == 0
    assert FakePopen.calls[-1]["cmd"] == base + [
        "--samples-file",
        "set_0001.samples.txt",
    ]
    assert _run(jct, _call_argv(out, names=False)) == 0
    assert FakePopen.calls[-1]["cmd"] == base
    extra = [
        "--threads",
        "3",
        "--window",
        "1000000",
        "--max-alt-alleles",
        "0",
        "--trim-alt-alleles",
        "--min-ac",
        "1",
    ]
    jct.bcftools.clear()
    assert _run(jct, _call_argv(out, *extra, names=False)) == 0
    assert FakePopen.calls[-1]["cmd"] == base + extra
    # The index is built on the call's threads.
    assert jct.bcftools[0] == [
        "bcftools",
        "index",
        "-f",
        "-t",
        "--threads",
        "3",
        vcf,
    ]
    # TMPDIR and the spill directory under the work directory; no password.
    env = FakePopen.calls[-1]["env"]
    assert env["TMPDIR"] == os.path.join(work, "tmp")
    assert jct.password_fetches == []
    # Both removed after the run, as are the local VCF and index.
    for name in ("tmp", "spill", "set_0001.chr21.vcf.gz"):
        assert not os.path.exists(os.path.join(work, name))
    assert not os.path.exists(vcf + ".tbi")


def test_call_option_refusals(call_setup, tmp_path):
    for extra in (
        ["--threads", "0"],
        ["--min-ac", "-1"],
        ["--max-alt-alleles", "x"],
    ):
        assert _run(call_setup, _call_argv(tmp_path, *extra)) == 2
    argv = _call_argv(tmp_path)
    argv[argv.index("set_0001")] = "set 1"
    assert _run(call_setup, argv) == 2
    assert FakePopen.calls == []


def test_call_publishes_locally_with_the_summary_last(call_setup, tmp_path):
    jct = call_setup
    out = tmp_path / "out"
    order = []
    real = jct.publish

    def spy(local, o, rel, move=False):
        order.append((os.path.basename(rel), move))
        return real(local, o, rel, move=move)

    jct.publish = spy
    assert _run(jct, _call_argv(out)) == 0
    assert order == [
        ("set_0001.samples.txt", False),
        ("set_0001.chr21.vcf.gz", True),
        ("set_0001.chr21.vcf.gz.tbi", False),
        ("set_0001.chr21.summary.json", False),
    ]
    folder = out / "gen7" / "set_0001"
    assert sorted(os.listdir(folder)) == [
        "set_0001.chr21.summary.json",
        "set_0001.chr21.vcf.gz",
        "set_0001.chr21.vcf.gz.tbi",
        "set_0001.samples.txt",
    ]
    assert (folder / "set_0001.chr21.vcf.gz").read_bytes() == b"VCF-BYTES"
    summary = json.loads((folder / "set_0001.chr21.summary.json").read_text())
    local = json.loads(open("set_0001.chr21.summary.json").read())
    assert summary == local
    assert summary["records"] == 3
    assert summary["samples"] == 2
    assert summary["generation"] == 7
    assert summary["vcf_bytes"] == len(b"VCF-BYTES")
    assert summary["set"] == "set_0001" and summary["chromosome"] == "chr21"
    assert summary["worker_line"].startswith("mosaic-joint-call: 1 unit(s)")
    assert (
        summary["closing_lines"][0] == "Completed! Total processed: 3 variants"
    )
    assert summary["closing_lines"][1].startswith("Catalog cross-check")
    assert set(summary["seconds"]) == {"driver", "index", "publish"}
    assert summary["published"]["vcf"] == str(folder / "set_0001.chr21.vcf.gz")
    assert summary["published"]["tbi"].endswith(".vcf.gz.tbi")
    assert summary["options"]["index"] is True
    assert jct.bcftools == [
        [
            "bcftools",
            "index",
            "-f",
            "-t",
            os.path.join(jct.work, "set_0001.chr21.vcf.gz"),
        ],
        [
            "bcftools",
            "index",
            "-n",
            os.path.join(jct.work, "set_0001.chr21.vcf.gz"),
        ],
    ]


def test_call_to_s3(call_setup):
    jct = call_setup
    assert _run(jct, _call_argv("s3://bk/jc")) == 0
    keys = [(op[0], op[2]) for op in jct.s3.ops]
    folder = "jc/gen7/set_0001"
    assert keys == [
        ("delete", f"{folder}/set_0001.chr21.summary.json"),
        ("upload", f"{folder}/set_0001.samples.txt"),
        ("upload", f"{folder}/set_0001.chr21.vcf.gz"),
        ("upload", f"{folder}/set_0001.chr21.vcf.gz.tbi"),
        ("upload", f"{folder}/set_0001.chr21.summary.json"),
    ]
    summary = json.loads(
        jct.s3.objects[("bk", f"{folder}/set_0001.chr21.summary.json")]
    )
    assert summary["published"]["vcf"] == (
        f"s3://bk/{folder}/set_0001.chr21.vcf.gz"
    )


def test_call_whole_cohort_set(call_setup, tmp_path):
    jct = call_setup
    out = tmp_path / "out"
    argv = _call_argv(out, names=False)
    argv[argv.index("set_0001")] = "cohort"
    assert _run(jct, argv) == 0
    folder = out / "gen7" / "cohort"
    assert sorted(os.listdir(folder)) == [
        "cohort.chr21.summary.json",
        "cohort.chr21.vcf.gz",
        "cohort.chr21.vcf.gz.tbi",
    ]
    summary = json.loads((folder / "cohort.chr21.summary.json").read_text())
    assert summary["samples"] == 3  # cohort.json's active samples


def test_call_no_index(call_setup, tmp_path):
    jct = call_setup
    out = tmp_path / "out"
    jct.index_count = 999  # not consulted
    assert _run(jct, _call_argv(out, "--no-index")) == 0
    assert jct.bcftools == []
    names = os.listdir(out / "gen7" / "set_0001")
    assert not any(n.endswith(".tbi") for n in names)


def test_awkward_chromosome_name(call_setup, tmp_path):
    jct = call_setup
    out = tmp_path / "out"
    argv = _call_argv(out, contig="HLA-A*01:01", index=4)
    assert _run(jct, argv) == 0
    assert FakePopen.calls[-1]["cmd"][6] == "HLA-A*01:01"
    folder = out / "gen7" / "set_0001"
    assert (folder / "set_0001.c4.vcf.gz").exists()
    summary = json.loads((folder / "set_0001.c4.summary.json").read_text())
    assert summary["chromosome"] == "HLA-A*01:01"
    assert summary["contig_index"] == 4
    for name, plain in (
        ("chr1", True),
        ("chrUn_KI270302v1", True),
        ("HLA-A*01:01", False),
        ("chr 1", False),
        ("..", False),
        ("", False),
    ):
        assert (jct.chrom_file_name(name, 9) == name) is plain, name


def _no_summary(jct, out):
    assert not os.path.exists("set_0001.chr21.summary.json")
    folder = os.path.join(str(out), "gen7", "set_0001")
    published = os.listdir(folder) if os.path.isdir(folder) else []
    assert not any(n.endswith(".summary.json") for n in published)
    assert not os.path.exists(os.path.join(jct.work, "set_0001.chr21.vcf.gz"))


@pytest.mark.parametrize("rc", [1, 2])
def test_call_driver_failure_status_and_no_summary(call_setup, tmp_path, rc):
    jct = call_setup
    out = tmp_path / "out"

    def behave(cmd, env):
        with open(cmd[2], "wb") as fh:
            fh.write(b"partial")
        return rc, "", "Error: refused\n"

    FakePopen.behave = behave
    assert _run(jct, _call_argv(out)) == rc
    _no_summary(jct, out)
    assert not out.exists()
    assert jct.bcftools == []


def test_call_killed_driver(call_setup, tmp_path):
    FakePopen.behave = lambda cmd, env: (-9, "", "")
    assert _run(call_setup, _call_argv(tmp_path / "o")) == 137


def test_record_count_check(call_setup, tmp_path, capsys):
    jct = call_setup
    out = tmp_path / "out"
    jct.index_count = 4  # the driver said 3
    assert _run(jct, _call_argv(out)) == 1
    assert "counts 4 records, the driver's Completed! line 3" in (
        capsys.readouterr().err
    )
    _no_summary(jct, out)
    assert not out.exists()  # nothing published
    jct.index_count = 3
    assert _run(jct, _call_argv(out)) == 0


def test_failed_index_leaves_no_summary(call_setup, tmp_path):
    jct = call_setup
    jct.index_count = "fail"
    out = tmp_path / "out"
    assert _run(jct, _call_argv(out)) == 1
    _no_summary(jct, out)


def test_no_completed_line_fails(call_setup, tmp_path, capsys):
    jct = call_setup

    def behave(cmd, env):
        open(cmd[2], "wb").close()
        return 0, "", "something else\n"

    FakePopen.behave = behave
    out = tmp_path / "out"
    assert _run(jct, _call_argv(out)) == 1
    assert "no 'Completed! Total processed' line" in capsys.readouterr().err
    _no_summary(jct, out)


def test_failed_upload_leaves_no_summary(call_setup):
    jct = call_setup
    folder = "jc/gen7/set_0001"
    jct.s3.fail_upload.add(f"{folder}/set_0001.chr21.vcf.gz.tbi")
    with pytest.raises(RuntimeError):
        jct.main(_call_argv("s3://bk/jc"))
    assert not os.path.exists("set_0001.chr21.summary.json")
    assert ("bk", f"{folder}/set_0001.chr21.summary.json") not in (
        jct.s3.objects
    )


def test_size_check_on_s3(call_setup, capsys):
    jct = call_setup
    folder = "jc/gen7/set_0001"
    jct.s3.short.add(f"{folder}/set_0001.chr21.vcf.gz")
    assert _run(jct, _call_argv("s3://bk/jc")) == 1
    assert "holds 8 bytes" in capsys.readouterr().err
    assert ("bk", f"{folder}/set_0001.chr21.vcf.gz") not in jct.s3.objects
    assert not any(k[1].endswith(".summary.json") for k in jct.s3.objects)
    assert not os.path.exists("set_0001.chr21.summary.json")


def test_size_check_of_the_summary_on_s3(call_setup):
    jct = call_setup
    key = "jc/gen7/set_0001/set_0001.chr21.summary.json"
    jct.s3.short.add(key)
    assert _run(jct, _call_argv("s3://bk/jc")) == 1
    assert ("bk", key) not in jct.s3.objects
    assert not os.path.exists("set_0001.chr21.summary.json")


def test_size_check_local(call_setup, tmp_path, monkeypatch):
    jct = call_setup
    real = jct.os.path.getsize
    out = str(tmp_path / "out")

    def lying(path):
        n = real(path)
        return n + 1 if path.startswith(out) and path.endswith(".tbi") else n

    monkeypatch.setattr(jct.os.path, "getsize", lying)
    assert _run(jct, _call_argv(out)) == 1
    _no_summary(jct, out)
    assert not os.path.exists(
        os.path.join(out, "gen7", "set_0001", "set_0001.chr21.vcf.gz.tbi")
    )


def test_an_earlier_summary_is_removed_first(call_setup):
    """A re-run's publish removes the earlier run's summary before it
    overwrites the VCF: a re-run that fails there leaves none. One that
    fails before publishing leaves the earlier outputs as they were."""
    jct = call_setup
    folder = "jc/gen7/set_0001"
    key = ("bk", f"{folder}/set_0001.chr21.summary.json")
    assert _run(jct, _call_argv("s3://bk/jc")) == 0
    assert key in jct.s3.objects
    jct.index_count = 5
    assert _run(jct, _call_argv("s3://bk/jc")) == 1
    assert key in jct.s3.objects
    jct.index_count = 3
    jct.s3.fail_upload.add(f"{folder}/set_0001.chr21.vcf.gz.tbi")
    with pytest.raises(RuntimeError):
        jct.main(_call_argv("s3://bk/jc"))
    assert key not in jct.s3.objects


def test_out_scheme_refused(call_setup, capsys):
    assert _run(call_setup, _call_argv("gs://bk/x")) == 1
    assert "only a local directory or s3://" in capsys.readouterr().err


# ── summarize ────────────────────────────────────────────────────────────


def _summary(path, set_name, chrom, idx, gen=7, records=5):
    doc = {
        "set": set_name,
        "chromosome": chrom,
        "contig_index": idx,
        "samples": 2,
        "generation": gen,
        "records": records,
        "vcf_bytes": 100 + idx,
        "seconds": {"driver": 1.5, "index": 0.25, "publish": 0.125},
        "published": {"vcf": f"/o/gen{gen}/{set_name}/{set_name}.{chrom}"},
    }
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return path


def test_summarize(jct, tmp_path):
    files = [
        _summary("a.json", "set_0002", "chr2", 1),
        _summary("b.json", "set_0001", "chr10", 9),
        _summary("c.json", "set_0001", "chr2", 1),
        _summary("d.json", "set_0002", "chr1", 0),
    ]
    out = tmp_path / "out"
    assert _run(jct, ["summarize", "--out", str(out)] + files) == 0
    lines = (out / "gen7" / "joint_call_summary.tsv").read_text().splitlines()
    assert lines[0].split("\t") == jct.TSV_COLUMNS
    rows = [x.split("\t") for x in lines[1:]]
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("set_0001", "chr2", "1"),
        ("set_0001", "chr10", "9"),
        ("set_0002", "chr1", "0"),
        ("set_0002", "chr2", "1"),
    ]
    assert rows[0][3:] == [
        "2",
        "5",
        "101",
        "1.5",
        "0.25",
        "0.125",
        "/o/gen7/set_0001/set_0001.chr2",
    ]
    assert open("joint_call_summary.tsv").read() == "\n".join(lines) + "\n"


def test_summarize_refuses_two_generations(jct, tmp_path, capsys):
    files = [
        _summary("a.json", "set_0001", "chr1", 0, gen=7),
        _summary("b.json", "set_0001", "chr2", 1, gen=8),
    ]
    out = tmp_path / "out"
    assert _run(jct, ["summarize", "--out", str(out)] + files) == 1
    assert "generations [7, 8]" in capsys.readouterr().err
    assert not out.exists()
