# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""workflow/bin/fetch_and_ingest.py's `mosaic ingest` command:
--defer-register, --ref-gq-bands, --region and --threads forwarding; no
--engine (removed with the earlier Python ingest worker: an argparse error
of the script). No download, no subprocess: subprocess.run and the download
are monkeypatched."""

import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BIN_DIR = os.path.join(REPO_ROOT, "workflow", "bin")
SCRIPT = os.path.join(BIN_DIR, "fetch_and_ingest.py")


@pytest.fixture
def fai(monkeypatch):
    """The script imported by path (its sibling with_db_password.py on
    sys.path), with subprocess.run capturing the command."""
    monkeypatch.syspath_prepend(BIN_DIR)
    spec = importlib.util.spec_from_file_location("fetch_and_ingest", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.calls = calls
    return mod


def _opt(cmd, name):
    """The value after option `name` in cmd (None: absent)."""
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_no_engine_in_the_command(fai):
    fai.ingest(["a.g.vcf.gz", "b.g.vcf.gz"], 100_000, 500_000, 4)
    cmd = fai.calls[-1]
    assert "--engine" not in cmd
    assert cmd[:4] == [sys.executable, "-m", "mosaic.cli", "ingest"]
    assert cmd[-2:] == ["a.g.vcf.gz", "b.g.vcf.gz"]
    with pytest.raises(TypeError):
        fai.ingest(["a"], 100_000, 500_000, 4, engine="rust")


def test_other_forwarding_unchanged(fai):
    fai.ingest(["a"], 100_000, 500_000, 3)
    cmd = fai.calls[-1]
    assert _opt(cmd, "--threads") == "3"
    assert _opt(cmd, "--pos-batch-size") == "100000"
    assert _opt(cmd, "--shard-size") == "500000"
    for absent in ("--defer-register", "--ref-gq-bands", "--region"):
        assert absent not in cmd
    fai.ingest(
        ["a"],
        100_000,
        500_000,
        8,
        region="chr21",
        defer_register=True,
        ref_gq_bands="20,60",
    )
    cmd = fai.calls[-1]
    assert _opt(cmd, "--threads") == "8"
    assert _opt(cmd, "--region") == "chr21"
    assert "--defer-register" in cmd
    assert _opt(cmd, "--ref-gq-bands") == "20,60"
    assert "--engine" not in cmd
    assert cmd[-1] == "a"


def test_main_forwards_the_cli_options(fai, monkeypatch, tmp_path):
    monkeypatch.delenv("MOSAIC_DB_PASSWORD_SECRET", raising=False)
    monkeypatch.setattr(fai, "fetch_db_password_into_env", lambda: None)
    monkeypatch.setattr(fai, "download_sample", lambda s, p, r: 1)
    argv = [
        "fetch_and_ingest.py",
        "--batch-id",
        "7",
        "--workdir",
        str(tmp_path),
        "--threads",
        "2",
        "--region",
        "chr21",
        "--defer-register",
        "--ref-gq-bands",
        "20",
    ]
    monkeypatch.setattr(sys, "argv", argv + ["HG00096", "HG00097"])
    fai.main()
    (cmd,) = fai.calls
    assert "--engine" not in cmd
    assert _opt(cmd, "--threads") == "2"
    assert _opt(cmd, "--region") == "chr21"
    assert "--defer-register" in cmd
    assert _opt(cmd, "--ref-gq-bands") == "20"
    assert cmd[-2:] == [
        os.path.join(str(tmp_path), "batch_7_HG00096.gvcf.gz"),
        os.path.join(str(tmp_path), "batch_7_HG00097.gvcf.gz"),
    ]


def test_engine_option_is_an_argparse_error(fai, monkeypatch, capsys):
    argv = ["fetch_and_ingest.py", "--batch-id", "7", "--engine", "rust"]
    monkeypatch.setattr(sys, "argv", argv + ["HG00096"])
    with pytest.raises(SystemExit) as e:
        fai.main()
    assert e.value.code == 2
    assert "unrecognized arguments: --engine" in capsys.readouterr().err
