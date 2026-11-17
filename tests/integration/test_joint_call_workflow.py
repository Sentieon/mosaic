# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The joint-call workflow, workflow/joint_call.nf, run by Nextflow under
the local
profile, from a temporary working directory (``work/``, ``.nextflow*``,
``reports/``, ``logs/`` and ``joint_calls/`` land there, never in the
repository). Which tasks ran, were cached or failed is read from a trace
file of the test's own (``-c``).

  * The wiring, with no store: the real workflow and the
    real helper (workflow/bin/joint_call_task.py) against a fake driver
    (``MOSAIC_JOINT_CALL_DRIVER``) that answers the four export options with
    canned output, at ``--joint_call_index false``: the sets of a list with
    comments, blank lines and a short last set; the set x chromosome tasks;
    a chromosome without sites skipped; ``--region`` narrowing and an
    unknown entry; the whole-cohort set; the workflow's refusals; an exit
    2 in
    COHORT_STATE and in JOINT_CALL ending the run after one submission, an
    exit 1 retried; ``-resume``; the options forwarded; the outputs' layout
    and the summary TSV.
  * A real run on the ``jc_test`` stores (the Docker stack, both binaries,
    bcftools and the chr21 downloads), the store moving forward test by
    test: each VCF's text equals ``joint_call.py --samples-file <set>
    --region <chrom>`` run directly against PostgreSQL with the same
    options, at set sizes 2, 4 and 6 and for the whole cohort; the index's
    record count; a stale cohort and an unknown name stopping the run in
    COHORT_STATE; ``-resume`` after a new aggregate (a new generation
    folder), with a larger ``--batch_size`` (the exports cached) and
    unchanged (COHORT_STATE only); an option set written to S3 (MinIO);
    and, last, two chromosomes on a store of their own.

Skips without Nextflow (``$MOSAIC_TEST_NEXTFLOW``, else the repository's
``nextflow`` launcher, else ``nextflow`` on PATH) or Java (``$JAVA_HOME``,
else ``java`` on PATH).
"""

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest

import jc_testlib as L

WORKFLOW = os.path.join(L.REPO_ROOT, "workflow", "joint_call.nf")
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
# The six samples in the order the batches ingest them (the header's).
SIX = ["HG00245", "HG00420", "HG00099", "HG00100", "HG00101", "HG00102"]
TRACE_FIELDS = "task_id,hash,name,status,exit,attempt,workdir"


def nextflow_path() -> Optional[str]:
    """$MOSAIC_TEST_NEXTFLOW, else <repo>/nextflow, else nextflow on PATH."""
    given = os.environ.get("MOSAIC_TEST_NEXTFLOW")
    if given:
        return given if os.access(given, os.X_OK) else None
    local = os.path.join(L.REPO_ROOT, "nextflow")
    if os.access(local, os.X_OK):
        return local
    return shutil.which("nextflow")


def java_home() -> Tuple[bool, Optional[str]]:
    """(Java usable, the JAVA_HOME to pass): $JAVA_HOME with bin/java, else
    a ``java`` on PATH that runs."""
    home = os.environ.get("JAVA_HOME")
    if home and os.access(os.path.join(home, "bin", "java"), os.X_OK):
        return True, home
    java = shutil.which("java")
    if java is None:
        return False, None
    try:
        ok = (
            subprocess.run(
                [java, "-version"], capture_output=True, timeout=60
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return ok, None


NEXTFLOW = nextflow_path()
JAVA_OK, JAVA_HOME = java_home()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        NEXTFLOW is None,
        reason="no Nextflow: set MOSAIC_TEST_NEXTFLOW, or put the launcher "
        "at <repo>/nextflow or nextflow on PATH",
    ),
    pytest.mark.skipif(
        not JAVA_OK,
        reason="no Java for Nextflow: set JAVA_HOME (with bin/java) or put "
        "a working java on PATH",
    ),
]

TEST_CONFIG = f"""
trace {{
    enabled   = true
    file      = 'trace.txt'
    overwrite = true
    fields    = '{TRACE_FIELDS}'
}}
report   {{ overwrite = true }}
timeline {{ overwrite = true }}
"""


class Run:
    """One ``nextflow run``: the process, its trace rows, its directory."""

    def __init__(self, proc: subprocess.CompletedProcess, cwd: str) -> None:
        self.proc = proc
        self.cwd = cwd
        self.rows: List[Dict[str, str]] = []
        path = os.path.join(cwd, "trace.txt")
        if os.path.exists(path):
            with open(path) as fh:
                self.rows = list(csv.DictReader(fh, delimiter="\t"))

    @property
    def ok(self) -> bool:
        return self.proc.returncode == 0

    @property
    def output(self) -> str:
        return self.proc.stdout + self.proc.stderr

    def tasks(
        self, process: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """The trace rows of a process (all when None) with a status."""
        return [
            r
            for r in self.rows
            if (process is None or r["name"].split(" (")[0] == process)
            and (status is None or r["status"] == status)
        ]

    def statuses(self) -> Dict[str, List[str]]:
        """{process: sorted statuses of its rows}."""
        out: Dict[str, List[str]] = {}
        for r in self.rows:
            out.setdefault(r["name"].split(" (")[0], []).append(r["status"])
        return {k: sorted(v) for k, v in out.items()}

    def tags(self, process: str) -> List[str]:
        """The tags of a process's rows, sorted."""
        return sorted(
            r["name"].split(" (", 1)[1].rstrip(")")
            for r in self.tasks(process)
        )


def run_nf(
    cwd: str,
    args: List[str],
    extra_env: Optional[Dict[str, str]] = None,
    resume: bool = False,
    drop_env: Tuple[str, ...] = (),
) -> Run:
    """``nextflow run workflow/joint_call.nf -profile local`` in ``cwd``
    under the harness's environment (the jc_test stores), the test
    interpreter's directory first on PATH (the helper's ``python3``)."""
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, "test.config"), "w") as fh:
        fh.write(TEST_CONFIG)
    trace = os.path.join(cwd, "trace.txt")
    if os.path.exists(trace):
        os.remove(trace)
    env = L.test_env(extra_env)
    for k in drop_env:
        env.pop(k, None)
    env["PATH"] = os.pathsep.join(
        [os.path.dirname(sys.executable), env.get("PATH", "")]
    )
    env["NXF_OFFLINE"] = "true"
    env["NXF_ANSI_LOG"] = "false"
    if JAVA_HOME:
        env["JAVA_HOME"] = JAVA_HOME
    cmd = [NEXTFLOW, "run", WORKFLOW, "-profile", "local", "-c", "test.config"]
    cmd += args + (["-resume"] if resume else [])
    assert NEXTFLOW is not None
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800
    )
    return Run(proc, cwd)


def write_list(path: str, lines: List[str]) -> str:
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def tsv_rows(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ── the wiring: a fake driver, no store ─────────────────────────────────────

FAKE_DRIVER = r'''"""A stand-in for joint_call.py in the joint-call workflow's wiring tests:
answers --cohort-state, --export-cohort, --export-contig and
--from-export with canned output and no store. Controlled by:

  FAKE_JC_GEN       the cohort generation (agg_run_id; default 1)
  FAKE_JC_LOG       a file each invocation appends one JSON line to
  FAKE_JC_FAIL      <mode>:<status>[:<text>]: the invocations of mode
                    (state, export-cohort, export-contig, call) whose
                    arguments or samples file hold <text> exit <status>
  FAKE_JC_ONCE      a directory: with it, each such failure happens once
                    (a marker file per mode and text)
"""

import json
import os
import sys

CONTIGS = [
    {"index": 0, "name": "chr1", "length": 1000, "has_sites": True},
    {"index": 1, "name": "chr2", "length": 900, "has_sites": True},
    {"index": 2, "name": "chrM", "length": 100, "has_sites": False},
    {"index": 3, "name": "chr3*alt", "length": 50, "has_sites": True},
]
SAMPLES = [f"S{i}" for i in range(1, 8)]


def opt(args, name):
    return args[args.index(name) + 1] if name in args else None


def state():
    return {
        "format": 1,
        "agg_run_id": int(os.environ.get("FAKE_JC_GEN", "1")),
        "status": "complete",
        "aggregated_at": "2026-11-12T00:00:00.000000",
        "completed_batches": 3,
        "active_samples": len(SAMPLES),
        "removed_samples": 0,
        "samples_sha256": "0" * 64,
        "stale": None,
    }


def main():
    args = sys.argv[1:]
    sf = opt(args, "--samples-file")
    names = None
    if sf:
        with open(sf) as fh:
            names = [x.strip() for x in fh if x.strip()]
    if "--cohort-state" in args:
        mode = "state"
    elif "--export-cohort" in args:
        mode = "export-cohort"
    elif "--export-contig" in args:
        mode = "export-contig"
    elif "--from-export" in args:
        mode = "call"
    else:
        print("fake driver: no mode", file=sys.stderr)
        sys.exit(1)
    log = os.environ.get("FAKE_JC_LOG")
    if log:
        with open(log, "a") as fh:
            rec = {
                "mode": mode,
                "argv": args,
                "cwd": os.getcwd(),
                "tmpdir": os.environ.get("TMPDIR"),
                "names": names,
            }
            fh.write(json.dumps(rec) + "\n")
    fail = os.environ.get("FAKE_JC_FAIL", "")
    if fail:
        f_mode, f_rc, *rest = fail.split(":", 2)
        text = rest[0] if rest else ""
        hay = " ".join(args) + " " + " ".join(names or [])
        if f_mode == mode and text in hay:
            once = os.environ.get("FAKE_JC_ONCE")
            marker = (
                os.path.join(once, f"{mode}.{text}".replace("/", "_"))
                if once
                else None
            )
            if marker is None or not os.path.exists(marker):
                if marker:
                    open(marker, "w").close()
                print(f"Error: injected failure of {mode}", file=sys.stderr)
                sys.exit(int(f_rc))
    if mode == "state":
        unknown = [n for n in names or [] if n not in SAMPLES]
        if unknown:
            print(
                "joint_call.py: error: argument --samples-file: not an "
                f"active sample: {', '.join(unknown)}",
                file=sys.stderr,
            )
            sys.exit(2)
        print("fake: checked", file=sys.stderr)
        print(json.dumps(state(), sort_keys=True, separators=(",", ":")))
        return
    if mode == "export-cohort":
        d = opt(args, "--export-cohort")
        os.makedirs(d, exist_ok=True)
        doc = {
            "format": 1,
            "state": state(),
            "contigs": CONTIGS,
            "samples": [
                {"id": i + 1, "name": n} for i, n in enumerate(SAMPLES)
            ],
            "removed": [],
        }
        with open(os.path.join(d, "cohort.json"), "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        print(f"Wrote {d}/cohort.json", file=sys.stderr)
        return
    if mode == "export-contig":
        d = opt(args, "--export-contig")
        name = opt(args, "--region")
        with open(os.path.join(d, "cohort.json")) as fh:
            doc = json.load(fh)
        if doc["state"] != state():
            print("Error: the cohort has changed", file=sys.stderr)
            sys.exit(2)
        c = next(x for x in doc["contigs"] if x["name"] == name)
        i = c["index"]
        for f in (f"sites.c{i}.parquet", f"alleles.c{i}.parquet"):
            with open(os.path.join(d, f), "w") as fh:
                fh.write(f"{f} of {name}\n")
        with open(os.path.join(d, f"export.c{i}.json"), "w") as fh:
            json.dump({"contig": c, "state": doc["state"]}, fh)
        print(f"{name}: exported", file=sys.stderr)
        return
    # call
    out = args[0]
    d = opt(args, "--from-export")
    name = opt(args, "--region")
    with open(os.path.join(d, "cohort.json")) as fh:
        doc = json.load(fh)
    c = next(x for x in doc["contigs"] if x["name"] == name)
    i = c["index"]
    for f in (f"sites.c{i}.parquet", f"alleles.c{i}.parquet"):
        if not os.path.exists(os.path.join(d, f)):
            print(f"Error: no {f}", file=sys.stderr)
            sys.exit(2)
    with open(out, "w") as fh:
        fh.write(f"{name}\t{','.join(names or ['cohort'])}\n")
    n = len(names) if names is not None else len(SAMPLES)
    print("engine=rust threads=4 window=10000000", file=sys.stderr)
    print(
        f"mosaic-joint-call: 1 unit(s), {n} sample(s), 0.01 s",
        file=sys.stderr,
    )
    print(f"\nCompleted! Total processed: {i + 2} variants", file=sys.stderr)
    if names is not None:
        print("Catalog cross-check: skipped under --samples", file=sys.stderr)
    else:
        print(
            "Catalog cross-check: the catalog's non-ref allele counts "
            f"(alleles.count) equal the emitted AC at every record ({i + 2} "
            "records)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
'''

LIST5 = [
    "# the sample list",
    "S1",
    "",
    "S2",
    "  S3  ",
    "# a comment",
    "S4",
    "S5",
]


class Fake:
    """A test's directory, the fake driver and its invocation log."""

    def __init__(self, tmp: str) -> None:
        self.tmp = tmp
        self.dir = os.path.join(tmp, "nf")
        os.makedirs(self.dir)
        self.driver = os.path.join(tmp, "fake_joint_call.py")
        with open(self.driver, "w") as fh:
            fh.write(FAKE_DRIVER)
        self.log = os.path.join(tmp, "fake.log")
        self.env = {
            "MOSAIC_JOINT_CALL_DRIVER": self.driver,
            "FAKE_JC_LOG": self.log,
        }
        self.list5 = write_list(os.path.join(tmp, "list5.txt"), LIST5)

    def run(self, args: List[str], resume: bool = False, **env: str) -> Run:
        return run_nf(
            self.dir,
            args + ["--joint_call_index", "false"],
            extra_env=dict(self.env, **env),
            resume=resume,
        )

    def calls(self, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log):
            return []
        with open(self.log) as fh:
            recs = [json.loads(x) for x in fh if x.strip()]
        return [r for r in recs if mode is None or r["mode"] == mode]

    def gen(self, n: int = 1) -> str:
        return os.path.join(self.dir, "joint_calls", f"gen{n}")


@pytest.fixture
def fake(tmp_path):
    return Fake(str(tmp_path))


def _opt(argv: List[str], name: str) -> Optional[str]:
    return argv[argv.index(name) + 1] if name in argv else None


def test_sets_tasks_and_layout(fake):
    r = fake.run(["--samples", fake.list5, "--batch_size", "2"])
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["COMPLETED"],
        "EXPORT_CONTIG": ["COMPLETED"] * 3,
        "JOINT_CALL": ["COMPLETED"] * 9,
        "JOINT_CALL_SUMMARY": ["COMPLETED"],
    }
    # chrM has no sites: no export, no task, a log line.
    assert r.tags("EXPORT_CONTIG") == ["chr1", "chr2", "chr3*alt"]
    assert "chrM: no sites in the cohort; no export and no call" in r.output
    sets = {"set_0001": ["S1", "S2"], "set_0002": ["S3", "S4"]}
    sets["set_0003"] = ["S5"]
    assert r.tags("JOINT_CALL") == sorted(
        f"{s}/{c}" for s in sets for c in ("chr1", "chr2", "chr3*alt")
    )
    # COHORT_STATE checked the whole list, comments and blank lines
    # skipped, in its order.
    (state,) = fake.calls("state")
    assert state["names"] == ["S1", "S2", "S3", "S4", "S5"]
    gen = fake.gen(1)
    assert sorted(os.listdir(gen)) == [
        "export",
        "joint_call_summary.tsv",
        "set_0001",
        "set_0002",
        "set_0003",
    ]
    assert sorted(os.listdir(os.path.join(gen, "export"))) == sorted(
        ["cohort.json"]
        + [
            f"{k}.c{i}.{e}"
            for i in (0, 1, 3)
            for k, e in (
                ("sites", "parquet"),
                ("alleles", "parquet"),
                ("export", "json"),
            )
        ]
    )
    records = {"chr1": 2, "chr2": 3, "chr3*alt": 5}
    files = {"chr1": "chr1", "chr2": "chr2", "chr3*alt": "c3"}
    for s, names in sets.items():
        folder = os.path.join(gen, s)
        assert sorted(os.listdir(folder)) == sorted(
            [f"{s}.samples.txt"]
            + [
                f"{s}.{f}.{x}"
                for f in files.values()
                for x in ("vcf.gz", "summary.json")
            ]
        )
        with open(os.path.join(folder, f"{s}.samples.txt")) as fh:
            assert fh.read().split() == names
        for chrom, f in files.items():
            summ = read_json(os.path.join(folder, f"{s}.{f}.summary.json"))
            assert summ["set"] == s and summ["chromosome"] == chrom
            assert summ["samples"] == len(names)
            assert summ["records"] == records[chrom]
            assert summ["generation"] == 1
            assert summ["options"]["index"] is False
            assert summ["published"]["vcf"] == os.path.join(
                folder, f"{s}.{f}.vcf.gz"
            )
            with open(summ["published"]["vcf"]) as fh:
                assert fh.read() == f"{chrom}\t{','.join(names)}\n"
    rows = tsv_rows(os.path.join(gen, "joint_call_summary.tsv"))
    assert [(x["set"], x["chromosome"]) for x in rows] == [
        (s, c) for s in sets for c in ("chr1", "chr2", "chr3*alt")
    ]
    assert [int(x["records"]) for x in rows[:3]] == [2, 3, 5]
    # Each call: the set's names file, the task's cpus, TMPDIR and the
    # spill directory under the task's directory, no output option.
    calls = fake.calls("call")
    assert len(calls) == 9
    for c in calls:
        argv = c["argv"]
        assert _opt(argv, "--threads") == "4"
        assert c["tmpdir"] == os.path.join(c["cwd"], "tmp")
        assert _opt(argv, "--spill-dir") == os.path.join(c["cwd"], "spill")
        assert _opt(argv, "--samples-file").endswith(".samples.txt")
        assert c["names"] == sets[_opt(argv, "--samples-file").split(".")[0]]
        for o in ("--max-alt-alleles", "--min-ac", "--window"):
            assert o not in argv
        assert "--trim-alt-alleles" not in argv
    # The logs, published to params.log_dir.
    logs = sorted(os.listdir(os.path.join(fake.dir, "logs")))
    assert logs == sorted(
        ["cohort_state.log", "export_cohort.log", "joint_call_summary.log"]
        + [f"export_{f}.log" for f in files.values()]
        + [f"joint_call_{s}.{f}.log" for s in sets for f in files.values()]
    )


def test_region_and_the_whole_cohort(fake):
    r = fake.run(["--region", "chr2,chrM,chr1"])
    assert r.ok, r.output
    assert r.tags("EXPORT_CONTIG") == ["chr1", "chr2"]
    assert r.tags("JOINT_CALL") == ["cohort/chr1", "cohort/chr2"]
    assert "chrM: no sites in the cohort" in r.output
    (state,) = fake.calls("state")
    assert "--samples-file" not in state["argv"]
    for c in fake.calls("call"):
        assert "--samples-file" not in c["argv"]
    folder = os.path.join(fake.gen(1), "cohort")
    assert sorted(os.listdir(folder)) == [
        "cohort.chr1.summary.json",
        "cohort.chr1.vcf.gz",
        "cohort.chr2.summary.json",
        "cohort.chr2.vcf.gz",
    ]
    summ = read_json(os.path.join(folder, "cohort.chr1.summary.json"))
    assert summ["samples"] == 7
    assert summ["closing_lines"][1].startswith("Catalog cross-check: the")


def test_unknown_region_entry_stops_before_any_contig_export(fake):
    r = fake.run(["--samples", fake.list5, "--region", "chr1,chr9"])
    assert not r.ok
    assert "--region: chr9 is not a contig of the cohort" in r.output
    assert set(r.statuses()) <= {"COHORT_STATE", "EXPORT_COHORT"}
    assert fake.calls("export-contig") == []


@pytest.mark.parametrize(
    "args, message",
    [
        (["--batch_size", "0"], "--batch_size must be an integer >= 1"),
        (
            ["--joint_call_max_alt_alleles", "-1"],
            "--joint_call_max_alt_alleles must be an integer >= 0",
        ),
        (
            ["--joint_call_min_ac", "-2"],
            "--joint_call_min_ac must be an integer >= 0",
        ),
        (["--region", "chr1,,chr2"], "an empty entry"),
        (["--region", "chr1:1-1000"], "chr1:1-1000 is a range"),
        (["EMPTY"], "the list holds no sample name"),
    ],
)
def test_refusals_before_any_task(fake, args, message):
    if args == ["EMPTY"]:
        empty = write_list(os.path.join(fake.tmp, "e.txt"), ["# none", ""])
        args = ["--samples", empty]
    else:
        args = ["--samples", fake.list5] + args
    r = fake.run(args)
    assert not r.ok
    assert message in r.output
    assert r.rows == []
    assert fake.calls() == []


def test_exit_2_in_cohort_state_ends_the_run(fake):
    r = fake.run(["--samples", fake.list5], FAKE_JC_FAIL="state:2")
    assert not r.ok
    (row,) = r.rows
    assert row["name"].startswith("COHORT_STATE")
    assert (row["status"], row["exit"], row["attempt"]) == ("FAILED", "2", "1")
    # A name the driver refuses (exit 2), with its message in the log.
    bad = write_list(os.path.join(fake.tmp, "bad.txt"), ["S1", "NOPE"])
    r = fake.run(["--samples", bad])
    assert not r.ok
    (row,) = r.rows
    assert (row["exit"], row["attempt"]) == ("2", "1")
    with open(os.path.join(row["workdir"], "cohort_state.log")) as fh:
        assert "not an active sample: NOPE" in fh.read()


def test_exit_2_in_joint_call_ends_the_run(fake):
    r = fake.run(
        ["--samples", fake.list5, "--batch_size", "2"],
        FAKE_JC_FAIL="call:2:set_0002",
    )
    assert not r.ok
    failed = r.tasks("JOINT_CALL", "FAILED")
    assert len(failed) == 1
    assert failed[0]["name"].startswith("JOINT_CALL (set_0002/")
    assert (failed[0]["exit"], failed[0]["attempt"]) == ("2", "1")
    assert all(x["attempt"] == "1" for x in r.rows)
    assert r.tasks("JOINT_CALL_SUMMARY") == []
    folder = os.path.join(fake.gen(1), "set_0002")
    if os.path.isdir(folder):
        assert not any(n.endswith(".summary.json") for n in os.listdir(folder))


def test_exit_1_is_retried(fake):
    once = os.path.join(fake.tmp, "once")
    os.makedirs(once)
    r = fake.run(
        ["--samples", fake.list5, "--batch_size", "2", "--region", "chr1"],
        FAKE_JC_FAIL="call:1:set_0001",
        FAKE_JC_ONCE=once,
    )
    assert r.ok, r.output
    rows = [x for x in r.tasks("JOINT_CALL") if "set_0001" in x["name"]]
    assert sorted((x["attempt"], x["status"], x["exit"]) for x in rows) == [
        ("1", "FAILED", "1"),
        ("2", "COMPLETED", "0"),
    ]
    assert len(r.tasks("JOINT_CALL", "COMPLETED")) == 3


def test_resume(fake):
    args = ["--samples", fake.list5, "--batch_size", "2", "--region", "chr1"]
    r = fake.run(args)
    assert r.ok, r.output
    # Unchanged: COHORT_STATE only.
    r = fake.run(args, resume=True)
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["CACHED"],
        "EXPORT_CONTIG": ["CACHED"],
        "JOINT_CALL": ["CACHED"] * 3,
        "JOINT_CALL_SUMMARY": ["CACHED"],
    }
    # A larger --batch_size: the exports reused, the new sets called.
    args4 = ["--samples", fake.list5, "--batch_size", "4", "--region", "chr1"]
    r = fake.run(args4, resume=True)
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["CACHED"],
        "EXPORT_CONTIG": ["CACHED"],
        "JOINT_CALL": ["COMPLETED"] * 2,
        "JOINT_CALL_SUMMARY": ["COMPLETED"],
    }
    # A changed state line: everything again, into the new generation.
    r = fake.run(args4, resume=True, FAKE_JC_GEN="2")
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["COMPLETED"],
        "EXPORT_CONTIG": ["COMPLETED"],
        "JOINT_CALL": ["COMPLETED"] * 2,
        "JOINT_CALL_SUMMARY": ["COMPLETED"],
    }
    assert sorted(os.listdir(fake.gen(2))) == [
        "export",
        "joint_call_summary.tsv",
        "set_0001",
        "set_0002",
    ]
    # A changed call option: the exports reused, the calls made again.
    r = fake.run(
        args4 + ["--joint_call_min_ac", "1"], resume=True, FAKE_JC_GEN="2"
    )
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["CACHED"],
        "EXPORT_CONTIG": ["CACHED"],
        "JOINT_CALL": ["COMPLETED"] * 2,
        "JOINT_CALL_SUMMARY": ["COMPLETED"],
    }
    # A new joint_call_out: every task again, so everything is published
    # there.
    other = os.path.join(fake.tmp, "other_out")
    r = fake.run(
        args4 + ["--joint_call_min_ac", "1", "--joint_call_out", other],
        resume=True,
        FAKE_JC_GEN="2",
    )
    assert r.ok, r.output
    assert set(r.statuses()["EXPORT_COHORT"]) == {"COMPLETED"}
    assert set(r.statuses()["JOINT_CALL"]) == {"COMPLETED"}
    assert sorted(os.listdir(os.path.join(other, "gen2"))) == [
        "export",
        "joint_call_summary.tsv",
        "set_0001",
        "set_0002",
    ]


def test_options_forwarded(fake):
    r = fake.run(
        [
            "--samples",
            fake.list5,
            "--batch_size",
            "5",
            "--region",
            "chr1",
            "--joint_call_max_alt_alleles",
            "3",
            "--joint_call_trim_alt_alleles",
            "--joint_call_min_ac",
            "1",
            "--joint_call_window",
            "1000",
            "--joint_call_threads",
            "2",
        ]
    )
    assert r.ok, r.output
    (c,) = fake.calls("call")
    argv = c["argv"]
    assert _opt(argv, "--max-alt-alleles") == "3"
    assert "--trim-alt-alleles" in argv
    assert _opt(argv, "--min-ac") == "1"
    assert _opt(argv, "--window") == "1000"
    assert _opt(argv, "--threads") == "2"
    summ = read_json(
        os.path.join(fake.gen(1), "set_0001", "set_0001.chr1.summary.json")
    )
    assert summ["options"] == {
        "index": False,
        "max_alt_alleles": 3,
        "min_ac": 1,
        "threads": 2,
        "trim_alt_alleles": True,
        "window": 1000,
    }


# ── a real run on the jc_test stores ────────────────────────────────────────


def _bcftools_count(path: str) -> int:
    proc = subprocess.run(
        ["bcftools", "index", "-n", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(proc.stdout.strip())


def _records(text: bytes) -> int:
    return sum(1 for x in text.split(b"\n") if x and not x.startswith(b"#"))


def _direct(
    work: str, tag: str, names: Optional[List[str]], region: str, **kw: Any
) -> Tuple[bytes, List[str]]:
    """The same call made directly from PostgreSQL: its text and its
    closing lines."""
    out = os.path.join(work, f"direct.{tag}.vcf.gz")
    sf = None
    if names is not None:
        sf = write_list(os.path.join(work, f"direct.{tag}.txt"), names)
    proc = L.run_joint_call(
        out, region=region, samples_file=sf, threads=4, **kw
    )
    text = L.vcf_text_bytes(out)
    os.remove(out)
    return text, L.closing_lines(proc.stderr)


def check_sets(
    store: Dict[str, Any],
    out: str,
    gen: int,
    sets: Dict[str, Optional[List[str]]],
    chroms: List[str] = ["chr21"],
    **kw: Any,
) -> None:
    """Each set x chromosome's published VCF equals the direct call; its
    index's record count and the summary's; the TSV's rows."""
    rows = tsv_rows(os.path.join(out, f"gen{gen}", "joint_call_summary.tsv"))
    assert sorted((x["set"], x["chromosome"]) for x in rows) == sorted(
        (s, c) for s in sets for c in chroms
    )
    for s, names in sets.items():
        folder = os.path.join(out, f"gen{gen}", s)
        if names is not None:
            with open(os.path.join(folder, f"{s}.samples.txt")) as fh:
                assert fh.read().split() == names
        for chrom in chroms:
            vcf = os.path.join(folder, f"{s}.{chrom}.vcf.gz")
            summ = read_json(os.path.join(folder, f"{s}.{chrom}.summary.json"))
            text, closing = _direct(
                store["work"], f"{s}.{chrom}", names, chrom, **kw
            )
            assert L.vcf_text_bytes(vcf) == text, (s, chrom)
            assert summ["records"] == _records(text)
            assert _bcftools_count(vcf) == summ["records"]
            assert summ["closing_lines"] == closing
            assert summ["generation"] == gen
            n = len(names) if names is not None else len(SIX)
            assert summ["samples"] == n
            assert summ["vcf_bytes"] == os.path.getsize(vcf)
            row = next(
                x for x in rows if (x["set"], x["chromosome"]) == (s, chrom)
            )
            assert int(row["records"]) == summ["records"]


def _gen(conn) -> int:
    gen = L.latest_agg_run_id(conn)
    assert gen is not None
    return int(gen)


@pytest.fixture(scope="module")
def store(docker_stack, tmp_path_factory):
    """Two 2-sample batches on chr21 (HG00245 + HG00420, HG00099 +
    HG00100), aggregated."""
    L.require_ingest_worker()
    L.require_joint_call_worker()
    missing = [p for p in DOWNLOADS_CHR21 if not os.path.exists(p)]
    if missing:
        pytest.skip(f"chr21 downloads not present: {missing}")
    if shutil.which("bcftools") is None:
        pytest.skip("bcftools is not on PATH")
    work = str(tmp_path_factory.mktemp("jc_workflow"))
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(L.GVCFS, region="chr21", threads=2)
    L.run_ingest(DOWNLOADS_CHR21[:2], region="chr21", threads=2)
    L.run_aggregate()
    nf = os.path.join(work, "nf")
    return {
        "work": work,
        "nf": nf,
        "out": os.path.join(nf, "joint_calls"),
        "list4": write_list(
            os.path.join(work, "list4.txt"),
            ["# four samples", "HG00245", "HG00420", "", "HG00099", "HG00100"],
        ),
        "list6": write_list(os.path.join(work, "list6.txt"), SIX),
    }


def real(store: Dict[str, Any], args: List[str], resume: bool = True) -> Run:
    """A run on the stores with the real driver, found by the helper at the
    repository root (no MOSAIC_JOINT_CALL_DRIVER)."""
    return run_nf(
        store["nf"],
        args,
        resume=resume,
        drop_env=("MOSAIC_JOINT_CALL_DRIVER",),
    )


def test_real_two_sets(store, pg_conn):
    r = real(
        store,
        [
            "--samples",
            store["list4"],
            "--batch_size",
            "2",
            "--region",
            "chr21",
        ],
        resume=False,
    )
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["COMPLETED"],
        "EXPORT_CONTIG": ["COMPLETED"],
        "JOINT_CALL": ["COMPLETED"] * 2,
        "JOINT_CALL_SUMMARY": ["COMPLETED"],
    }
    gen = _gen(pg_conn)
    store["gen1"] = gen
    check_sets(
        store,
        store["out"],
        gen,
        {"set_0001": SIX[:2], "set_0002": SIX[2:4]},
    )
    # The published export is the one the tasks used.
    export = os.path.join(store["out"], f"gen{gen}", "export")
    (cohort_task,) = r.tasks("EXPORT_COHORT")
    (contig_task,) = r.tasks("EXPORT_CONTIG")
    (task_dir,) = glob.glob(os.path.join(contig_task["workdir"], "export_c*/"))
    assert sorted(os.listdir(export)) == sorted(os.listdir(task_dir))
    for name in os.listdir(task_dir):
        with (
            open(os.path.join(task_dir, name), "rb") as a,
            open(os.path.join(export, name), "rb") as b,
        ):
            assert a.read() == b.read(), name
    with (
        open(
            os.path.join(cohort_task["workdir"], "export", "cohort.json"), "rb"
        ) as a,
        open(os.path.join(export, "cohort.json"), "rb") as b,
    ):
        assert a.read() == b.read()
    assert (
        read_json(os.path.join(export, "cohort.json"))["state"]["agg_run_id"]
        == gen
    )


def test_real_stale_cohort_stops_in_cohort_state(store):
    L.run_ingest(DOWNLOADS_CHR21[2:], region="chr21", threads=2)
    r = real(store, ["--samples", store["list6"], "--batch_size", "2"])
    assert not r.ok
    (row,) = r.rows
    assert row["name"].startswith("COHORT_STATE")
    assert (row["status"], row["exit"], row["attempt"]) == ("FAILED", "2", "1")
    with open(os.path.join(row["workdir"], "cohort_state.log")) as fh:
        assert "Refusing to joint-call stale aggregates" in fh.read()


def test_real_new_generation(store, pg_conn):
    L.run_aggregate()
    gen = _gen(pg_conn)
    assert gen != store["gen1"]
    store["gen2"] = gen
    r = real(
        store,
        [
            "--samples",
            store["list6"],
            "--batch_size",
            "2",
            "--region",
            "chr21",
        ],
    )
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["COMPLETED"],
        "EXPORT_CONTIG": ["COMPLETED"],
        "JOINT_CALL": ["COMPLETED"] * 3,
        "JOINT_CALL_SUMMARY": ["COMPLETED"],
    }
    check_sets(
        store,
        store["out"],
        gen,
        {"set_0001": SIX[:2], "set_0002": SIX[2:4], "set_0003": SIX[4:]},
    )


def test_real_larger_sets_reuse_the_exports(store):
    gen = store["gen2"]
    r = real(
        store,
        [
            "--samples",
            store["list6"],
            "--batch_size",
            "4",
            "--region",
            "chr21",
        ],
    )
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["CACHED"],
        "EXPORT_CONTIG": ["CACHED"],
        "JOINT_CALL": ["COMPLETED"] * 2,
        "JOINT_CALL_SUMMARY": ["COMPLETED"],
    }
    # set_0001 spans two ingest batches.
    check_sets(
        store, store["out"], gen, {"set_0001": SIX[:4], "set_0002": SIX[4:]}
    )
    r = real(
        store,
        [
            "--samples",
            store["list6"],
            "--batch_size",
            "6",
            "--region",
            "chr21",
        ],
    )
    assert r.ok, r.output
    assert r.tasks("EXPORT_CONTIG", "CACHED")
    assert len(r.tasks("JOINT_CALL", "COMPLETED")) == 1
    check_sets(store, store["out"], gen, {"set_0001": SIX})


def test_real_unchanged_resume_runs_cohort_state_only(store):
    r = real(
        store,
        [
            "--samples",
            store["list6"],
            "--batch_size",
            "4",
            "--region",
            "chr21",
        ],
    )
    assert r.ok, r.output
    assert r.statuses() == {
        "COHORT_STATE": ["COMPLETED"],
        "EXPORT_COHORT": ["CACHED"],
        "EXPORT_CONTIG": ["CACHED"],
        "JOINT_CALL": ["CACHED"] * 2,
        "JOINT_CALL_SUMMARY": ["CACHED"],
    }


def test_real_whole_cohort(store):
    r = real(store, ["--region", "chr21"])
    assert r.ok, r.output
    assert r.tags("JOINT_CALL") == ["cohort/chr21"]
    assert r.tasks("EXPORT_CONTIG", "CACHED")
    check_sets(store, store["out"], store["gen2"], {"cohort": None})
    summ = read_json(
        os.path.join(
            store["out"],
            f"gen{store['gen2']}",
            "cohort",
            "cohort.chr21.summary.json",
        )
    )
    assert any(
        x.startswith("Catalog cross-check: ") for x in summ["closing_lines"]
    )
    assert not os.path.exists(
        os.path.join(
            store["out"], f"gen{store['gen2']}", "cohort", "cohort.samples.txt"
        )
    )


def test_real_options_to_s3(store):
    """One set under the trim, --min-ac 1 and a cap of 3, written to S3
    (MinIO, the test bucket) under its own joint_call_out."""
    prefix = "jcwf-options"
    r = real(
        store,
        [
            "--samples",
            store["list6"],
            "--batch_size",
            "6",
            "--region",
            "chr21",
            "--joint_call_trim_alt_alleles",
            "--joint_call_min_ac",
            "1",
            "--joint_call_max_alt_alleles",
            "3",
            "--joint_call_out",
            f"s3://{L.TEST_BUCKET}/{prefix}/",
        ],
    )
    assert r.ok, r.output
    gen = store["gen2"]
    s3 = L.s3_client()
    local = os.path.join(store["work"], "s3copy")
    os.makedirs(local, exist_ok=True)
    base = f"{prefix}/gen{gen}"
    keys = {
        o["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=L.TEST_BUCKET, Prefix=base + "/"
        )
        for o in page.get("Contents", [])
    }
    want = {
        f"{base}/set_0001/set_0001.samples.txt",
        f"{base}/set_0001/set_0001.chr21.vcf.gz",
        f"{base}/set_0001/set_0001.chr21.vcf.gz.tbi",
        f"{base}/set_0001/set_0001.chr21.summary.json",
        f"{base}/joint_call_summary.tsv",
        f"{base}/export/cohort.json",
    }
    assert want <= keys
    export = sorted(k for k in keys if k.startswith(f"{base}/export/"))
    assert len(export) == 4 and f"{base}/export/cohort.json" in export
    vcf = os.path.join(local, "set_0001.chr21.vcf.gz")
    s3.download_file(
        L.TEST_BUCKET, f"{base}/set_0001/set_0001.chr21.vcf.gz", vcf
    )
    s3.download_file(
        L.TEST_BUCKET,
        f"{base}/set_0001/set_0001.chr21.vcf.gz.tbi",
        vcf + ".tbi",
    )
    summ = json.loads(
        s3.get_object(
            Bucket=L.TEST_BUCKET,
            Key=f"{base}/set_0001/set_0001.chr21.summary.json",
        )["Body"].read()
    )
    text, closing = _direct(
        store["work"],
        "options",
        SIX,
        "chr21",
        trim_alt_alleles=True,
        min_ac=1,
        max_alt_alleles=3,
    )
    assert L.vcf_text_bytes(vcf) == text
    assert summ["closing_lines"] == closing
    assert any(x.startswith("ALT trim") for x in closing)
    assert any(x.startswith("Minimum AC (--min-ac 1)") for x in closing)
    assert any(x.startswith("ALT cap (--max-alt-alleles 3)") for x in closing)
    assert summ["records"] == _records(text) == _bcftools_count(vcf)
    assert summ["published"]["vcf"] == (
        f"s3://{L.TEST_BUCKET}/{base}/set_0001/set_0001.chr21.vcf.gz"
    )


def test_real_unknown_name_stops_in_cohort_state(store):
    bad = write_list(
        os.path.join(store["work"], "bad.txt"), ["HG00245", "NOT_A_SAMPLE"]
    )
    r = real(store, ["--samples", bad, "--batch_size", "2"])
    assert not r.ok
    (row,) = r.rows
    assert row["name"].startswith("COHORT_STATE")
    assert (row["exit"], row["attempt"]) == ("2", "1")
    with open(os.path.join(row["workdir"], "cohort_state.log")) as fh:
        log = fh.read()
    assert "argument --samples-file" in log and "unknown: NOT_A_SAMPLE" in log


# ── two chromosomes (a store of its own; last in the file) ─────────────────


def test_real_two_chromosomes(docker_stack, tmp_path):
    if not os.path.exists(SLICE_2CHR):
        pytest.skip(
            f"{SLICE_2CHR} not downloaded (bcftools view -r "
            "chr21:1-15000000,chr22:10500000-20000000 of HG00099)"
        )
    L.require_ingest_worker()
    L.require_joint_call_worker()
    L.reset_state()
    L.run_init(SLICE_2CHR, "chr21,chr22")
    L.run_ingest([SLICE_2CHR], threads=2)
    L.run_aggregate()
    conn = L.get_pg_conn()
    try:
        gen = _gen(conn)
    finally:
        conn.close()
    work = str(tmp_path)
    store = {"work": work, "nf": os.path.join(work, "nf")}
    names = write_list(os.path.join(work, "one.txt"), ["HG00099"])
    r = real(store, ["--samples", names, "--batch_size", "1"], resume=False)
    assert r.ok, r.output
    assert r.tags("JOINT_CALL") == ["set_0001/chr21", "set_0001/chr22"]
    check_sets(
        store,
        os.path.join(store["nf"], "joint_calls"),
        gen,
        {"set_0001": ["HG00099"]},
        chroms=["chr21", "chr22"],
    )
