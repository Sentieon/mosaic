# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Binary-level tests of the Rust ingest worker on crafted gVCFs.

A Python unit test of the Python worker drove its functions on crafted
vcflib-like records. Its binary-level replacement writes the same records
as a small bgzipped, tabix-indexed gVCF (``write_gvcf``), runs
``mosaic-ingest-worker`` on a manifest over one or more shards
(``run_worker``: the production manifest and delivery code,
``mosaic.ingest.rust_manifest`` / ``run_rust_shards``) and asserts on the
local files' rows (``local_rows``) and on the shard result dicts.

No Docker: the worker reads the gVCFs and writes local files only. The
tests skip with ``jc_testlib.RUST_WORKER_SKIP_REASON`` without a binary
(``requires_worker``).

Positions in ``VRec`` are 1-based, as in the VCF text.
"""

import os
import subprocess
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import jc_testlib as L
import sample_slices
from mosaic import ingest

CHROM = "chr21"
CHROM_LENGTH = 46_709_983
BATCH_ID = 7

requires_worker = pytest.mark.skipif(
    not L.rust_worker_available(), reason=L.RUST_WORKER_SKIP_REASON
)

# The header declarations the ingest logic reads (as DRAGEN declares them;
# MQSQ / RAW_MQandDP / RAW_MQ / GTC / INFO SB as GATK-style callers do).
HEADER = [
    "##fileformat=VCFv4.2",
    '##FILTER=<ID=PASS,Description="All filters passed">',
    '##FILTER=<ID=LowQual,Description="Low quality">',
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End">',
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">',
    '##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS MQ">',
    '##INFO=<ID=MQSQ,Number=2,Type=Float,Description="MQ squared sum">',
    '##INFO=<ID=RAW_MQandDP,Number=2,Type=Integer,Description="Raw MQ">',
    '##INFO=<ID=RAW_MQ,Number=1,Type=Float,Description="Raw MQ">',
    '##INFO=<ID=GTC,Number=3,Type=Float,Description="Genotype counts">',
    '##INFO=<ID=SB,Number=4,Type=Integer,Description="Strand bias">',
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">',
    '##FORMAT=<ID=F1R2,Number=R,Type=Integer,Description="F1R2">',
    '##FORMAT=<ID=F2R1,Number=R,Type=Integer,Description="F2R1">',
    '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="GQ">',
    '##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description="Min DP">',
    '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="PL">',
    '##FORMAT=<ID=SB,Number=4,Type=Integer,Description="Strand bias">',
]


def _value(v: Any) -> str:
    if v is None:
        return "."
    if isinstance(v, (list, tuple)):
        return ",".join(_value(x) for x in v)
    return str(v)


class VRec:
    """One gVCF record of one sample: 1-based ``pos``, ``alts`` as VCF
    strings (``<NON_REF>`` included), ``info`` and ``fmt`` in their order
    (``GT`` first in FORMAT). A value None is written ``.``; a list is
    joined by commas."""

    def __init__(
        self,
        pos: int,
        ref: str,
        alts: Sequence[str],
        gt: str,
        info: Optional[Dict[str, Any]] = None,
        qual: Optional[float] = None,
        filt: str = "PASS",
        **fmt: Any,
    ) -> None:
        self.pos = pos
        self.ref = ref
        self.alts = list(alts)
        self.info = dict(info or {})
        self.qual = qual
        self.filt = filt
        self.fmt = {"GT": gt, **fmt}

    def line(self, chrom: str = CHROM) -> str:
        info = ";".join(f"{k}={_value(v)}" for k, v in self.info.items())
        return "\t".join(
            [
                chrom,
                str(self.pos),
                ".",
                self.ref,
                ",".join(self.alts) or ".",
                _value(self.qual),
                self.filt,
                info or ".",
                ":".join(self.fmt),
                ":".join(_value(v) for v in self.fmt.values()),
            ]
        )


def ref_block(
    start: int,
    end: int,
    gt: str = "0/0",
    min_dp: Optional[int] = 10,
    ref: str = "A",
    **fmt: Any,
) -> VRec:
    """A ref block over [start, end] (1-based, inclusive)."""
    f: Dict[str, Any] = {}
    if min_dp is not None:
        f["MIN_DP"] = min_dp
    f.update(fmt)
    return VRec(start, ref, ["<NON_REF>"], gt, {"END": end}, None, **f)


def write_gvcf(
    path: str,
    sample: str,
    records: Sequence[Any],
    chrom: str = CHROM,
    length: int = CHROM_LENGTH,
    extra_header: Sequence[str] = (),
) -> str:
    """Write ``records`` (``VRec`` or raw tab-separated lines) as a
    bgzipped gVCF of one sample with its tabix index; returns ``path``
    (which must end in ``.gz``). Records must be in position order."""
    assert path.endswith(".gz"), path
    text = path[: -len(".gz")]
    lines = list(HEADER) + list(extra_header)
    lines.append(f"##contig=<ID={chrom},length={length}>")
    lines.append(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample
    )
    for r in records:
        lines.append(r if isinstance(r, str) else r.line(chrom))
    with open(text, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    subprocess.run(["bgzip", "-f", text], check=True, capture_output=True)
    subprocess.run(
        ["tabix", "-f", "-p", "vcf", path], check=True, capture_output=True
    )
    return path


class _VF:
    """What ``rust_manifest`` reads of a vcflib VCF: its samples."""

    def __init__(self, samples: List[str]) -> None:
        self.samples = samples


def _sample_of(path: str) -> List[str]:
    out = subprocess.run(
        ["bcftools", "query", "-l", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return out


def run_worker(
    paths: Sequence[str],
    shards: Sequence[Tuple[str, int, int]],
    tmpdir: str,
    pos_batch_size: int = 100_000,
    ref_gq_bands: Optional[str] = None,
    sample_ids: Optional[Sequence[Optional[int]]] = None,
    contig_map: Optional[Dict[str, int]] = None,
    committed: FrozenSet[Tuple[str, int]] = frozenset(),
    threads: int = 1,
) -> List[Dict[str, Any]]:
    """The worker on ``shards`` (0-based half-open starts, as
    ``Sharder.cut`` makes them) over ``paths``; the result dicts in shard
    order. ``sample_ids`` default 1, 2, ... in input order (None: a stream
    without a sample id); ``contig_map`` default ``{chr21: 0}``, the unit
    tests' chromosome index."""
    paths = list(paths)
    vfs = [_VF(_sample_of(p)) for p in paths]
    if sample_ids is None:
        sample_ids = list(range(1, len(paths) + 1))
    sample_ids = list(sample_ids)
    names = [vf.samples[0] for vf, s in zip(vfs, sample_ids) if s is not None]
    manifest = ingest.rust_manifest(
        BATCH_ID,
        paths,
        vfs,
        sample_ids,
        dict(contig_map or {CHROM: 0}),
        sorted({s for s in sample_ids if s is not None}),
        list(shards),
        pos_batch_size,
        ref_gq_bands,
        tmpdir,
        committed,
        names,
    )
    import json

    mpath = os.path.join(tmpdir, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh)
    L.require_ingest_worker()
    worker = L.rust_worker_path()
    assert worker is not None

    def collect(results: Any, rv: Any = None) -> Any:
        if rv is None:
            return []
        results.append(rv)
        return results

    return ingest.run_rust_shards(worker, mpath, threads, len(shards), collect)


def read_local(path: str) -> Any:
    """A local file as one pyarrow Table: a sample-slice container (its
    blobs concatenated in trailer order) or a plain Parquet file."""
    if path.endswith(".slices"):
        return sample_slices.read_sample_slices(path)
    return pq.read_table(path)


def local_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Every row of ``paths`` (in order) as dicts."""
    out: List[Dict[str, Any]] = []
    for p in paths:
        out += read_local(p).to_pylist()
    return out
