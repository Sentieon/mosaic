# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Binary-level variant-site tests of the Rust ingest worker (K6).

Replaces ``test_allele_catalog.py::
test_sample_site_info_alleles_are_dictionary_encoded_strings``, which
drove the Python worker's ``grouper`` + ``collect_variant_group_data`` +
``bulk_upload`` + ``_write_parquet_local`` on two crafted records. Here the
same two records (a SNP and a deletion of two samples at one site) are
written as gVCFs and run through ``mosaic-ingest-worker``; the local
``sample_site_info`` sample-slice container holds one blob per sample whose
``alleles`` column is ``list<string>``, dictionary-encoded, and holds the
record's own allele list.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sample_slices import read_sample_slice_index
from worker_harness import VRec, requires_worker, run_worker, write_gvcf

pytestmark = [pytest.mark.unit, requires_worker]

SHARD = ("chr21", 0, 1_000_000)


def test_sample_site_info_alleles_are_dictionary_encoded_strings(tmp_path):
    # Python's Rec(100, ...) is 0-based: VCF POS 101.
    snp = VRec(101, "A", ["T", "<NON_REF>"], "0/1")
    dele = VRec(101, "AT", ["A", "<NON_REF>"], "0/1")
    paths = [
        write_gvcf(str(tmp_path / "s1.g.vcf.gz"), "s1", [snp]),
        write_gvcf(str(tmp_path / "s2.g.vcf.gz"), "s2", [dele]),
    ]
    out = tmp_path / "out"
    out.mkdir()
    (res,) = run_worker(paths, [SHARD], str(out), sample_ids=[1, 2])
    (path,) = res["si_paths"]
    # A sample-slice container (E2): one standalone Parquet blob per sample.
    slices = read_sample_slice_index(path)
    assert [sl.sample_id for sl in slices] == [1, 2]
    with open(path, "rb") as f:
        blobs = []
        for sl in slices:
            f.seek(sl.offset)
            blobs.append(f.read(sl.length))
    want = {1: [["A", "xT", "*"]], 2: [["A", "-AT", "*"]]}
    for sl, blob in zip(slices, blobs):
        pf = pq.ParquetFile(pa.BufferReader(blob))
        field = pf.schema_arrow.field("alleles")
        assert field.type == pa.list_(pa.string())
        table = pf.read()
        assert table.column("alleles").to_pylist() == want[sl.sample_id]
        md = pf.metadata.row_group(0)
        col = next(
            md.column(i)
            for i in range(pf.metadata.num_columns)
            if md.column(i).path_in_schema.startswith("alleles")
        )
        assert col.has_dictionary_page
