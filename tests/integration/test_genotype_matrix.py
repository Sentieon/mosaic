# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""batch_genotypes and sample_qc against truth built from bcftools output.

One batch of real chr21 DRAGEN gVCFs (the two suite fixtures, plus HG00096
when present: the fixtures carry no ``./.`` ref blocks, HG00096 has many) is
ingested once per module. The truth is computed here, independently of the
ingest code, from ``bcftools query`` over the same subset files:

  * a sample's own variant record at the site gives its GT (alleles named
    with ``truth_alleles.format_alt_allele``, the test-side restatement of
    the encoding rule, pinned by test_allele_encoding; it imports no worker
    code);
  * else, inside the sample's own deletion, ``0/*`` or ``*/*`` by how many GT
    slots carry a deletion allele covering the site (the joint caller's
    convention), with the deletion record's GQ;
  * else the sample's latest record starting at or before the site, if it
    still covers it: a ``0/0`` block is hom-ref, a ``./.`` block a no call
    (both with the block's GQ);
  * else a no call with GQ unknown (255).

GQ is FORMAT/GQ capped at 99, else PL second-smallest minus smallest, else
255. The batch's row set must equal its variant-record positions and
batch_site_info's.
"""

import os
import subprocess
from typing import Dict, List, Optional, Tuple

import pytest

import jc_testlib as L
from mosaic.genotypes import UNRESOLVED, read_genotype_matrix
from truth_alleles import format_alt_allele
from mosaic.init_db import SAMPLE_QC_COLUMNS

pytestmark = pytest.mark.integration

REGION = "chr21:5000000-10000000"
SHARD_SIZE = 1_000_000  # several shard boundaries inside the region
HG00096 = os.path.join(
    L.REPO_ROOT,
    "test_data/input_gvcfs5/HG00096.chr21.final.hard-filtered.gvcf.gz",
)
QUERY = (
    "%POS\t%REF\t%ALT\t%FILTER\t%INFO/END\t%INFO/DP"
    "\t[%GT\t%DP\t%GQ\t%PL\t%MIN_DP]\n"
)
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
HOM_REF = ("0/0", "0|0", "0")
NO_CALL = ("./.", ".|.", ".")


# ── bcftools-side truth ──────────────────────────────────────────────────────


def _int(x: str) -> Optional[int]:
    return None if x == "." else int(x)


class Rec:
    def __init__(self, line: str) -> None:
        f = line.split("\t")
        self.pos = int(f[0])
        self.ref = f[1]
        self.alts = f[2].split(",") if f[2] != "." else []
        self.filter = f[3]
        self.block = self.ref == "." or (
            len(self.alts) == 1 and self.alts[0] in ("<NON_REF>", "*")
        )
        info_end = _int(f[4])
        self.end = (
            info_end if info_end is not None else self.pos + len(self.ref) - 1
        )
        self.info_dp = _int(f[5])
        self.gt = f[6]
        self.dp = _int(f[7])
        gq = _int(f[8])
        pl = [int(x) for x in f[9].split(",")] if f[9] != "." else None
        self.min_dp = _int(f[10])
        if gq is None and pl is not None and len(pl) >= 2:
            lo = sorted(pl)
            gq = lo[1] - lo[0]
        self.gq = 255 if gq is None else max(0, min(99, gq))
        fields = self.gt.replace("|", "/").split("/")
        self.gt_fields = [None if x == "." else int(x) for x in fields]

    def allele(self, x: Optional[int]) -> Optional[str]:
        if x is None:
            return None
        if x == 0:
            return self.ref[0]
        return format_alt_allele(self.ref, self.alts[x - 1])

    def del_ends(self) -> List[int]:
        """Per GT slot: last base its deletion allele deletes (0 if none)."""
        ends = []
        for x in self.gt_fields[:2]:
            e = 0
            if x is not None and x > 0:
                alt = self.alts[x - 1]
                if alt != "<NON_REF>" and len(alt) < len(self.ref):
                    e = self.pos + len(self.ref) - len(alt)
            ends.append(e)
        return ends


def query_records(path: str) -> List[Rec]:
    out = subprocess.run(
        ["bcftools", "query", "-f", QUERY, path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [Rec(line) for line in out.splitlines()]


def truth_cells(
    recs: List[Rec], sites: List[int], site_ref: Dict[int, str]
) -> Dict[int, Tuple]:
    """{site: (a1, a2, gq)} for one sample at the batch's sites (ascending).
    ``site_ref`` is the reference base at each site (allele 0's name)."""
    variant_at = {r.pos: r for r in recs if not r.block}
    deletions = [
        r for r in recs if not r.block and any(e > 0 for e in r.del_ends())
    ]
    out: Dict[int, Tuple] = {}
    i_rec = 0  # recs[:i_rec] start at or before the site
    i_del = 0
    gov: Optional[Rec] = None  # the furthest-reaching deletion so far
    for q in sites:
        while i_rec < len(recs) and recs[i_rec].pos <= q:
            i_rec += 1
        while i_del < len(deletions) and deletions[i_del].pos < q:
            d = deletions[i_del]
            if gov is None or d.end > gov.end:
                gov = d
            i_del += 1
        if q in variant_at:
            r = variant_at[q]
            fields = r.gt_fields + [None]
            out[q] = (r.allele(fields[0]), r.allele(fields[1]), r.gq)
            continue
        if gov is not None:
            ends = gov.del_ends()
            covered = [e >= q for e in ends]
            if any(covered):
                if len(ends) == 1:
                    out[q] = ("*", None, gov.gq)
                elif all(covered):
                    out[q] = ("*", "*", gov.gq)
                else:
                    out[q] = (site_ref[q], "*", gov.gq)
                continue
        last = recs[i_rec - 1] if i_rec else None
        if last is not None and last.end >= q and last.block:
            if last.gt in HOM_REF:
                # A haploid block (GT "0") is (ref, no-call), like a haploid
                # variant record (a, -1).
                second = None if last.gt == "0" else site_ref[q]
                out[q] = (site_ref[q], second, last.gq)
                continue
            if last.gt in NO_CALL:
                out[q] = (None, None, last.gq)
                continue
        out[q] = (None, None, 255)
    return out


def truth_qc(recs: List[Rec]) -> Dict[str, int]:
    """sample_qc counters over every record of one sample's file."""
    q = dict.fromkeys(SAMPLE_QC_COLUMNS, 0)
    for r in recs:
        if r.block:
            n = r.end - r.pos + 1
            if r.gt in HOM_REF:
                q["rb_bases"] += n
                dp = r.min_dp if r.min_dp is not None else r.info_dp
                q["rb_dp_sum"] += (dp or 0) * n
                if r.gq != 255 and r.gq >= 20:
                    q["rb_bases_gq20"] += n
            elif r.gt in NO_CALL:
                q["rb_bases_no_call"] += n
            continue
        q["n_var_records"] += 1
        if r.filter not in ("PASS", "."):
            q["n_var_filtered"] += 1
        g = r.gt_fields
        if all(x is None for x in g):
            q["n_no_call"] += 1
            continue
        if r.dp is not None:
            q["dp_sum_var"] += r.dp
            q["n_dp_var"] += 1
        carried = {
            x
            for x in g
            if x is not None
            and x > 0
            and r.alts[x - 1] not in ("<NON_REF>", "*")
        }
        if len(g) == 1:
            q["n_haploid"] += 1
        elif None not in g:
            if g[0] == g[1]:
                q["n_hom_ref" if g[0] == 0 else "n_hom_alt"] += 1
            elif carried:
                q["n_het"] += 1
        for x in carried:
            alt = r.alts[x - 1]
            t_ref, t_alt = r.ref, alt
            while t_ref and t_alt and t_ref[-1] == t_alt[-1]:
                t_ref, t_alt = t_ref[:-1], t_alt[:-1]
            if len(t_ref) == len(t_alt):
                if len(t_alt) == 1:
                    q["n_snv"] += 1
                    pair = (r.ref[0], t_alt)
                    if set(pair) <= set("ACGT"):
                        q["n_ti" if pair in TRANSITIONS else "n_tv"] += 1
                else:
                    q["n_mnp"] += 1
            elif len(t_alt) > len(t_ref):
                q["n_ins"] += 1
            else:
                q["n_del"] += 1
    return q


def bcftools_psc(path: str) -> Dict[str, int]:
    out = subprocess.run(
        ["bcftools", "stats", "-s", "-", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("PSC\t"):
            f = line.split("\t")
            return {"nTransitions": int(f[6]), "nTransversions": int(f[7])}
    raise AssertionError("no PSC line")


# ── the shared ingest ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def batch(docker_stack, tmp_path_factory):
    """Ingest the region of 2-3 real samples as ONE batch; collect truth."""
    work = str(tmp_path_factory.mktemp("gtm"))
    sources = list(L.GVCFS) + ([HG00096] if os.path.exists(HG00096) else [])
    subsets = []
    for j, src in enumerate(sources):
        dst = os.path.join(work, f"s{j}.g.vcf.gz")
        L.subset_gvcf(src, REGION, dst)
        subsets.append(dst)
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(subsets, region=REGION, shard_size=SHARD_SIZE, threads=2)

    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sample_name, id FROM samples")
            sid_of = dict(cur.fetchall())
        matrix = read_genotype_matrix(L.get_catalog(), conn, 0, 1, 10**9)
        qc = L.sample_qc_rows(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT batch_id, sample_ids FROM ingestion_progress")
            progress = cur.fetchall()
    finally:
        conn.close()
    recs = {}
    for path in subsets:
        name = subprocess.run(
            ["bcftools", "query", "-l", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        recs[sid_of[name]] = (path, query_records(path))
    yield {
        "matrix": matrix,
        "qc": qc,
        "recs": recs,
        "progress": progress,
    }


def test_slot_order_persisted(batch):
    ((batch_id, sample_ids),) = batch["progress"]
    assert sample_ids == sorted(batch["recs"])
    assert batch["matrix"].sample_ids.tolist() == sample_ids
    assert set(batch["matrix"].batch_ids.tolist()) == {batch_id}


def test_matrix_matches_gvcf_truth(batch):
    m = batch["matrix"]
    sites = m.positions.tolist()
    assert len(sites) > 1000
    assert not (m.gt_1 == UNRESOLVED).any(), "one batch resolves every cell"
    cells = L.genotype_matrix_cells(m)
    site_ref = {
        r.pos: r.ref[0]
        for _p, recs in batch["recs"].values()
        for r in recs
        if not r.block
    }
    tally: Dict[str, int] = {}
    mismatches = []
    for sid, (_path, recs) in batch["recs"].items():
        truth = truth_cells(recs, sites, site_ref)
        for q in sites:
            want = truth[q]
            if "*" in want[:2]:
                kind = "star"
            elif want[0] is None:
                kind = "no_call"
            elif want[:2] == (site_ref[q], site_ref[q]):
                kind = "hom_ref"
            else:
                kind = "called"
            tally[kind] = tally.get(kind, 0) + 1
            got = cells[(q, sid)]
            if got != want:
                mismatches.append((sid, q, got, want))
    assert not mismatches, f"{len(mismatches)} cells differ: {mismatches[:10]}"
    # Every rule is exercised (no_call covers ./. blocks and uncovered).
    for kind in ("called", "hom_ref", "star", "no_call"):
        assert tally.get(kind, 0) > 0, tally
    # Allele list shape: ref first, '*' last and only when some slot uses it.
    for i, alleles in enumerate(m.alleles):
        assert "*" not in alleles[:-1]
        if alleles[-1] == "*":
            idx = len(alleles) - 1
            assert (m.gt_1[i] == idx).any() or (m.gt_2[i] == idx).any()


def test_row_set_matches_batch_site_info(batch):
    m = batch["matrix"]
    expected = sorted(
        {
            r.pos
            for _p, recs in batch["recs"].values()
            for r in recs
            if not r.block
        }
    )
    assert m.positions.tolist() == expected
    arrow = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_site_info")
        .scan()
        .to_arrow()
    )
    bsi = set(
        zip(
            arrow.column("batch_id").to_pylist(),
            arrow.column("chromosome_index").to_pylist(),
            arrow.column("position").to_pylist(),
        )
    )
    bgt = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_genotypes")
        .scan()
        .to_arrow()
    )
    keys = list(
        zip(
            bgt.column("batch_id").to_pylist(),
            bgt.column("chromosome_index").to_pylist(),
            bgt.column("position").to_pylist(),
        )
    )
    assert len(keys) == len(set(keys)), "duplicate batch_genotypes rows"
    assert set(keys) == bsi


def test_sample_qc_matches_independent_counts(batch):
    qc = batch["qc"]
    assert len(qc) == len(batch["recs"])
    for sid, (path, recs) in batch["recs"].items():
        ((key, row),) = [(k, v) for k, v in qc.items() if k[1] == sid]
        assert key[2] == 0  # chr21
        got = dict(zip(SAMPLE_QC_COLUMNS, row))
        assert got == truth_qc(recs), f"sample {sid}"
        assert got["n_var_records"] > 1000 and got["rb_bases"] > 1_000_000


def test_sample_qc_tstv_matches_bcftools_stats(batch):
    """Sanity check where the definitions coincide: bcftools PSC
    nTransitions / nTransversions count, per record, the distinct carried
    SNV alleles, as n_ti / n_tv do. (PSC nHets / nNonRefHom are SNP-only
    and nIndels counts records, not alleles, so those are not compared.)"""
    for sid, (path, _recs) in batch["recs"].items():
        ((_, row),) = [(k, v) for k, v in batch["qc"].items() if k[1] == sid]
        got = dict(zip(SAMPLE_QC_COLUMNS, row))
        psc = bcftools_psc(path)
        assert (got["n_ti"], got["n_tv"]) == (
            psc["nTransitions"],
            psc["nTransversions"],
        )


def test_sample_qc_summary_view(batch, docker_stack):
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sample_id, n_ti, n_tv, ti_tv, het_hom, mean_dp_var, "
                "mean_rb_dp FROM sample_qc_summary ORDER BY sample_id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == sorted(batch["recs"])
    for _sid, n_ti, n_tv, ti_tv, het_hom, mean_dp, mean_rb_dp in rows:
        assert ti_tv == pytest.approx(n_ti / n_tv)
        assert 1.0 < ti_tv < 3.0
        assert het_hom > 0 and mean_dp > 0 and mean_rb_dp > 0
