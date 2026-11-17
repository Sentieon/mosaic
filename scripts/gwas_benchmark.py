# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""
Benchmark Iceberg query performance with a GWAS-style allele-frequency calculation.

For each site where any alt allele has global AF above a user-supplied threshold,
the script recalculates allele frequencies for a user-specified sample subset by
querying Iceberg, then prints one output row per alt allele.

The results are not used for any downstream analysis; the goal is to measure
Iceberg IO at different levels of sample batching.

A tombstoned sample (samples.status = 'removed', mosaic/tombstones.py) is left
out of the subset and its rows are dropped from the subset recount. The
global AF comes from the PostgreSQL aggregates, so its denominator counts the
samples the latest complete aggregation run counted: those of its ingest
batches less those of its removal batches (`mosaic remove` + `mosaic
aggregate`). A sample
tombstoned by `mosaic remove` but not yet aggregated out is still counted.

Usage:
    python scripts/gwas_benchmark.py --af-cutoff 0.01 --samples HG00245 HG00420

Output (tab-separated, to stdout):
    CHROM  POS  REF  ALT  GLOBAL_AF  SUBSET_AF
"""

import argparse
import sys
import time
from collections import defaultdict

import psycopg2
import psycopg2.extras

from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import And, EqualTo, In


from mosaic.config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
)
from mosaic.tombstones import (
    and_filter,
    missing_status_remedy,
    removed_sample_ids,
    samples_status_missing,
    tombstone_filter,
)

# ========== Allele helpers ==========


def decode_allele(db_allele, base_ref):
    """Return (ref_str, alt_str) for a DB-encoded non-ref allele."""
    if db_allele.startswith("x"):
        return base_ref, db_allele[1:]  # SNP / MNP
    elif db_allele.startswith("+"):
        return base_ref, base_ref + db_allele[1:]  # insertion
    elif db_allele.startswith("-"):
        return base_ref + db_allele[1:], base_ref  # deletion
    else:  # '*'
        return base_ref, "*"


# ========== Main ==========


def main():
    parser = argparse.ArgumentParser(
        description=(
            "GWAS-style AF benchmark: select high-AF sites from the DB, "
            "recalculate AF for a sample subset via Iceberg queries."
        )
    )
    parser.add_argument(
        "--af-cutoff",
        type=float,
        required=True,
        help="Minimum global AF for a site to be included (e.g. 0.01)",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="Sample names forming the subset for AF recalculation",
    )
    args = parser.parse_args()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------
    conn = psycopg2.connect(**DB_CONFIG)
    if samples_status_missing(conn):
        print(f"Error: {missing_status_remedy()}", file=sys.stderr)
        sys.exit(1)
    catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
    site_info_table = catalog.load_table(
        f"{ICEBERG_NAMESPACE}.sample_site_info"
    )

    cur = conn.cursor()

    # ------------------------------------------------------------------
    # Total sample count → global AN denominator: the samples the global
    # allele counts (PG alleles.count) include, i.e. those of the latest
    # COMPLETE aggregation run's ingest batches less those of its removal
    # batches (a sample tombstoned but not yet aggregated out is still
    # counted; a failed or running run's partitions are not trusted).
    # Every sample when no run has completed.
    # ------------------------------------------------------------------
    cur.execute(
        "SELECT p.kind, p.sample_names FROM ingestion_progress p "
        "WHERE p.batch_id = ANY(COALESCE((SELECT batch_ids FROM "
        "aggregation_progress WHERE status = 'complete' "
        "ORDER BY agg_run_id DESC LIMIT 1), '{}'::int[]))"
    )
    counted, subtracted = set(), set()
    for kind, names in cur.fetchall():
        (subtracted if kind == "removal" else counted).update(names)
    if counted:
        total_samples = len(counted - subtracted)
    else:
        cur.execute("SELECT COUNT(*) FROM samples")
        total_samples = cur.fetchone()[0]
    global_an = 2 * total_samples
    print(
        f"Total samples in DB: {total_samples}  (global AN={global_an})",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Resolve subset sample names → IDs
    # ------------------------------------------------------------------
    cur.execute(
        "SELECT id, sample_name, status FROM samples "
        "WHERE sample_name = ANY(%s)",
        (args.samples,),
    )
    rows = cur.fetchall()
    subset_id_to_name = {
        sid: name for sid, name, status in rows if status == "active"
    }
    found_names = {name for _, name, _status in rows}
    missing = set(args.samples) - found_names
    if missing:
        print(
            f"Warning: samples not found in DB: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )
    tombstoned = sorted(name for _, name, status in rows if status != "active")
    if tombstoned:
        print(
            "Warning: tombstoned samples left out of the subset: "
            f"{', '.join(tombstoned)}",
            file=sys.stderr,
        )
    # Rows of tombstoned samples are dropped from every scan (the subset
    # filter already excludes them; this also covers an empty subset).
    tomb_pushdown, tomb_mask = tombstone_filter(removed_sample_ids(conn))

    subset_ids = list(subset_id_to_name.keys())
    subset_an = 2 * len(subset_ids)
    print(
        f"Subset size: {len(subset_ids)} samples  (subset AN={subset_an})",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Select sites where any alt allele exceeds the AF cutoff.
    # Global AF ≈ alleles.count / global_an  (ref-block counts not included,
    # so this is an approximation — sufficient for site selection).
    # ------------------------------------------------------------------
    count_threshold = args.af_cutoff * global_an
    print(
        f"AF cutoff={args.af_cutoff}  →  selecting sites with any alt "
        f"count > {count_threshold:.1f}",
        file=sys.stderr,
    )

    cur.execute(
        """
        SELECT DISTINCT a.chromosome_index, a.position, c.chromosome
        FROM alleles a
        JOIN contigs c ON a.chromosome_index = c.id
        WHERE (a.allele LIKE 'x%%' OR a.allele LIKE '+%%' OR a.allele LIKE '-%%')
          AND a.count > %s
        ORDER BY a.chromosome_index, a.position
        """,
        (count_threshold,),
    )
    selected_sites = cur.fetchall()  # (chrom_idx, pos, chrom)
    print(
        f"Selected {len(selected_sites)} sites above AF cutoff",
        file=sys.stderr,
    )

    if not selected_sites:
        print("No sites selected. Exiting.", file=sys.stderr)
        cur.close()
        conn.close()
        return

    # ------------------------------------------------------------------
    # Fetch all alleles for selected sites in one query.
    # Use a temp table to look up (chromosome_index, position) pairs.
    # ------------------------------------------------------------------
    cur.execute("""CREATE TEMPORARY TABLE lookup_site_keys
                  (chrom_idx INTEGER, pos INTEGER)""")
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO lookup_site_keys (chrom_idx, pos) VALUES %s",
        [(row[0], row[1]) for row in selected_sites],
    )
    cur.execute("""
        SELECT a.chromosome_index, a.position, a.allele, a.count
        FROM alleles a
        JOIN lookup_site_keys l
          ON a.chromosome_index = l.chrom_idx AND a.position = l.pos
        ORDER BY a.chromosome_index, a.position, a.allele COLLATE "C"
        """)
    # pos_alleles: (chrom_idx, pos) → [(allele_str, allele_count)]
    pos_alleles = defaultdict(list)
    for ci, pos, allele_str, allele_count in cur.fetchall():
        pos_alleles[(ci, pos)].append((allele_str, allele_count))
    cur.execute("DROP TABLE lookup_site_keys")

    cur.close()

    # Group selected sites by chromosome for efficient Iceberg queries.
    # chrom_sites: chrom_idx → [(pos, chrom_name)]
    chrom_sites = defaultdict(list)
    for chrom_idx, pos, chrom in selected_sites:
        chrom_sites[chrom_idx].append((pos, chrom))

    # ------------------------------------------------------------------
    # Stream results: single Iceberg scan per chromosome using Arrow
    # batch reader (constant memory, single S3 pass).
    # ------------------------------------------------------------------
    print("CHROM\tPOS\tREF\tALT\tGLOBAL_AF\tSUBSET_AF")

    total_iceberg_rows = 0
    t_iceberg = 0.0

    for chrom_idx in sorted(chrom_sites.keys()):
        sites_on_chrom = chrom_sites[chrom_idx]
        selected_positions = set(s[0] for s in sites_on_chrom)

        # --------------------------------------------------------------
        # Single streaming scan: subset samples × entire chromosome.
        # Only fetch the 4 columns needed for AF counting: each row's
        # allele strings and its GT as local indices into them.
        # --------------------------------------------------------------
        t0 = time.time()
        row_filter = EqualTo("chromosome_index", int(chrom_idx))
        if subset_ids:
            row_filter = And(
                row_filter, In("sample_id", [int(s) for s in subset_ids])
            )
        fields = ("position", "alleles", "gt_1", "gt_2")
        if tomb_mask is not None:
            fields += ("sample_id",)

        reader = site_info_table.scan(
            row_filter=and_filter(row_filter, tomb_pushdown),
            selected_fields=fields,
        ).to_arrow_batch_reader()

        # Accumulate allele counts directly: pos → {allele_str → count}
        subset_counts = defaultdict(lambda: defaultdict(int))
        chrom_iceberg_rows = 0
        for batch in reader:
            if tomb_mask is not None:
                batch = tomb_mask(batch)
            positions = batch.column("position").to_pylist()
            row_alleles = batch.column("alleles").to_pylist()
            gt1s = batch.column("gt_1").to_pylist()
            gt2s = batch.column("gt_2").to_pylist()

            for pos, alleles, g1, g2 in zip(
                positions, row_alleles, gt1s, gt2s
            ):
                if pos in selected_positions:
                    if g1 is not None:
                        subset_counts[pos][alleles[g1]] += 1
                    if g2 is not None:
                        subset_counts[pos][alleles[g2]] += 1
                    chrom_iceberg_rows += 1

        t_iceberg += time.time() - t0
        total_iceberg_rows += chrom_iceberg_rows
        print(
            f"  Chromosome {chrom_idx}: scanned {chrom_iceberg_rows} matching rows "
            f"in {time.time() - t0:.1f}s",
            file=sys.stderr,
        )

        # --------------------------------------------------------------
        # Emit one output row per alt allele per site
        # --------------------------------------------------------------
        for pos, chrom_name in sites_on_chrom:
            alleles = pos_alleles[(chrom_idx, pos)]

            # Identify the reference allele string
            base_ref = next(
                (
                    a_str
                    for a_str, _ in alleles
                    if not a_str.startswith(("x", "+", "-", "*"))
                ),
                None,
            )
            if not base_ref:
                continue

            # Allele counts already accumulated during scan
            counts_at_pos = subset_counts.get(pos, {})

            for a_str, allele_count in alleles:
                if not a_str.startswith(("x", "+", "-")):
                    continue  # skip ref and '*' alleles

                ref_str, alt_str = decode_allele(a_str, base_ref)
                global_af = allele_count / global_an if global_an > 0 else 0.0
                subset_ac = counts_at_pos.get(a_str, 0)
                subset_af = subset_ac / subset_an if subset_an > 0 else 0.0

                print(
                    f"{chrom_name}\t{pos}\t{ref_str}\t{alt_str}\t"
                    f"{global_af:.6f}\t{subset_af:.6f}"
                )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(
        f"\nDone: {len(selected_sites)} sites, "
        f"{len(chrom_sites)} chromosome scan(s), "
        f"{total_iceberg_rows} matching Iceberg rows, "
        f"{t_iceberg:.2f}s Iceberg IO, {elapsed:.2f}s total",
        file=sys.stderr,
    )

    conn.close()


if __name__ == "__main__":
    main()
