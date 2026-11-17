// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The shard worker (originally a port of the Python worker's
//! `_shard_worker`): the per-group order of operations, the window
//! flushes, the shard-end writes, the QC roll-up, the stderr lines and the
//! result record.

use std::collections::HashSet;
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Value as Json, json};

use crate::banding::{RefBandSpec, RefBlockBander};
use crate::grouper::Grouper;
use crate::localfiles::{
    batch_key, join, write_bac, write_bgt, write_brb_window, write_bsi, write_rb_container,
    write_si_container,
};
use crate::manifest::{Manifest, Shard};
use crate::matrix::{F_GT, GenotypeRowBuilder, parse_gt};
use crate::partials::{BacRow, BsiRow, SiteBuf, bulk_upload};
use crate::pyops::pv;
use crate::qc::{N_QC, Qc, qc_ref_block, qc_variant_record};
use crate::reader::ShardReader;
use crate::record::Record;
use crate::ref_blocks::{
    DelState, ForeignSpans, NewBlock, RbOut, RbRange, RbRow, collect_sample_ref_block_data,
    update_sample_del_state,
};
use crate::sites::{ParseStats, VarRec, collect_variant_group_data, merge_site_alleles};
use crate::windows::{WinKey, WindowBufs, iceberg_window, is_complete, route_ref_block_output};

/// The batch-wide inputs of every shard (from the manifest).
pub struct Batch {
    pub batch_id: i64,
    pub sample_ids: Vec<Option<i64>>,
    pub slot_sample_ids: Vec<i64>,
    pub sample_names: Vec<String>,
    pub pos_batch_size: i64,
    pub bands: Option<RefBandSpec>,
    pub tmpdir: String,
    pub contig_map: std::collections::HashMap<String, i64>,
    pub committed: HashSet<(String, i64)>,
}

impl Batch {
    pub fn from_manifest(m: &Manifest) -> Result<Batch> {
        Ok(Batch {
            batch_id: m.batch_id,
            sample_ids: m.inputs.iter().map(|i| i.sample_id).collect(),
            slot_sample_ids: m.slot_sample_ids.clone(),
            sample_names: m.sample_names.clone(),
            pos_batch_size: m.pos_batch_size,
            bands: m
                .ref_gq_bands
                .as_deref()
                .map(RefBandSpec::parse)
                .transpose()?,
            tmpdir: m.shard_tmpdir.clone(),
            contig_map: m.contig_map.clone(),
            committed: m.committed_shards.iter().cloned().collect(),
        })
    }
}

/// Seconds per bucket (`--timing`).
#[derive(Debug, Default, Clone, Copy)]
pub struct Timing {
    /// Reading and grouping (inside the grouper).
    pub read_group: f64,
    /// The dense `batch_ref_blocks` windows: sum + write.
    pub dense_write: f64,
    /// The other local files.
    pub other_write: f64,
    /// The whole shard.
    pub total: f64,
}

impl Timing {
    pub fn logic(&self) -> f64 {
        self.total - self.read_group - self.dense_write - self.other_write
    }
}

/// What a shard returns: the result record's fields.
pub struct ShardResult {
    pub json: Json,
    pub timing: Timing,
}

struct Paths {
    rb: Vec<String>,
    si: Vec<String>,
    brb: Vec<String>,
    bsi: Vec<String>,
    bac: Vec<String>,
    bgt: Vec<String>,
}

/// `_write_parquet_local` for the given windows (rb then si per window).
fn write_local(
    b: &Batch,
    bufs: &mut WindowBufs,
    windows: &[WinKey],
    shard_id: &str,
    paths: &mut Paths,
) -> Result<()> {
    for &(wc, ws) in windows {
        let key = batch_key(&b.sample_names, wc, ws);
        if let Some(rows) = bufs.rb.get_mut(&(wc, ws))
            && !rows.is_empty()
        {
            let p = join(&b.tmpdir, &format!("ref_blocks_{shard_id}_{key}.slices"));
            write_rb_container(rows, &p)?;
            paths.rb.push(p);
        }
        if let Some(rows) = bufs.si.get_mut(&(wc, ws))
            && !rows.is_empty()
        {
            let p = join(&b.tmpdir, &format!("site_info_{shard_id}_{key}.slices"));
            write_si_container(rows, &p)?;
            paths.si.push(p);
        }
    }
    Ok(())
}

/// `_flush_local_iceberg_windows` (no-op at `pos_batch_size` 0).
fn flush_local(
    b: &Batch,
    bufs: &mut WindowBufs,
    flush: Option<(i64, i64)>,
    shard_id: &str,
    paths: &mut Paths,
) -> Result<()> {
    let pbs = b.pos_batch_size;
    if pbs == 0 {
        return Ok(());
    }
    let mut all: Vec<WinKey> = bufs.rb.keys().chain(bufs.si.keys()).copied().collect();
    all.sort_unstable();
    all.dedup();
    let done: Vec<WinKey> = all
        .into_iter()
        .filter(|&w| is_complete(w, flush, pbs))
        .collect();
    if !done.is_empty() {
        write_local(b, bufs, &done, shard_id, paths)?;
        for w in &done {
            bufs.rb.remove(w);
            bufs.si.remove(w);
        }
    }
    Ok(())
}

/// `_flush_completed_brb_windows` (no-op at `pos_batch_size` 0).
fn flush_brb(
    b: &Batch,
    bufs: &mut WindowBufs,
    flush: Option<(i64, i64)>,
    shard_id: &str,
    paths: &mut Paths,
) -> Result<()> {
    let pbs = b.pos_batch_size;
    if pbs == 0 {
        return Ok(());
    }
    let done: Vec<WinKey> = bufs
        .brb
        .keys()
        .copied()
        .filter(|&w| is_complete(w, flush, pbs))
        .collect();
    for w in done {
        let seg = bufs.brb.remove(&w).expect("present");
        let p = join(
            &b.tmpdir,
            &format!("batch_ref_blocks_{shard_id}_{}.parquet", w.1),
        );
        if let Some(p) = write_brb_window(&seg, b.batch_id, &p)? {
            paths.brb.push(p);
        }
    }
    Ok(())
}

fn site_names(s: &SiteBuf) -> Vec<String> {
    s.alleles.iter().map(|(a, _)| a.clone()).collect()
}

/// Run one shard. `readers` are the inputs' readers in stream order.
pub fn run_shard(
    b: &Batch,
    readers: Vec<Option<ShardReader>>,
    shard: &Shard,
) -> Result<(ShardResult, Vec<Option<ShardReader>>)> {
    let t_start = Instant::now();
    let mut tm = Timing::default();
    let chrom = shard.chrom.as_str();
    let (shard_start, shard_end) = (shard.start, shard.end);
    let bounds = (shard_start + 1, shard_end);
    let shard_id = format!("{chrom}_{shard_start}");
    let already = b.committed.contains(&(chrom.to_string(), shard_start));
    let c = *b
        .contig_map
        .get(chrom)
        .with_context(|| format!("KeyError: {chrom:?} is not in the contig map"))?;
    let pbs = b.pos_batch_size;
    let n = b.sample_ids.len();
    let sids = &b.sample_ids[..];

    let mut readers = readers;
    for r in readers.iter_mut().flatten() {
        r.start_shard(chrom, shard_start, shard_end);
    }
    let t0 = Instant::now();
    let mut g = Grouper::new(readers)?;
    tm.read_group += t0.elapsed().as_secs_f64();

    let mut last_start = vec![-1i64; n];
    let mut bander = b
        .bands
        .clone()
        .map(|s| RefBlockBander::new(s, pbs, Some(bounds)));
    let mut del_state: Vec<Option<DelState>> = vec![None; n];
    let mut foreign = ForeignSpans::default();
    let mut processed: Vec<Rc<Record>> = Vec::new();
    let mut sites: Vec<SiteBuf> = Vec::new();
    let mut bufs = WindowBufs::default();
    let mut bsi: Vec<BsiRow> = Vec::new();
    let mut bac: Vec<BacRow> = Vec::new();
    let mut paths = Paths {
        rb: vec![],
        si: vec![],
        brb: vec![],
        bsi: vec![],
        bac: vec![],
        bgt: vec![],
    };
    let (mut group_count, mut variant_count, mut ref_block_count) = (0i64, 0i64, 0i64);
    let mut prev_pos: Option<i64> = None;
    let mut stats = ParseStats::default();
    let mut last_flushed: Option<WinKey> = None;
    let mut gt = GenotypeRowBuilder::new(sids, &b.slot_sample_ids, c, b.batch_id);
    let mut qc: Vec<Qc> = vec![[0; N_QC]; n];
    let mut new_blocks: Vec<NewBlock> = Vec::new();
    let mut var_records: Vec<VarRec> = Vec::new();
    let mut rb_rows: Vec<RbRow> = Vec::new();
    let mut rb_partials: Vec<RbRange> = Vec::new();

    loop {
        let t0 = Instant::now();
        let next = g.next_group()?;
        tm.read_group += t0.elapsed().as_secs_f64();
        let Some(pos) = next else { break };
        let grp = &g.grp;
        group_count += 1;
        if prev_pos != Some(pos) {
            processed.clear();
            prev_pos = Some(pos);
        }

        update_sample_del_state(grp, sids, c, &mut del_state)?;

        new_blocks.clear();
        rb_partials.clear();
        rb_rows.clear();
        collect_sample_ref_block_data(
            grp,
            sids,
            c,
            &mut last_start,
            &del_state,
            Some(bounds),
            bander.as_mut(),
            RbOut {
                rows: &mut rb_rows,
                new_blocks: &mut new_blocks,
                partials: &mut rb_partials,
                foreign: &mut foreign,
            },
        )?;
        ref_block_count += route_ref_block_output(&rb_rows, &rb_partials, bounds, pbs, &mut bufs);
        for nb in &new_blocks {
            let info = gt.block_info(nb.k, &nb.v)?;
            qc_ref_block(
                &mut qc[nb.k],
                &nb.v,
                info.3,
                info.2,
                nb.start,
                nb.end,
                bounds.0,
                bounds.1,
            )?;
        }

        if pos < shard_start {
            continue;
        }

        let same_site = sites
            .last()
            .is_some_and(|s| (s.c_idx, s.pos) == (c, pos + 1));
        var_records.clear();
        let seed = if same_site {
            Some(sites.last().expect("same site").alleles.as_slice())
        } else {
            None
        };
        let result = collect_variant_group_data(
            pos,
            grp,
            &mut processed,
            sids,
            &mut stats,
            seed,
            &mut var_records,
            &del_state,
        )?;
        if let Some(gs) = result {
            let key = (c, pos + 1);
            if let Some(last) = sites.last_mut().filter(|s| (s.c_idx, s.pos) == key) {
                last.alleles = merge_site_alleles(&last.alleles, &gs.alleles);
                let (e, nw) = (&mut last.site_info, gs.site_info);
                e.qual += nw.qual;
                e.dp += nw.dp;
                e.mq0 += nw.mq0;
                e.mq1 += nw.mq1;
                for j in 0..4 {
                    e.sb[j] += nw.sb[j];
                }
                for j in 0..3 {
                    e.gtc[j] += nw.gtc[j];
                }
                last.rows.extend(gs.rows);
            } else {
                if gt.pending {
                    let names = site_names(sites.last().expect("a pending site"));
                    gt.end_site(&names)?;
                }
                sites.push(SiteBuf {
                    c_idx: c,
                    pos: pos + 1,
                    alleles: gs.alleles,
                    site_info: gs.site_info,
                    rows: gs.rows,
                });
                variant_count += 1;
                gt.begin_site(pos + 1, grp, &del_state)?;
            }
            for vr in &var_records {
                let gtf = match &vr.v.s0 {
                    Some(s) => parse_gt(pv(&s[F_GT]))?,
                    None => None,
                };
                gt.add_record(vr.k, &vr.v, &vr.fmts, gtf.as_deref())?;
                if vr.v.s0.is_some() {
                    qc_variant_record(&mut qc[vr.k], &vr.v, &vr.fmts, gtf.as_deref())?;
                }
            }
        }

        if pbs > 0 {
            let head = pos + 1;
            let cur = iceberg_window(c, head, pbs);
            if Some(cur) != last_flushed {
                if let Some(bd) = bander.as_mut() {
                    let (rows, ranges) = bd.close_before(c, head);
                    ref_block_count +=
                        route_ref_block_output(&rows, &ranges, bounds, pbs, &mut bufs);
                }
                let t1 = Instant::now();
                flush_local(b, &mut bufs, Some((c, head)), &shard_id, &mut paths)?;
                let t2 = Instant::now();
                flush_brb(b, &mut bufs, Some((c, head)), &shard_id, &mut paths)?;
                tm.other_write += (t2 - t1).as_secs_f64();
                tm.dense_write += t2.elapsed().as_secs_f64();
                last_flushed = Some(cur);
            }
        }
    }
    let readers_back = g.into_sources();

    if gt.pending {
        let names = site_names(sites.last().expect("a pending site"));
        gt.end_site(&names)?;
    }

    let t_stage0 = Instant::now();
    if !sites.is_empty() {
        bulk_upload(&sites, pbs, &mut bufs.si, &mut bsi, &mut bac);
    }
    let t_stage = t_stage0.elapsed().as_secs_f64();
    drop(sites);

    if let Some(bd) = bander.as_mut() {
        let (rows, ranges) = bd.close_all();
        ref_block_count += route_ref_block_output(&rows, &ranges, bounds, pbs, &mut bufs);
    }

    let t1 = Instant::now();
    flush_local(b, &mut bufs, None, &shard_id, &mut paths)?;
    let t2 = Instant::now();
    flush_brb(b, &mut bufs, None, &shard_id, &mut paths)?;
    tm.other_write += (t2 - t1).as_secs_f64();
    tm.dense_write += t2.elapsed().as_secs_f64();

    if pbs == 0 {
        let mut all: Vec<WinKey> = bufs.rb.keys().chain(bufs.si.keys()).copied().collect();
        all.sort_unstable();
        all.dedup();
        let t1 = Instant::now();
        if !all.is_empty() {
            write_local(b, &mut bufs, &all, &shard_id, &mut paths)?;
        }
        let t2 = Instant::now();
        if let Some(seg) = bufs.brb.remove(&(-1, 0)) {
            let p = join(&b.tmpdir, &format!("batch_ref_blocks_{shard_id}_0.parquet"));
            if let Some(p) = write_brb_window(&seg, b.batch_id, &p)? {
                paths.brb.push(p);
            }
        }
        tm.other_write += (t2 - t1).as_secs_f64();
        tm.dense_write += t2.elapsed().as_secs_f64();
    }

    let t1 = Instant::now();
    if !bsi.is_empty() {
        let p = join(&b.tmpdir, &format!("batch_site_info_{shard_id}.parquet"));
        write_bsi(&mut bsi, b.batch_id, &p)?;
        paths.bsi.push(p);
    }
    if !bac.is_empty() {
        let p = join(
            &b.tmpdir,
            &format!("batch_allele_counts_{shard_id}.parquet"),
        );
        write_bac(&mut bac, b.batch_id, &p)?;
        paths.bac.push(p);
    }
    if !gt.rows.is_empty() {
        let p = join(&b.tmpdir, &format!("batch_genotypes_{shard_id}.parquet"));
        write_bgt(&mut gt.rows, b.batch_id, c, &p)?;
        paths.bgt.push(p);
    }
    tm.other_write += t1.elapsed().as_secs_f64();

    // Per-sample QC: all-zero entries dropped, a repeated sample summed.
    let mut qc_order: Vec<i64> = Vec::new();
    let mut qc_sum: std::collections::HashMap<i64, Qc> = std::collections::HashMap::new();
    for (k, counters) in qc.iter().enumerate() {
        let Some(sid) = sids[k] else { continue };
        if counters.iter().all(|&x| x == 0) {
            continue;
        }
        match qc_sum.get_mut(&sid) {
            Some(acc) => {
                for i in 0..N_QC {
                    acc[i] += counters[i];
                }
            }
            None => {
                qc_order.push(sid);
                qc_sum.insert(sid, *counters);
            }
        }
    }

    let mut records_dropped = 0;
    let mut readers_back = readers_back;
    for r in readers_back.iter_mut().flatten() {
        records_dropped += r.records_dropped;
        r.records_dropped = 0;
    }
    stats.records_dropped += records_dropped;
    stats.gt_matrix_conflicts += gt.conflicts;

    tm.total = t_start.elapsed().as_secs_f64();
    if group_count > 0 {
        let elapsed = t_start.elapsed().as_secs_f64();
        let re = if already {
            " (reprocessed for Iceberg; checkpointed by an earlier attempt)"
        } else {
            ""
        };
        eprintln!(
            "  Shard {chrom}:{shard_start}-{shard_end}: {group_count} groups, \
             {variant_count} variants, {ref_block_count} ref-blocks in {elapsed:.1}s \
             (stage {t_stage:.2}s){re}"
        );
    }
    if !stats.is_empty() {
        eprintln!(
            "  WARNING shard {chrom}:{shard_start}-{shard_end} handled bad input: \
             {} records dropped, {} site_info fields and {} genotype fields \
             unparseable (counted as 0), {} genotype-matrix slot conflicts \
             (second record at one site dropped)",
            stats.records_dropped,
            stats.site_info_parse_errors,
            stats.gt_parse_errors,
            stats.gt_matrix_conflicts
        );
    }

    let mut ps = serde_json::Map::new();
    for (k, v) in [
        ("records_dropped", stats.records_dropped),
        ("site_info_parse_errors", stats.site_info_parse_errors),
        ("gt_parse_errors", stats.gt_parse_errors),
        ("gt_matrix_conflicts", stats.gt_matrix_conflicts),
    ] {
        if v > 0 {
            ps.insert(k.to_string(), json!(v));
        }
    }
    let sample_qc: Vec<Json> = qc_order
        .iter()
        .map(|sid| json!([sid, c, qc_sum[sid].to_vec()]))
        .collect();
    let fds: Vec<Json> = foreign
        .order
        .iter()
        .map(|key| json!([key.0, key.1, key.2, foreign.map[key]]))
        .collect();
    let out = json!({
        "kind": "shard",
        "index": shard.index,
        "shard_chrom": chrom,
        "shard_start": shard_start,
        "rb_paths": paths.rb,
        "si_paths": paths.si,
        "brb_paths": paths.brb,
        "bsi_paths": paths.bsi,
        "bac_paths": paths.bac,
        "bgt_paths": paths.bgt,
        "sample_qc": sample_qc,
        "variant_count": variant_count,
        "ref_block_count": ref_block_count,
        "group_count": group_count,
        "parse_stats": Json::Object(ps),
        "foreign_del_spans": fds,
    });
    Ok((
        ShardResult {
            json: out,
            timing: tm,
        },
        readers_back,
    ))
}
