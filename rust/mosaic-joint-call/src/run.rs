// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The calling run: the units of the manifest's contigs, handed out in
//! genomic order to `--threads` threads, each
//! from its own inputs with a lookback and no state from the unit before
//! it, each compressing its own text on its thread and handing it to the
//! ordered writer (`ordered::OrderedWriter`: the head unit writes the
//! output as it goes, the others buffer up to a bound and then spill, and
//! are copied in unit order); the main thread prints one stderr line per
//! unit in unit order and one JSON summary line on stdout.
//!
//! A unit `[ws, we]` reads: the `sites` and `alleles` exports' rows in it;
//! the deletion lookback (the catalog rows at the positions below `ws` that
//! hold a deletion with `del_end > ws`); the `sample_site_info` rows from
//! the lookback's first position (rows below `ws` are used only as
//! deletion-site rows); the `sample_ref_blocks` rows overlapping it. The
//! site loop advances the alleles cursor, the site rows and the ref blocks
//! to each site in turn, keeping per sample its current block, the active
//! deletions (`del_end` past the site) and each active deletion site's
//! rows.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write as _;
use std::ops::ControlFlow;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use futures::{StreamExt, TryStreamExt};
use serde::Serialize;

use crate::call::{
    CapCounts, DelSite, Deletion, Restrict, SiteInputs, TrimCounts, genotype_allele_counts,
    is_nonref, process_variant_site, restrict_site,
};
use crate::count::{Unit, commas, plan_units};
use crate::exports::{AlleleCursor, CatalogAllele, Export, SiteCursor, lookback};
use crate::failhook::{FAIL_HOOK_ENV, FailUnit};
use crate::fisher::Fisher;
use crate::format::{Call, generate_sample_call};
use crate::manifest::Manifest;
use crate::ordered::{DEFAULT_UNIT_BUFFER_BYTES, OrderedWriter};
use crate::reader::{Counters, Mask, Merge, OpenFile, Row, Table};
use crate::rows::{Block, BlockCols, SiteRow, site_row};
use crate::store::Stores;
use crate::writer::{Output, Record, format_info, format_record, header_text};

/// The catalog cross-check's counts (J0's `CatalogCrossCheck`): records
/// whose catalog non-ref allele counts (the `*` ALT aside) differ from the
/// emitted AC (`catalog_ac_differs`), per reason: `stale` (the run's
/// reason, when the run is stale), else `unresolved` (a sample's own call
/// written `./.` while both its GT indices name alleles of its row:
/// `own_call_unresolved`) or `other`. A tombstoned sample is no reason of
/// its own: before the aggregate
/// that applies its removal the run is stale, and after it the catalog no
/// longer counts the sample.
#[derive(Debug, Default, Clone, Serialize)]
pub struct CrossCheck {
    pub records: u64,
    pub differing: BTreeMap<String, u64>,
    pub first: Vec<String>,
}

const CROSS_CHECK_LIMIT: usize = 10;

/// Whether a record's catalog AC differs from its emitted AC over its
/// ALTs, the `*` ALT aside (its catalog count is the star count the
/// record itself derives).
pub fn catalog_ac_differs(alts: &[String], catalog_ac: &[i64], emitted_ac: &[i64]) -> bool {
    alts.iter()
        .zip(catalog_ac.iter().zip(emitted_ac))
        .any(|(a, (c, e))| a != "*" && c != e)
}

/// Whether a sample's own call at the record is written `./.` although
/// both GT indices of its row name alleles of the row (an allele of its
/// genotype, e.g. its <NON_REF>, is no VCF allele of the record).
/// `own[c]` is column `c`'s own row at the site.
pub fn own_call_unresolved(calls: &[Call], own: &[Option<&SiteRow>]) -> bool {
    calls.iter().zip(own).any(|(c, o)| {
        c.gt == "./."
            && o.is_some_and(|r| r.gt_allele(r.gt_1).is_some() && r.gt_allele(r.gt_2).is_some())
    })
}

impl CrossCheck {
    fn check(&mut self, run_reason: Option<&str>, site: String, differs: bool, unresolved: bool) {
        self.records += 1;
        if !differs {
            return;
        }
        let reason = run_reason.unwrap_or(if unresolved { "unresolved" } else { "other" });
        *self.differing.entry(reason.to_string()).or_default() += 1;
        if self.first.len() < CROSS_CHECK_LIMIT {
            self.first.push(site);
        }
    }

    fn add(&mut self, o: &CrossCheck) {
        self.records += o.records;
        for (k, v) in &o.differing {
            *self.differing.entry(k.clone()).or_default() += v;
        }
        for s in &o.first {
            if self.first.len() < CROSS_CHECK_LIMIT {
                self.first.push(s.clone());
            }
        }
    }
}

/// Seconds per phase of a unit.
#[derive(Debug, Default, Clone, Serialize)]
pub struct Phases {
    /// The exports: the lookback and the cursors.
    pub exports: f64,
    /// Advancing the `sample_site_info` merge (decode included).
    pub site_rows: f64,
    /// Advancing the `sample_ref_blocks` merge.
    pub ref_blocks: f64,
    /// The site logic and the sample calls.
    pub calls: f64,
    /// The record text.
    pub format: f64,
    /// Compressing the unit's text and handing it to the ordered writer
    /// (written, buffered or spilled).
    pub write: f64,
}

impl Phases {
    fn add(&mut self, o: &Phases) {
        self.exports += o.exports;
        self.site_rows += o.site_rows;
        self.ref_blocks += o.ref_blocks;
        self.calls += o.calls;
        self.format += o.format;
        self.write += o.write;
    }
}

/// A unit's text is handed to the writer whenever it reaches this many
/// bytes (and at the unit's end), so a unit's memory does not grow with
/// samples × sites.
pub const TEXT_CHUNK: usize = 4 << 20;

/// What a unit produced (its text went to the sink).
pub struct UnitResult {
    pub records: u64,
    pub skipped: u64,
    pub warnings: Vec<String>,
    pub cross_check: CrossCheck,
    pub phases: Phases,
    pub site_rows: u64,
    pub ref_rows: u64,
    pub deletions: u64,
    /// The ALT cap's counts (K3).
    pub cap: CapCounts,
    /// The trim's counts (K5).
    pub trim: TrimCounts,
    /// The records `--min-ac` left out (K5).
    pub min_ac_left_out: u64,
}

/// The run's fixed inputs.
pub struct RunCtx<'a> {
    pub m: &'a Manifest,
    /// Header column of each sample id the run writes: every active
    /// sample, or the subset (K4).
    pub column: HashMap<i32, usize>,
    /// The reader's positive mask (since the sample subset): a
    /// per-sample row is kept iff its sample has a header column.
    pub mask: Mask,
    /// The cross-check's run reason: `stale`, or none.
    pub run_reason: Option<&'static str>,
    /// `--max-alt-alleles` (K3; 0 = no cap): `RunOpts::max_alt_alleles`.
    pub cap: usize,
    /// `--trim-alt-alleles` (K5): `RunOpts::trim_alt_alleles`.
    pub trim: bool,
    /// `--min-ac` (K5; 0 = off): `RunOpts::min_ac`.
    pub min_ac: u64,
}

impl<'a> RunCtx<'a> {
    pub fn new(m: &'a Manifest) -> RunCtx<'a> {
        RunCtx {
            m,
            column: m
                .samples
                .iter()
                .enumerate()
                .map(|(i, s)| (s.id, i))
                .collect(),
            mask: Mask::Keep(Arc::new(m.samples.iter().map(|s| s.id).collect())),
            run_reason: m.stale.then_some("stale"),
            cap: 0,
            trim: false,
            min_ac: 0,
        }
    }
}

/// One table's merge with a one-row look-ahead (the row stays valid in
/// its batch until the merge advances).
struct Stream {
    merge: Merge,
    peeked: Option<Row>,
    done: bool,
    rows: u64,
}

impl Stream {
    fn new(merge: Merge) -> Stream {
        Stream {
            merge,
            peeked: None,
            done: false,
            rows: 0,
        }
    }

    async fn peek(&mut self) -> Result<Option<Row>> {
        if self.peeked.is_none() && !self.done {
            self.peeked = self.merge.next().await?;
            self.done = self.peeked.is_none();
        }
        Ok(self.peeked)
    }

    /// Take the peeked row whose key is at most `max_key`.
    async fn next_upto(&mut self, max_key: i64) -> Result<Option<Row>> {
        match self.peek().await? {
            Some(r) if i64::from(r.key) <= max_key => {
                self.peeked = None;
                self.rows += 1;
                Ok(Some(r))
            }
            _ => Ok(None),
        }
    }
}

/// The unit's site-loop state.
struct State {
    /// The active deletions, `del_start` order.
    deletions: Vec<Deletion>,
    /// Per active deletion site: its non-ref set and rows.
    del_sites: HashMap<i32, DelSite>,
    /// Per header column: the last block seen with `start ≤` the site.
    blocks: Vec<Option<Block>>,
}

impl State {
    /// Register a catalog position's rows: its deletion alleles join the
    /// active list and the site gets its non-ref set (rows come later).
    fn register(&mut self, group: &[CatalogAllele]) {
        let dels: Vec<Deletion> = group
            .iter()
            .filter_map(|r| {
                r.del.map(|(s, e)| Deletion {
                    pos: r.position,
                    allele: r.allele.clone(),
                    del_start: s,
                    del_end: e,
                })
            })
            .collect();
        if dels.is_empty() {
            return;
        }
        let pos = group[0].position;
        let nonref = group
            .iter()
            .filter(|r| is_nonref(&r.allele))
            .map(|r| r.allele.clone())
            .collect();
        self.del_sites.insert(
            pos,
            DelSite {
                nonref,
                rows: Vec::new(),
            },
        );
        self.deletions.extend(dels);
    }

    /// Drop the deletions that end at or before `pos` (they span no site
    /// from here on) and the sites left without one.
    fn prune(&mut self, pos: i32) {
        let before = self.deletions.len();
        self.deletions.retain(|d| d.del_end > pos);
        if self.deletions.len() != before {
            let live: HashSet<i32> = self.deletions.iter().map(|d| d.pos).collect();
            self.del_sites.retain(|p, _| live.contains(p));
        }
    }
}

/// Run one unit: its records' text and counts.
pub async fn run_unit(
    ctx: &RunCtx<'_>,
    unit: &Unit,
    files: [&[Arc<OpenFile>]; 2],
    counters: [Arc<Counters>; 2],
    fisher: &mut Fisher,
    sink: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<UnitResult> {
    let contig = &ctx.m.contigs[unit.contig];
    let (ws, we) = (unit.ws, unit.we);
    let mut ph = Phases::default();
    let t = Instant::now();
    let sites_path = contig.sites.as_deref().expect("validated");
    let alleles_path = contig.alleles.as_deref().expect("validated");
    let sites_export = Export::open_sites(sites_path)?;
    let alleles_export = Export::open_alleles(alleles_path)?;
    let mut sites = SiteCursor::new(&sites_export, ws, we)?;
    let mut catalog = AlleleCursor::new(&alleles_export, ws, we)?;
    // The lookback: the deletions from below ws that may span the unit.
    let mut st = State {
        deletions: Vec::new(),
        del_sites: HashMap::new(),
        blocks: vec![None; ctx.m.samples.len()],
    };
    let back = lookback(&alleles_export, ws)?;
    let mut lb = ws;
    for group in back.chunk_by(|a, b| a.position == b.position) {
        st.register(group);
        lb = lb.min(i64::from(group[0].position));
    }
    st.prune(i32::try_from(ws).unwrap_or(i32::MAX));
    ph.exports += t.elapsed().as_secs_f64();

    let mut site_stream = Stream::new(Merge::start(
        Table::SiteInfo,
        files[0],
        lb,
        we,
        ctx.mask.clone(),
        counters[0].clone(),
    ));
    let mut ref_stream = Stream::new(Merge::start(
        Table::RefBlocks,
        files[1],
        ws,
        we,
        ctx.mask.clone(),
        counters[1].clone(),
    ));
    let n = ctx.m.samples.len();
    let mut text = String::new();
    let mut out = UnitResult {
        records: 0,
        skipped: 0,
        warnings: Vec::new(),
        cross_check: CrossCheck::default(),
        phases: Phases::default(),
        site_rows: 0,
        ref_rows: 0,
        deletions: 0,
        cap: CapCounts::default(),
        trim: TrimCounts::default(),
        min_ac_left_out: 0,
    };
    let mut deletions_seen: u64 = st.deletions.len() as u64;
    let mut own: Vec<Option<SiteRow>> = vec![None; n];
    // Per file: the current batch's typed columns (by its seq).
    let mut block_cols: Vec<Option<(u64, BlockCols)>> = Vec::new();
    let chrom = contig.name.as_str();
    loop {
        let t = Instant::now();
        let Some(site) = sites.next_site()? else {
            ph.exports += t.elapsed().as_secs_f64();
            break;
        };
        let p = site.position;
        let p64 = i64::from(p);
        // 1. The catalog through p: the positions below register their
        // deletions; p's rows are the site's catalog (its own deletions
        // registered too: they never span p itself).
        let mut site_catalog: Vec<CatalogAllele> = Vec::new();
        while let Some(group) = catalog.next_group(p64)? {
            let before = st.deletions.len();
            st.register(&group);
            deletions_seen += (st.deletions.len() - before) as u64;
            if group[0].position == p {
                site_catalog = group;
            }
        }
        ph.exports += t.elapsed().as_secs_f64();

        // 2. The sample_site_info rows through p: at a deletion site, kept
        // as its rows; at p, the samples' own rows (the last row wins).
        let t = Instant::now();
        for o in own.iter_mut() {
            *o = None;
        }
        while let Some(r) = site_stream.next_upto(p64).await? {
            let at_p = r.key == p;
            let at_del = st.del_sites.contains_key(&r.key);
            if !at_p && !at_del {
                continue;
            }
            let row = site_row(site_stream.merge.batch(&r), r.row)
                .with_context(|| format!("sample_site_info at {chrom}:{}", r.key))?;
            if at_del && let Some(ds) = st.del_sites.get_mut(&r.key) {
                ds.rows.push(row.clone());
            }
            if at_p && let Some(&c) = ctx.column.get(&row.sample_id) {
                own[c] = Some(row);
            }
        }
        ph.site_rows += t.elapsed().as_secs_f64();

        // 3. The ref blocks starting at or before p.
        let t = Instant::now();
        while let Some(r) = ref_stream.next_upto(p64).await? {
            if let Some(&c) = ctx.column.get(&r.sample_id) {
                if block_cols.len() <= r.file {
                    block_cols.resize_with(r.file + 1, || None);
                }
                if block_cols[r.file]
                    .as_ref()
                    .is_none_or(|(seq, _)| *seq != r.seq)
                {
                    block_cols[r.file] = Some((r.seq, BlockCols::new(ref_stream.merge.batch(&r))?));
                }
                let cols = &block_cols[r.file].as_ref().expect("just set").1;
                st.blocks[c] = Some(
                    cols.block(r.row)
                        .with_context(|| format!("sample_ref_blocks at {chrom}:{}", r.key))?,
                );
            }
        }
        ph.ref_blocks += t.elapsed().as_secs_f64();

        // 4. The site.
        let t = Instant::now();
        st.prune(p);
        let own_refs: Vec<Option<&SiteRow>> = own.iter().map(Option::as_ref).collect();
        let blocks: Vec<Option<Block>> = st
            .blocks
            .iter()
            .map(|b| b.filter(|b| b.start <= p && p <= b.end))
            .collect();
        let inputs = SiteInputs {
            catalog: &site_catalog,
            own: &own_refs,
            deletions: &st.deletions,
            del_sites: &st.del_sites,
            column: &ctx.column,
        };
        let Some(mut sc) = process_variant_site(&site, &inputs, fisher) else {
            out.warnings
                .push(format!("Warning: No base ref found for {chrom}:{p}"));
            out.skipped += 1;
            ph.calls += t.elapsed().as_secs_f64();
            continue;
        };
        // The trim, then the ALT cap (K5, K3): the site's alleles
        // restricted to REF and the kept ALTs before any call is built;
        // under --min-ac a record whose written ALT observations fall below
        // it is left out here, before its calls are built (its trim and cap
        // counts are counted: they are the projections' of the record).
        let r = restrict_site(
            &mut sc,
            &own_refs,
            Restrict {
                trim: ctx.trim,
                cap: ctx.cap,
                written_ac: ctx.min_ac > 0,
            },
        );
        out.trim.add(&r.trim);
        if let Some(cc) = &r.cap {
            out.cap.add(cc);
        }
        if r.written_ac.is_some_and(|w| w < ctx.min_ac) {
            out.min_ac_left_out += 1;
            ph.calls += t.elapsed().as_secs_f64();
            continue;
        }
        let calls: Vec<Call> = (0..n)
            .map(|c| {
                let star = sc.stars[c].as_ref();
                let nonref = star
                    .and_then(|s| st.del_sites.get(&s.site))
                    .map(|d| &d.nonref);
                generate_sample_call(own_refs[c], star, blocks[c].as_ref(), &sc, nonref)
            })
            .collect();
        let gts: Vec<&str> = calls.iter().map(|c| c.gt.as_str()).collect();
        let (ac, an, af) = genotype_allele_counts(&gts, sc.final_alts.len());
        // A capped record is not compared: a no-called carrier lowers the
        // emitted AC of a kept allele it also carried. Under a subset (K4)
        // no record is compared: the catalog's counts are the whole
        // cohort's.
        if r.cap.is_none() && !ctx.m.subset {
            let unresolved = own_call_unresolved(&calls, &own_refs);
            // A trimmed record (K5) on its untrimmed alleles: the ranking
            // pass's AC, which the untrimmed calls would write.
            let differs = match &r.untrimmed {
                Some((alts, pass_ac)) => catalog_ac_differs(alts, &sc.catalog_ac, pass_ac),
                None => catalog_ac_differs(&sc.final_alts, &sc.catalog_ac, &ac),
            };
            out.cross_check
                .check(ctx.run_reason, format!("{chrom}:{p}"), differs, unresolved);
        }
        ph.calls += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let info = format_info(&ac, an, &af, &sc.info);
        let rec = Record {
            chrom,
            pos: p,
            reference: &sc.max_ref,
            alts: &sc.final_alts,
            qual: site.qual,
            info: &info,
        };
        format_record(&mut text, &rec, &calls);
        out.records += 1;
        ph.format += t.elapsed().as_secs_f64();
        if text.len() >= TEXT_CHUNK {
            let t = Instant::now();
            sink(&text)?;
            text.clear();
            ph.write += t.elapsed().as_secs_f64();
        }
    }
    let t = Instant::now();
    if !text.is_empty() {
        sink(&text)?;
    }
    ph.write += t.elapsed().as_secs_f64();
    out.site_rows = site_stream.rows;
    out.ref_rows = ref_stream.rows;
    out.deletions = deletions_seen;
    out.phases = ph;
    Ok(out)
}

#[derive(Serialize)]
struct TableSummary {
    files: usize,
    file_bytes: u64,
    footer_bytes: u64,
    rows_merged: u64,
    bytes_requested: u64,
    bytes_fetched: u64,
    requests: u64,
}

/// How many units may be in flight past the first unwritten one, per
/// thread (J3's gate): unit `i` starts only while `i < next + HELD_PER_THREAD
/// × threads`, `next` the first unit not yet written. So at most
/// `2 × threads − 1` finished units await writing (the first unwritten unit
/// is never held: it is the head, which writes as it goes; K2), and that
/// unit, already started (units start in index order), always runs: the
/// gate cannot deadlock. The ordered writer advances it (K2), so it also
/// bounds the buffers and spill files of the units past the head.
pub const HELD_PER_THREAD: usize = 2;

/// The gate on starting a unit (J3; the ingest worker's R5 `pending_files`
/// gate in units).
pub(crate) struct UnitGate {
    bound: usize,
    st: Mutex<(usize, bool)>,
    cv: Condvar,
}

impl UnitGate {
    pub(crate) fn new(bound: usize) -> UnitGate {
        UnitGate {
            bound: bound.max(1),
            st: Mutex::new((0, false)),
            cv: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, (usize, bool)> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Wait until unit `i` may start; an error once the gate is closed.
    pub(crate) fn admit(&self, i: usize) -> Result<()> {
        let mut g = self.lock();
        while !g.1 && i >= g.0 + self.bound {
            g = self.cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        if g.1 {
            bail!("the run stopped");
        }
        Ok(())
    }

    /// Units before `next` are written.
    pub(crate) fn advance(&self, next: usize) {
        self.lock().0 = next;
        self.cv.notify_all();
    }

    /// Fail every waiting and later `admit`.
    pub(crate) fn close(&self) {
        self.lock().1 = true;
        self.cv.notify_all();
    }

    /// Whether the gate is closed (the run stopped).
    pub(crate) fn is_closed(&self) -> bool {
        self.lock().1
    }
}

/// The calling run's options beyond the manifest.
#[derive(Debug)]
pub struct RunOpts {
    /// `MOSAIC_TEST_FAIL_JOINT_CALL_UNIT` (the binary's test seam).
    pub fail_unit: Option<FailUnit>,
    /// A unit's error ends the process at once (`Error: unit ...` on
    /// stderr, exit 1, the other threads' units abandoned): the binary's
    /// behavior. Otherwise (tests) the run stops starting units, waits for
    /// the running ones and returns the error.
    pub exit_on_unit_error: bool,
    /// Tests only (the binary never sets it): sleep this long before
    /// running unit `.0`, to hold the units after it.
    pub delay_unit: Option<(usize, std::time::Duration)>,
    /// `--unit-buffer-bytes` (K2): a unit that is not the head spills its
    /// buffer once it exceeds this many bytes.
    pub unit_buffer_bytes: u64,
    /// Tests only (the binary never sets it): on a failed run, the names of
    /// the spill directory's files just before it is removed.
    pub spill_probe: Option<Arc<Mutex<Option<Vec<String>>>>>,
    /// `--max-alt-alleles` (K3): a record with more ALTs keeps this many
    /// (`call::restrict_site`); 0 = no cap. Its `Default` is 0, the uncapped
    /// output the frozen expectations hold; the binary always sets it from
    /// its option, whose default is 6.
    pub max_alt_alleles: usize,
    /// `--trim-alt-alleles` (K5): an ALT no written genotype names leaves
    /// the record (`call::restrict_site`, before the cap).
    pub trim_alt_alleles: bool,
    /// `--min-ac` (K5; 0 = off): a record whose written ALT observations
    /// sum below it is not written.
    pub min_ac: u64,
}

impl Default for RunOpts {
    fn default() -> RunOpts {
        RunOpts {
            fail_unit: None,
            exit_on_unit_error: false,
            delay_unit: None,
            unit_buffer_bytes: DEFAULT_UNIT_BUFFER_BYTES,
            spill_probe: None,
            max_alt_alleles: 0,
            trim_alt_alleles: false,
            min_ac: 0,
        }
    }
}

/// A finished unit as the main thread receives it (its output went to the
/// ordered writer).
struct Finished {
    r: UnitResult,
    secs: f64,
}

/// The spill directory: the manifest's `spill_dir`, else one named for the
/// process under the system temp dir.
fn spill_dir(m: &Manifest) -> std::path::PathBuf {
    match &m.spill_dir {
        Some(d) => std::path::PathBuf::from(d),
        None => {
            std::env::temp_dir().join(format!("mosaic-joint-call-spill-{}", std::process::id()))
        }
    }
}

/// The calling run with the binary's defaults (tests); prints the summary
/// line.
#[cfg(test)]
pub fn run(m: &Manifest, threads: usize, window: i64, rt: &tokio::runtime::Runtime) -> Result<()> {
    run_with(m, threads, window, rt, &RunOpts::default()).map(|_| ())
}

/// The calling run: the units on `threads` threads, the output written
/// in unit order by the ordered writer; prints the
/// summary line and returns it.
pub fn run_with(
    m: &Manifest,
    threads: usize,
    window: i64,
    rt: &tokio::runtime::Runtime,
    opts: &RunOpts,
) -> Result<serde_json::Value> {
    m.validate_for_call()?;
    let t0 = Instant::now();
    let mut ctx = RunCtx::new(m);
    ctx.cap = opts.max_alt_alleles;
    ctx.trim = opts.trim_alt_alleles;
    ctx.min_ac = opts.min_ac;
    let stores = Arc::new(Stores::new(m.s3.clone()));
    // Every data file's footer of the called contigs, once.
    let mut opened: Vec<[Vec<Arc<OpenFile>>; 2]> = Vec::new();
    for c in &m.contigs {
        let mut per: [Vec<Arc<OpenFile>>; 2] = Default::default();
        if m.region.as_ref().is_none_or(|r| r.contig == c.name) {
            for (t, &table) in Table::ALL.iter().enumerate() {
                let list = match table {
                    Table::SiteInfo => &c.sample_site_info,
                    Table::RefBlocks => &c.sample_ref_blocks,
                };
                let stores = stores.clone();
                per[t] = rt.block_on(async {
                    futures::stream::iter(list.iter())
                        .map(|df| {
                            let stores = stores.clone();
                            async move { OpenFile::open(&stores, table, df).await.map(Arc::new) }
                        })
                        .buffered(16)
                        .try_collect::<Vec<_>>()
                        .await
                })?;
            }
        }
        opened.push(per);
    }
    let footer_secs = t0.elapsed().as_secs_f64();

    let output_path = m.output.as_deref().expect("validated");
    let tw = Instant::now();
    let mut output = Output::create(output_path)?;
    output.write_text(header_text(m.header_lines.as_deref().expect("validated")).as_bytes())?;
    let header_secs = tw.elapsed().as_secs_f64();

    let units = plan_units(m, window);
    let threads = threads.clamp(1, units.len().max(1));
    let counters = [Arc::new(Counters::default()), Arc::new(Counters::default())];
    let writer = OrderedWriter::new(
        output,
        units.len(),
        HELD_PER_THREAD * threads,
        opts.unit_buffer_bytes,
        spill_dir(m),
    );
    let handle = rt.handle().clone();
    let t1 = Instant::now();
    // On its thread: the unit's records, each text chunk compressed as it
    // reaches TEXT_CHUNK and handed to the writer; the result is the
    // unit's counts.
    let work = |fisher: &mut Fisher, i: usize| -> Result<Finished> {
        writer.admit(i)?;
        let u = &units[i];
        let s = Instant::now();
        let name = &m.contigs[u.contig].name;
        if let Some(f) = &opts.fail_unit
            && f.contig == *name
            && f.ws == u.ws
        {
            bail!(
                "{FAIL_HOOK_ENV}: injected failure of unit {}:{}",
                name,
                u.ws
            );
        }
        if let Some((d, wait)) = opts.delay_unit
            && d == i
        {
            std::thread::sleep(wait);
        }
        let files = [&opened[u.contig][0][..], &opened[u.contig][1][..]];
        let mut sink = |text: &str| writer.push(i, text.as_bytes());
        let r = handle.block_on(run_unit(
            &ctx,
            u,
            files,
            counters.clone(),
            fisher,
            &mut sink,
        ))?;
        let secs = s.elapsed().as_secs_f64();
        writer.end_unit(i)?;
        Ok(Finished { r, secs })
    };

    // The unit lines in unit order, as the results arrive.
    let mut done: BTreeMap<usize, Finished> = BTreeMap::new();
    let mut printed = 0usize;
    let mut phases = Phases::default();
    let mut cross_check = CrossCheck::default();
    let mut cap = CapCounts::default();
    let mut trim = TrimCounts::default();
    let mut min_ac_left_out = 0u64;
    let (mut records, mut skipped, mut site_rows, mut ref_rows) = (0u64, 0u64, 0u64, 0u64);
    let mut error: Option<anyhow::Error> = None;
    let stderr = std::io::stderr();
    let flow = mosaic_core::queue::run_queue(units.len(), threads, work, |i, r| {
        let f = match r {
            Ok(f) => f,
            Err(e) => {
                let u = &units[i];
                let e = e.context(format!(
                    "unit {}:{}-{}",
                    m.contigs[u.contig].name, u.ws, u.we
                ));
                if opts.exit_on_unit_error {
                    // Exit now: the other threads' units are abandoned
                    // (the driver deletes the partial output); the spill
                    // directory goes first, best effort.
                    eprintln!("Error: {e:#}");
                    writer.close();
                    writer.remove_spill_dir();
                    std::process::exit(1);
                }
                error = Some(e);
                writer.close();
                return ControlFlow::Break(());
            }
        };
        done.insert(i, f);
        while let Some(f) = done.remove(&printed) {
            let u = &units[printed];
            let r = &f.r;
            let mut err = stderr.lock();
            for w in &r.warnings {
                let _ = writeln!(err, "{w}");
            }
            let _ = writeln!(
                err,
                "Unit {}/{}: {}:{}-{}: {} records, {} skipped; {} sample_site_info rows, {} \
                 sample_ref_blocks rows, {} deletions ({:.2} s)",
                printed + 1,
                units.len(),
                m.contigs[u.contig].name,
                commas(u.ws),
                commas(u.we),
                r.records,
                r.skipped,
                r.site_rows,
                r.ref_rows,
                r.deletions,
                f.secs
            );
            drop(err);
            records += r.records;
            skipped += r.skipped;
            site_rows += r.site_rows;
            ref_rows += r.ref_rows;
            phases.add(&r.phases);
            cross_check.add(&r.cross_check);
            cap.add(&r.cap);
            trim.add(&r.trim);
            min_ac_left_out += r.min_ac_left_out;
            printed += 1;
        }
        ControlFlow::Continue(())
    });
    let failed = |writer: &OrderedWriter, e: anyhow::Error| -> Result<serde_json::Value> {
        if let Some(p) = &opts.spill_probe {
            let names = std::fs::read_dir(writer.spill_dir()).ok().map(|d| {
                let mut v: Vec<String> = d
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect();
                v.sort();
                v
            });
            *p.lock().unwrap_or_else(|x| x.into_inner()) = names;
        }
        writer.remove_spill_dir();
        Err(e)
    };
    if let Some(e) = error {
        return failed(&writer, e);
    }
    if flow.is_break() || printed != units.len() || writer.next() != units.len() {
        let e = anyhow::anyhow!("{} of {} units written", writer.next(), units.len());
        return failed(&writer, e);
    }
    let unit_secs = t1.elapsed().as_secs_f64();
    let output = writer.take_output().expect("taken once");
    let text_bytes = output.text_bytes;
    let tf = Instant::now();
    if let Err(e) = output.finish() {
        return failed(&writer, e);
    }
    let (write_secs, copy_secs) = writer.seconds();
    let writer_secs = header_secs + write_secs + tf.elapsed().as_secs_f64();
    let held = writer.stats();
    writer.remove_spill_dir();

    let mut tables = serde_json::Map::new();
    for (t, table) in Table::ALL.iter().enumerate() {
        let files: Vec<&Arc<OpenFile>> = opened.iter().flat_map(|p| p[t].iter()).collect();
        let (req, fetched, requests) = counters[t].get();
        let s = TableSummary {
            files: files.len(),
            file_bytes: files.iter().map(|f| f.size).sum(),
            footer_bytes: files.iter().map(|f| f.footer.1).sum(),
            rows_merged: if t == 0 { site_rows } else { ref_rows },
            bytes_requested: req,
            bytes_fetched: fetched,
            requests,
        };
        tables.insert(table.name().to_string(), serde_json::to_value(s)?);
    }
    let mut summary = serde_json::json!({
        "kind": "summary",
        "mode": "call",
        "records": records,
        "skipped": skipped,
        // Under a subset (K4) the cross-check is not run: null.
        "cross_check": if m.subset { serde_json::Value::Null } else { serde_json::to_value(&cross_check)? },
        "subset": m.subset,
        "samples": m.samples.len(),
        "units": units.len(),
        "threads": threads,
        "window": window,
        "removed_ids": m.removed_ids.len(),
        "output_text_bytes": text_bytes,
        "held": {
            "bound_units": HELD_PER_THREAD * threads - 1,
            "peak_units": held.peak_units,
            "peak_bytes": held.peak_bytes,
            "bound_bytes": opts.unit_buffer_bytes,
            "spilled_units": held.spilled_units,
            "spilled_bytes": held.spilled_bytes,
        },
        "tables": tables,
        "seconds": {
            "footers": footer_secs,
            "units": unit_secs,
            "writer": writer_secs,
            "copy": copy_secs,
            "phases": phases,
            "total": t0.elapsed().as_secs_f64(),
        },
    });
    // The cap's counts, only under a cap (K3): at 0 the summary is the
    // uncapped run's.
    if opts.max_alt_alleles > 0 {
        let mut c = serde_json::to_value(cap)?;
        c["max_alt_alleles"] = serde_json::json!(opts.max_alt_alleles);
        summary["cap"] = c;
    }
    // K5: the trim's and --min-ac's counts, each only when its option is
    // set.
    if opts.trim_alt_alleles {
        summary["trim"] = serde_json::to_value(trim)?;
    }
    if opts.min_ac > 0 {
        summary["min_ac"] = serde_json::json!({
            "min_ac": opts.min_ac,
            "records_left_out": min_ac_left_out,
        });
    }
    println!("{summary}");
    Ok(summary)
}

#[cfg(test)]
mod tests {
    //! The calling run over a crafted local lake: the output does not
    //! depend on the window (a unit starting inside a deletion loads it
    //! and its site's rows through the lookback; a ref block starting
    //! before a unit serves its sites), and the calls it makes.
    use std::path::Path;

    use super::*;
    use crate::testlake::*;

    fn r(
        alleles: &[&str],
        gt: (i32, i32),
        dp: i32,
        ad: &[i32],
        pl: &[i32],
        sample: i32,
    ) -> SiteRow {
        let o = |v: &[i32]| Some(v.iter().map(|&x| Some(x)).collect());
        SiteRow {
            sample_id: sample,
            alleles: Some(alleles.iter().map(|x| x.to_string()).collect()),
            gt_1: Some(gt.0),
            gt_2: Some(gt.1),
            dp: Some(dp),
            ad: o(ad),
            f1r2: None,
            f2r1: None,
            pl: o(pl),
        }
    }

    fn blk(sample: i32, start: i32, end: i32, sc: i32) -> Block {
        Block {
            sample_id: sample,
            start,
            end,
            ad_1: Some(5),
            ad_2: Some(0),
            dp: Some(5),
            pl_1: Some(0),
            pl_2: Some(15),
            pl_3: Some(30),
            sample_counts: Some(sc),
        }
    }

    /// Two samples on a 1,000 bp contig: S1's deletion -ACGT at 100
    /// (spanning 101-103) makes it a `*` carrier at the SNP site 102; S2's
    /// long deletion at 340 (spanning 341-351) a `*` carrier at 350; S2's
    /// ref block [103, 300] serves 200; its ./. block [301, 339] fills ./.
    /// at 320.
    fn lake(dir: &Path, rpg: usize) -> Manifest {
        let mut rows = [
            (
                100,
                r(
                    &["A", "-ACGT", "*"],
                    (0, 1),
                    9,
                    &[5, 4, 0],
                    &[40, 0, 50, 60, 70, 80],
                    1,
                ),
            ),
            (
                102,
                r(
                    &["C", "xT", "*"],
                    (0, 1),
                    8,
                    &[4, 4, 0],
                    &[30, 0, 30, 60, 60, 90],
                    2,
                ),
            ),
            (
                200,
                r(
                    &["G", "xA", "*"],
                    (1, 1),
                    6,
                    &[0, 6, 0],
                    &[90, 20, 0, 99, 99, 99],
                    1,
                ),
            ),
            (
                320,
                r(
                    &["G", "xC", "*"],
                    (0, 1),
                    9,
                    &[5, 4, 0],
                    &[40, 0, 50, 60, 70, 80],
                    1,
                ),
            ),
            (
                340,
                r(
                    &["T", "-TTTTTTTTTTTT", "*"],
                    (1, 1),
                    12,
                    &[0, 12, 0],
                    &[200, 30, 0, 300, 300, 300],
                    2,
                ),
            ),
            (
                345,
                r(
                    &["C", "xA", "*"],
                    (0, 1),
                    9,
                    &[5, 4, 0],
                    &[40, 0, 50, 60, 70, 80],
                    1,
                ),
            ),
            (
                350,
                r(
                    &["T", "xG", "*"],
                    (0, 1),
                    9,
                    &[5, 4, 0],
                    &[40, 0, 50, 60, 70, 80],
                    1,
                ),
            ),
        ];
        rows.sort_by_key(|(p, row)| (*p, row.sample_id));
        let mut blocks = [
            blk(1, 1, 99, 1),
            blk(1, 104, 199, 1),
            blk(1, 201, 319, 1),
            blk(1, 321, 344, 1),
            blk(1, 346, 349, 1),
            blk(1, 351, 1000, 1),
            blk(2, 50, 101, 1),
            blk(2, 103, 300, 1),
            blk(2, 301, 339, 0),
            blk(2, 352, 400, 0),
            blk(2, 401, 1000, 1),
        ];
        blocks.sort_by_key(|b| (b.start, b.sample_id));
        // One file per sample (the whole-sample layout).
        let s1: Vec<_> = rows
            .iter()
            .filter(|x| x.1.sample_id == 1)
            .cloned()
            .collect();
        let s2: Vec<_> = rows
            .iter()
            .filter(|x| x.1.sample_id == 2)
            .cloned()
            .collect();
        let b1: Vec<_> = blocks
            .iter()
            .filter(|b| b.sample_id == 1)
            .copied()
            .collect();
        let b2: Vec<_> = blocks
            .iter()
            .filter(|b| b.sample_id == 2)
            .copied()
            .collect();
        let ssi = vec![
            site_file(dir, "s1.parquet", 0, &s1, rpg),
            site_file(dir, "s2.parquet", 0, &s2, rpg),
        ];
        let srb = vec![
            block_file(dir, "b1.parquet", 0, &b1, rpg),
            block_file(dir, "b2.parquet", 0, &b2, rpg),
        ];
        let site = |p: i32| {
            (
                p,
                Some(50.0),
                Some(10),
                Some(36000.0),
                Some(10),
                Some([3, 4, 5, 6]),
                Some([1, 1, 0]),
                5,
                1,
            )
        };
        let sites = sites_export(
            &dir.join("sites.parquet"),
            &[
                site(100),
                site(102),
                site(200),
                site(320),
                site(340),
                site(345),
                site(350),
            ],
            rpg,
        );
        let alleles = alleles_export(
            &dir.join("alleles.parquet"),
            &[
                (100, "*", 0, None),
                (100, "-ACGT", 1, Some((100, 104))),
                (100, "A", 1, None),
                (102, "*", 0, None),
                (102, "C", 3, None),
                (102, "xT", 1, None),
                (200, "*", 0, None),
                (200, "G", 0, None),
                (200, "xA", 2, None),
                (320, "*", 0, None),
                (320, "G", 0, None),
                (320, "xC", 1, None),
                (340, "*", 0, None),
                (340, "-TTTTTTTTTTTT", 2, Some((340, 352))),
                (340, "T", 0, None),
                (345, "*", 0, None),
                (345, "C", 1, None),
                (345, "xA", 1, None),
                (350, "*", 0, None),
                (350, "T", 1, None),
                (350, "xG", 1, None),
            ],
            rpg,
        );
        manifest(
            &dir.join("out.vcf"),
            &[(1, "S1"), (2, "S2")],
            ("chrT", 1000),
            sites,
            alleles,
            ssi,
            srb,
        )
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    fn call(m: &Manifest, window: i64) -> String {
        run(m, 1, window, &rt()).unwrap();
        std::fs::read_to_string(m.output.as_ref().unwrap()).unwrap()
    }

    fn sample_gts(text: &str, pos: i32) -> Vec<String> {
        let line = text
            .lines()
            .find(|l| l.split('\t').nth(1) == Some(&pos.to_string()))
            .unwrap_or_else(|| panic!("no record at {pos}"));
        line.split('\t')
            .skip(9)
            .map(|c| c.split(':').next().unwrap().to_string())
            .collect()
    }

    #[test]
    fn window_independent_with_lookback() {
        for rpg in [1, 2, 100] {
            let dir = tmpdir(&format!("lake{rpg}"));
            let m = lake(&dir, rpg);
            let whole = call(&m, 0);
            // The calls.
            assert_eq!(sample_gts(&whole, 100), ["0/1", "0/0"]);
            // S1 is a `*` carrier at 102 (its deletion at 100), VCF [C, T, *].
            assert!(whole.contains("chrT\t102\t.\tC\tT,*\t"));
            assert_eq!(sample_gts(&whole, 102), ["0/2", "0/1"]);
            // S2's block [103, 300] serves 200.
            assert_eq!(sample_gts(&whole, 200), ["1/1", "0/0"]);
            // S2's ./. block [301, 339] fills ./. at 320, with its AD / DP /
            // PL; its deletion at 340 spans 345 and 350 (no block of its
            // covers 341-351, the deletion split): a star call (count 2).
            assert!(whole.contains("\tGT:AD:DP:PL\t0/1:5,4:9:40,0,50\t./.:5,0:5:0,15,30\n"));
            assert_eq!(sample_gts(&whole, 345), ["0/1", "2/2"]);
            assert_eq!(sample_gts(&whole, 350), ["0/1", "2/2"]);
            // AC / AN from the genotypes.
            assert!(whole.contains("\tAC=1,2;AF=0.25,0.5;AN=4;"));
            // Windows of 1 bp, 7, 101 (unit 2 starts at 102, inside the
            // deletion at 100), 150 (unit 2 starts at 151, after S2's block
            // start 103), 345 (unit 2 starts at 346, inside 340's span).
            for w in [1, 7, 101, 150, 345, 1000] {
                assert_eq!(call(&m, w), whole, "window {w} rpg {rpg}");
            }
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// The run with `--unit-buffer-bytes bound`, its spill directory under
    /// the lake's; returns the summary.
    fn run_bound(m: &Manifest, threads: usize, window: i64, bound: u64) -> serde_json::Value {
        let opts = RunOpts {
            unit_buffer_bytes: bound,
            ..RunOpts::default()
        };
        let s = run_with(m, threads, window, &rt(), &opts).unwrap();
        assert_spill_gone(m);
        s
    }

    /// No spill directory is left (and, at one thread, nothing was held,
    /// buffered or spilled).
    fn assert_spill_gone(m: &Manifest) {
        let d = m.spill_dir.as_ref().unwrap();
        assert!(!Path::new(d).exists(), "{d} is left");
    }

    fn with_spill_dir(mut m: Manifest, dir: &Path) -> Manifest {
        m.spill_dir = Some(dir.join("spill").display().to_string());
        m
    }

    fn held(s: &serde_json::Value, k: &str) -> u64 {
        s["held"][k].as_u64().unwrap()
    }

    /// The BGZF EOF block.
    pub(super) const BGZF_EOF: [u8; 28] = [
        0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0, 0x42, 0x43, 0x02, 0, 0x1b, 0, 0x03,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    pub(super) fn eof_blocks(bytes: &[u8]) -> usize {
        bytes.windows(28).filter(|w| *w == BGZF_EOF).count()
    }

    fn gunzip_bytes(bytes: &[u8]) -> String {
        use std::io::Read;
        let mut s = String::new();
        noodles_bgzf::io::Reader::new(bytes)
            .read_to_string(&mut s)
            .unwrap();
        s
    }

    #[test]
    fn threads_and_windows_give_the_same_bytes() {
        for rpg in [1, 100] {
            let dir = tmpdir(&format!("lake-threads{rpg}"));
            let mut m = with_spill_dir(lake(&dir, rpg), &dir);
            let whole = call(&m, 0);
            // The default bound and bound 1 (every chunk of a unit that is
            // not the head spills).
            for bound in [DEFAULT_UNIT_BUFFER_BYTES, 1] {
                for threads in [1, 2, 4] {
                    for w in [0, 1, 7, 101, 345, 1000] {
                        let s = run_bound(&m, threads, w, bound);
                        let tag = format!("bound {bound} threads {threads} window {w}");
                        let text = std::fs::read_to_string(m.output.as_ref().unwrap()).unwrap();
                        assert_eq!(text, whole, "{tag}");
                        if threads == 1 {
                            for k in ["peak_units", "peak_bytes", "spilled_units", "spilled_bytes"]
                            {
                                assert_eq!(held(&s, k), 0, "{tag}: {k}");
                            }
                        }
                        if bound > 1 {
                            assert_eq!(held(&s, "spilled_units"), 0, "{tag}");
                        }
                        assert_eq!(held(&s, "bound_bytes"), bound);
                    }
                }
            }
            // BGZF: the same text, one EOF block at the end; the blocks
            // depend on the window (the units' chunks), not the threads or
            // the bound.
            m.output = Some(dir.join("out.vcf.gz").display().to_string());
            for w in [0, 7, 101] {
                let mut first: Option<Vec<u8>> = None;
                for bound in [DEFAULT_UNIT_BUFFER_BYTES, 1] {
                    for threads in [1, 2, 4] {
                        run_bound(&m, threads, w, bound);
                        let bytes = std::fs::read(m.output.as_ref().unwrap()).unwrap();
                        let tag = format!("bound {bound} threads {threads} window {w}");
                        assert_eq!(gunzip_bytes(&bytes), whole, "{tag}");
                        assert_eq!(eof_blocks(&bytes), 1);
                        assert!(bytes.ends_with(&BGZF_EOF));
                        match &first {
                            None => first = Some(bytes),
                            Some(f) => assert!(*f == bytes, "{tag}"),
                        }
                    }
                }
            }
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn the_gate_bounds_the_held_units() {
        let dir = tmpdir("lake-gate");
        let m = with_spill_dir(lake(&dir, 2), &dir);
        let whole = call(&m, 0);
        // 1,000 one-bp units; unit 99 (position 100) held back 300 ms: the
        // other threads finish units up to the gate's bound (102's record
        // among them) and wait.
        for threads in [1, 2, 4] {
            let opts = RunOpts {
                delay_unit: Some((99, std::time::Duration::from_millis(300))),
                ..RunOpts::default()
            };
            let s = run_with(&m, threads, 1, &rt(), &opts).unwrap();
            let held = &s["held"];
            let bound = HELD_PER_THREAD * threads - 1;
            assert_eq!(held["bound_units"].as_u64().unwrap() as usize, bound);
            let peak = held["peak_units"].as_u64().unwrap() as usize;
            // At one thread nothing is ever held; above it the units after
            // the slow one fill the gate exactly.
            assert_eq!(
                peak,
                if threads == 1 { 0 } else { bound },
                "threads {threads}"
            );
            if threads > 1 {
                assert!(held["peak_bytes"].as_u64().unwrap() > 0);
            } else {
                assert_eq!(held["peak_bytes"], 0);
            }
            // The default bound: the held units stay in memory.
            assert_eq!(held["spilled_units"], 0, "threads {threads}");
            assert_spill_gone(&m);
            assert_eq!(s["units"], 1000);
            let text = std::fs::read_to_string(m.output.as_ref().unwrap()).unwrap();
            assert_eq!(text, whole, "threads {threads}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The head delayed under bound 1 (K2): the units after it finish
    /// first, spill, and are copied in order once it ends.
    #[test]
    fn a_delayed_head_spills_the_held_units_and_copies_them_in_order() {
        let dir = tmpdir("lake-spill");
        let mut m = with_spill_dir(lake(&dir, 2), &dir);
        let whole = call(&m, 0);
        for gz in [false, true] {
            if gz {
                m.output = Some(dir.join("out.vcf.gz").display().to_string());
            }
            for threads in [1, 2, 4] {
                // Window 50: units chrT:1-50 (unit 0, no record, delayed),
                // 51-100 (100), 101-150 (102), 151-200 (200), 201-250,
                // 251-300, 301-350 (320, 340, 345, 350), ...
                let opts = RunOpts {
                    delay_unit: Some((0, std::time::Duration::from_millis(300))),
                    unit_buffer_bytes: 1,
                    ..RunOpts::default()
                };
                let s = run_with(&m, threads, 50, &rt(), &opts).unwrap();
                assert_spill_gone(&m);
                let bound = (HELD_PER_THREAD * threads - 1) as u64;
                let tag = format!("gz {gz} threads {threads}");
                if threads == 1 {
                    for k in ["peak_units", "peak_bytes", "spilled_units", "spilled_bytes"] {
                        assert_eq!(held(&s, k), 0, "{tag}: {k}");
                    }
                } else {
                    // Units 1 to 2T − 1 finish while unit 0 sleeps: held,
                    // and the ones with records spilled.
                    assert_eq!(held(&s, "peak_units"), bound, "{tag}");
                    assert!(held(&s, "spilled_units") > 0, "{tag}");
                    assert!(held(&s, "spilled_bytes") > 0, "{tag}");
                    assert!(s["seconds"]["copy"].as_f64().unwrap() >= 0.0);
                }
                let bytes = std::fs::read(m.output.as_ref().unwrap()).unwrap();
                if gz {
                    assert_eq!(gunzip_bytes(&bytes), whole, "{tag}");
                    // The window-50 bytes are the same at every thread count.
                    let one = run_bound(&m, 1, 50, DEFAULT_UNIT_BUFFER_BYTES);
                    assert_eq!(held(&one, "spilled_units"), 0);
                    assert!(
                        std::fs::read(m.output.as_ref().unwrap()).unwrap() == bytes,
                        "{tag}"
                    );
                } else {
                    assert_eq!(String::from_utf8(bytes).unwrap(), whole, "{tag}");
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The bound on the buffers (K2): the peak buffered bytes never exceed
    /// `(2 × threads − 1) × (bound + the largest compressed chunk)`; at one
    /// thread nothing is buffered or spilled at any bound.
    #[test]
    fn the_buffers_stay_within_their_bound() {
        let dir = tmpdir("lake-bound");
        let mut m = with_spill_dir(lake(&dir, 2), &dir);
        let whole = call(&m, 0);
        // Each unit's text is one chunk (far below TEXT_CHUNK): its largest
        // compressed chunk is at most the whole text's size plus the
        // blocks' framing.
        let largest = whole.len() as u64 + 1024;
        for gz in [false, true] {
            if gz {
                m.output = Some(dir.join("out.vcf.gz").display().to_string());
            }
            for bound in [0, 1, 100, DEFAULT_UNIT_BUFFER_BYTES] {
                for threads in [1, 2, 4] {
                    for (w, delay) in [(1, Some(99)), (50, Some(0)), (7, None), (101, None)] {
                        let opts = RunOpts {
                            delay_unit: delay.map(|d| (d, std::time::Duration::from_millis(50))),
                            unit_buffer_bytes: bound,
                            ..RunOpts::default()
                        };
                        let s = run_with(&m, threads, w, &rt(), &opts).unwrap();
                        assert_spill_gone(&m);
                        let tag = format!("gz {gz} bound {bound} threads {threads} window {w}");
                        let n = (HELD_PER_THREAD * threads - 1) as u64;
                        let peak = held(&s, "peak_bytes");
                        assert!(peak <= n * bound.saturating_add(largest), "{tag}: {peak}");
                        // A buffer never holds more than its unit's bytes.
                        assert!(peak <= n * largest, "{tag}: {peak}");
                        if threads == 1 {
                            assert_eq!(peak, 0, "{tag}");
                            assert_eq!(held(&s, "spilled_units"), 0, "{tag}");
                        }
                        // A unit spills only once its buffer exceeds the
                        // bound: nothing spills under a bound above the
                        // whole output.
                        if bound >= largest {
                            assert_eq!(held(&s, "spilled_units"), 0, "{tag}");
                        }
                        let bytes = std::fs::read(m.output.as_ref().unwrap()).unwrap();
                        let text = if gz {
                            gunzip_bytes(&bytes)
                        } else {
                            String::from_utf8(bytes).unwrap()
                        };
                        assert_eq!(text, whole, "{tag}");
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A unit error with spill files present (K2): the head delayed under
    /// bound 1, a later admitted unit fails; the run returns its error and
    /// leaves no spill directory.
    #[test]
    fn a_unit_error_with_spill_files_leaves_no_spill_directory() {
        let dir = tmpdir("lake-fail-spill");
        let m = with_spill_dir(lake(&dir, 2), &dir);
        // Window 50 at 4 threads: units 1 to 7 are admitted while unit 0
        // (chrT:1-50) sleeps; unit 7 (chrT:351-400) fails, after several of
        // units 1 to 6 (records in 1, 2, 3 and 6) finished and spilled.
        let probe = Arc::new(Mutex::new(None));
        let opts = RunOpts {
            fail_unit: Some(FailUnit {
                contig: "chrT".into(),
                ws: 351,
            }),
            delay_unit: Some((0, std::time::Duration::from_millis(300))),
            unit_buffer_bytes: 1,
            spill_probe: Some(probe.clone()),
            ..RunOpts::default()
        };
        let e = run_with(&m, 4, 50, &rt(), &opts).unwrap_err();
        let msg = format!("{e:#}");
        assert!(
            msg.starts_with("unit chrT:351-400: MOSAIC_TEST_FAIL_JOINT_CALL_UNIT: injected"),
            "{msg}"
        );
        // Spill files were there just before the cleanup, and are gone.
        let files = probe
            .lock()
            .unwrap()
            .clone()
            .expect("the spill directory existed");
        assert!(!files.is_empty(), "no spill file");
        assert!(
            files
                .iter()
                .all(|f| f.starts_with("unit-") && f.ends_with(".spill")),
            "{files:?}"
        );
        assert_spill_gone(&m);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_unit_error_stops_the_run() {
        let dir = tmpdir("lake-fail");
        let m = with_spill_dir(lake(&dir, 2), &dir);
        let opts = |c: &str, ws| RunOpts {
            fail_unit: Some(FailUnit {
                contig: c.to_string(),
                ws,
            }),
            ..RunOpts::default()
        };
        // Window 100: units chrT:1-100, 101-200, ...; the fourth fails.
        let e = run_with(&m, 4, 100, &rt(), &opts("chrT", 301)).unwrap_err();
        let msg = format!("{e:#}");
        assert!(
            msg.starts_with("unit chrT:301-400: MOSAIC_TEST_FAIL_JOINT_CALL_UNIT: injected"),
            "{msg}"
        );
        // A start no unit has, or another contig: inert.
        let whole = call(&m, 0);
        for (c, ws) in [("chrT", 302), ("chrU", 301)] {
            run_with(&m, 4, 100, &rt(), &opts(c, ws)).unwrap();
            let text = std::fs::read_to_string(m.output.as_ref().unwrap()).unwrap();
            assert_eq!(text, whole);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn region_equals_the_slice() {
        let dir = tmpdir("lake-region");
        let mut m = lake(&dir, 2);
        let whole = call(&m, 0);
        m.region = Some(crate::manifest::Region {
            contig: "chrT".into(),
            start: 102,
            end: 345,
        });
        let region = call(&m, 0);
        let body = |t: &str| -> Vec<String> {
            t.lines()
                .filter(|l| !l.starts_with('#'))
                .map(String::from)
                .collect()
        };
        let want: Vec<String> = body(&whole)
            .into_iter()
            .filter(|l| {
                let p: i32 = l.split('\t').nth(1).unwrap().parse().unwrap();
                (102..=345).contains(&p)
            })
            .collect();
        assert_eq!(body(&region), want);
        assert!(region.starts_with("##fileformat=VCFv4.2\n##contig=<ID=chrT,length=1000>\n#CHROM"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod slice_tests {
    //! The checked-in slice fixture (`tests/data/jc_slice`, written by
    //! `rust/tools/gen_jc_slice.py` from the six chr21 samples as three
    //! 2-sample batches): the run over its local files writes the frozen
    //! expectation `expected.vcf.gz` byte for byte (decompressed). That file
    //! was recorded from the Python engine's `--region` output and is kept
    //! as recorded since that engine's removal; regenerating it is a
    //! re-recording, only for a change meant to change the output.
    use std::io::Read;
    use std::path::PathBuf;

    use noodles_bgzf as bgzf;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;
    use crate::manifest::{Contig, DataFile, Region, Sample};
    use crate::testlake::tmpdir;

    fn data() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/jc_slice")
    }

    fn data_file(p: PathBuf) -> DataFile {
        let f = std::fs::File::open(&p).unwrap();
        let rows = SerializedFileReader::new(f)
            .unwrap()
            .metadata()
            .file_metadata()
            .num_rows();
        DataFile {
            uri: format!("file://{}", p.display()),
            record_count: rows as u64,
            file_size_in_bytes: std::fs::metadata(&p).unwrap().len(),
            delete_files: vec![],
        }
    }

    fn gunzip(p: &std::path::Path) -> String {
        let mut s = String::new();
        bgzf::io::Reader::new(std::fs::File::open(p).unwrap())
            .read_to_string(&mut s)
            .unwrap();
        s
    }

    #[test]
    fn slice_equals_the_frozen_expectation() {
        let d = data();
        let Ok(text) = std::fs::read_to_string(d.join("slice.json")) else {
            eprintln!("skipped: {} not present", d.display());
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let expected = gunzip(&d.join("expected.vcf.gz"));
        let header: Vec<String> = expected
            .lines()
            .take_while(|l| l.starts_with('#'))
            .map(String::from)
            .collect();
        let c = &v["contig"];
        let r = &v["region"];
        let out_dir = tmpdir("slice");
        let mut m = Manifest {
            output: None,
            header_lines: Some(header),
            samples: v["samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| Sample {
                    id: s["id"].as_i64().unwrap() as i32,
                    name: s["name"].as_str().unwrap().to_string(),
                })
                .collect(),
            removed_ids: v["removed_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect(),
            contigs: vec![Contig {
                index: c["index"].as_i64().unwrap() as i32,
                name: c["name"].as_str().unwrap().to_string(),
                length: c["length"].as_i64().unwrap(),
                sites: Some(d.join("sites.parquet").display().to_string()),
                alleles: Some(d.join("alleles.parquet").display().to_string()),
                sample_site_info: vec![data_file(d.join("sample_site_info.parquet"))],
                sample_ref_blocks: vec![data_file(d.join("sample_ref_blocks.parquet"))],
            }],
            s3: None,
            window: None,
            region: Some(Region {
                contig: r["contig"].as_str().unwrap().to_string(),
                start: r["start"].as_i64().unwrap(),
                end: r["end"].as_i64().unwrap(),
            }),
            tmpdir: None,
            spill_dir: None,
            stale: false,
            subset: false,
        };
        m.validate().unwrap();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let records = expected.lines().filter(|l| !l.starts_with('#')).count();
        assert!(records > 1000, "{records}");
        for w in [0, 1_000_000, 100_000, 30_000] {
            let mut first: Option<Vec<u8>> = None;
            for threads in [1, 2, 4] {
                let out = out_dir.join(format!("w{w}.t{threads}.vcf.gz"));
                m.output = Some(out.display().to_string());
                run(&m, threads, w, &rt).unwrap();
                let got = gunzip(&out);
                assert!(
                    got == expected,
                    "window {w} threads {threads}: the text differs"
                );
                let bytes = std::fs::read(&out).unwrap();
                assert_eq!(super::tests::eof_blocks(&bytes), 1);
                match &first {
                    None => first = Some(bytes),
                    Some(f) => assert!(*f == bytes, "window {w} threads {threads}"),
                }
            }
        }
        // Bound 1 (K2): every chunk of a unit that is not the head spills;
        // the bytes are the same as the default bound's at the window.
        let spill = out_dir.join("spill");
        m.spill_dir = Some(spill.display().to_string());
        for w in [100_000, 30_000] {
            let base = out_dir.join(format!("w{w}.t1.vcf.gz"));
            let base = std::fs::read(&base).unwrap();
            for threads in [2, 4] {
                let out = out_dir.join(format!("w{w}.t{threads}.b1.vcf.gz"));
                m.output = Some(out.display().to_string());
                let opts = RunOpts {
                    unit_buffer_bytes: 1,
                    ..RunOpts::default()
                };
                run_with(&m, threads, w, &rt, &opts).unwrap();
                assert!(!spill.exists(), "window {w} threads {threads}: spill left");
                assert!(
                    std::fs::read(&out).unwrap() == base,
                    "window {w} threads {threads} bound 1: the bytes differ"
                );
            }
        }
        // The plain output.
        for threads in [1, 4] {
            let out = out_dir.join(format!("t{threads}.vcf"));
            m.output = Some(out.display().to_string());
            run(&m, threads, 100_000, &rt).unwrap();
            assert!(
                std::fs::read_to_string(&out).unwrap() == expected,
                "plain, threads {threads}"
            );
        }
        std::fs::remove_dir_all(&out_dir).unwrap();
    }

    /// The slice's expectation under an ALT cap (K3): the SHA-256, record
    /// and byte counts of `tests/vcf_project.py`'s projection of
    /// `expected.vcf.gz`'s text at the cap, and its four counts (records
    /// capped, genotypes written ./., dropped ALT observations, kept-allele
    /// observations lost), printed by `rust/tools/gen_jc_slice_cap_digests.py`.
    struct CapDigest {
        cap: usize,
        sha256: &'static str,
        records: usize,
        bytes: usize,
        counts: [u64; 4],
    }

    const CAP_DIGESTS: [CapDigest; 2] = [
        CapDigest {
            cap: 1,
            sha256: "201cad4c0a8cccc1553ce53e3746997fc7e71bc69316485caa68508754513eeb",
            records: 6112,
            bytes: 1671868,
            counts: [318, 409, 473, 220],
        },
        CapDigest {
            cap: 3,
            sha256: "fa2ddad5f82ba854d10d518e9b2de76f3b436dd0944de7c93b5e538a92f294df",
            records: 6112,
            bytes: 1719801,
            counts: [56, 28, 29, 20],
        },
    ];

    /// The cap's header line, as the driver writes it (`joint_call.py`'s
    /// `MAX_ALT_ALLELES_HEADER`), before the first `##contig` line.
    fn header_with_cap(header: &[String], cap: usize) -> Vec<String> {
        let at = header
            .iter()
            .position(|l| l.starts_with("##contig=") || l.starts_with("#CHROM"))
            .unwrap();
        let mut h = header.to_vec();
        h.insert(at, format!("##mosaic_max_alt_alleles={cap}"));
        h
    }

    #[test]
    fn slice_under_caps_equals_the_projection() {
        let d = data();
        let Ok(text) = std::fs::read_to_string(d.join("slice.json")) else {
            eprintln!("skipped: {} not present", d.display());
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let expected = gunzip(&d.join("expected.vcf.gz"));
        let header: Vec<String> = expected
            .lines()
            .take_while(|l| l.starts_with('#'))
            .map(String::from)
            .collect();
        let c = &v["contig"];
        let r = &v["region"];
        let out_dir = tmpdir("slice-cap");
        let mut m = Manifest {
            output: None,
            header_lines: None,
            samples: v["samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| Sample {
                    id: s["id"].as_i64().unwrap() as i32,
                    name: s["name"].as_str().unwrap().to_string(),
                })
                .collect(),
            removed_ids: vec![],
            contigs: vec![Contig {
                index: c["index"].as_i64().unwrap() as i32,
                name: c["name"].as_str().unwrap().to_string(),
                length: c["length"].as_i64().unwrap(),
                sites: Some(d.join("sites.parquet").display().to_string()),
                alleles: Some(d.join("alleles.parquet").display().to_string()),
                sample_site_info: vec![data_file(d.join("sample_site_info.parquet"))],
                sample_ref_blocks: vec![data_file(d.join("sample_ref_blocks.parquet"))],
            }],
            s3: None,
            window: None,
            region: Some(Region {
                contig: r["contig"].as_str().unwrap().to_string(),
                start: r["start"].as_i64().unwrap(),
                end: r["end"].as_i64().unwrap(),
            }),
            tmpdir: None,
            spill_dir: None,
            stale: false,
            subset: false,
        };
        assert!(v["removed_ids"].as_array().unwrap().is_empty());
        m.validate().unwrap();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        for want in &CAP_DIGESTS {
            m.header_lines = Some(header_with_cap(&header, want.cap));
            for (threads, w) in [(1, 0), (2, 100_000), (4, 30_000)] {
                let tag = format!("cap {} threads {threads} window {w}", want.cap);
                let out = out_dir.join(format!("cap{}.t{threads}.vcf.gz", want.cap));
                m.output = Some(out.display().to_string());
                let opts = RunOpts {
                    max_alt_alleles: want.cap,
                    ..RunOpts::default()
                };
                let s = run_with(&m, threads, w, &rt, &opts).unwrap();
                let got = gunzip(&out);
                let records = got.lines().filter(|l| !l.starts_with('#')).count();
                assert_eq!((records, got.len()), (want.records, want.bytes), "{tag}");
                assert_eq!(sha256_hex(got.as_bytes()), want.sha256, "{tag}");
                let cap = &s["cap"];
                let counts = [
                    "records_capped",
                    "genotypes_no_called",
                    "dropped_alt_observations",
                    "kept_alt_observations_lost",
                ]
                .map(|k| cap[k].as_u64().unwrap());
                assert_eq!(counts, want.counts, "{tag}");
                assert_eq!(
                    s["cross_check"]["records"].as_u64().unwrap(),
                    (want.records as u64) - want.counts[0],
                    "{tag}"
                );
            }
        }
        std::fs::remove_dir_all(&out_dir).unwrap();
    }

    /// The slice's expectation under a sample subset (K4): the SHA-256,
    /// record and byte counts of `tests/vcf_project.py`'s
    /// `project(project_samples(<expected.vcf.gz's text>, names), cap)`
    /// and the cap's four counts, printed by
    /// `rust/tools/gen_jc_slice_cap_digests.py`.
    struct SubsetDigest {
        names: &'static [&'static str],
        cap: usize,
        sha256: &'static str,
        records: usize,
        bytes: usize,
        counts: [u64; 4],
    }

    const SUBSET_DIGESTS: [SubsetDigest; 4] = [
        SubsetDigest {
            names: &["HG00100", "HG00245"],
            cap: 0,
            sha256: "14b63ff6f6420810afec3e53be749bd177c0d1b946af247e2ae384f57fb89466",
            records: 6112,
            bytes: 1003294,
            counts: [0, 0, 0, 0],
        },
        SubsetDigest {
            names: &["HG00101", "HG00102"],
            cap: 0,
            sha256: "a88121fff828ed75738f4b3296cf6aa2a932b495c3b68fb351b33df36867524a",
            records: 6112,
            bytes: 1002859,
            counts: [0, 0, 0, 0],
        },
        SubsetDigest {
            names: &["HG00099"],
            cap: 0,
            sha256: "963c7aa7c245db6567a23ac4130278bbd82900a4ab0afd5093c9f12b3259e8b1",
            records: 6112,
            bytes: 821483,
            counts: [0, 0, 0, 0],
        },
        SubsetDigest {
            names: &["HG00100", "HG00245"],
            cap: 3,
            sha256: "fbd92400ea840548c23f24e37078ee8da398d8d4b6d32049116ae7ce38e2bd1b",
            records: 6112,
            bytes: 992505,
            counts: [55, 0, 0, 0],
        },
    ];

    #[test]
    fn slice_under_subsets_equals_the_projection() {
        let d = data();
        let Ok(text) = std::fs::read_to_string(d.join("slice.json")) else {
            eprintln!("skipped: {} not present", d.display());
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let expected = gunzip(&d.join("expected.vcf.gz"));
        let header: Vec<String> = expected
            .lines()
            .take_while(|l| l.starts_with('#'))
            .map(String::from)
            .collect();
        let c = &v["contig"];
        let r = &v["region"];
        let out_dir = tmpdir("slice-subset");
        let all: Vec<Sample> = v["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| Sample {
                id: s["id"].as_i64().unwrap() as i32,
                name: s["name"].as_str().unwrap().to_string(),
            })
            .collect();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        for want in &SUBSET_DIGESTS {
            // The subset in the header's order (the manifest's samples are
            // in it).
            let samples: Vec<Sample> = all
                .iter()
                .filter(|s| want.names.contains(&s.name.as_str()))
                .cloned()
                .collect();
            assert_eq!(samples.len(), want.names.len());
            let mut h = if want.cap > 0 {
                header_with_cap(&header, want.cap)
            } else {
                header.clone()
            };
            let chrom = h.last_mut().unwrap();
            let cols: Vec<&str> = chrom.split('\t').take(9).collect();
            let names: Vec<&str> = samples.iter().map(|s| s.name.as_str()).collect();
            *chrom = format!("{}\t{}", cols.join("\t"), names.join("\t"));
            let mut m = Manifest {
                output: None,
                header_lines: Some(h),
                samples,
                removed_ids: vec![],
                contigs: vec![Contig {
                    index: c["index"].as_i64().unwrap() as i32,
                    name: c["name"].as_str().unwrap().to_string(),
                    length: c["length"].as_i64().unwrap(),
                    sites: Some(d.join("sites.parquet").display().to_string()),
                    alleles: Some(d.join("alleles.parquet").display().to_string()),
                    sample_site_info: vec![data_file(d.join("sample_site_info.parquet"))],
                    sample_ref_blocks: vec![data_file(d.join("sample_ref_blocks.parquet"))],
                }],
                s3: None,
                window: None,
                region: Some(Region {
                    contig: r["contig"].as_str().unwrap().to_string(),
                    start: r["start"].as_i64().unwrap(),
                    end: r["end"].as_i64().unwrap(),
                }),
                tmpdir: None,
                spill_dir: None,
                stale: false,
                subset: true,
            };
            m.validate().unwrap();
            for (threads, w) in [(1, 0), (4, 100_000)] {
                let tag = format!(
                    "{:?} cap {} threads {threads} window {w}",
                    want.names, want.cap
                );
                let out = out_dir.join(format!("sub.t{threads}.vcf.gz"));
                m.output = Some(out.display().to_string());
                let opts = RunOpts {
                    max_alt_alleles: want.cap,
                    ..RunOpts::default()
                };
                let s = run_with(&m, threads, w, &rt, &opts).unwrap();
                let got = gunzip(&out);
                let records = got.lines().filter(|l| !l.starts_with('#')).count();
                assert_eq!((records, got.len()), (want.records, want.bytes), "{tag}");
                assert_eq!(sha256_hex(got.as_bytes()), want.sha256, "{tag}");
                assert!(s["cross_check"].is_null(), "{tag}");
                assert_eq!(s["subset"], true);
                if want.cap > 0 {
                    let counts = [
                        "records_capped",
                        "genotypes_no_called",
                        "dropped_alt_observations",
                        "kept_alt_observations_lost",
                    ]
                    .map(|k| s["cap"][k].as_u64().unwrap());
                    assert_eq!(counts, want.counts, "{tag}");
                } else {
                    assert!(s.get("cap").is_none());
                }
            }
        }
        std::fs::remove_dir_all(&out_dir).unwrap();
    }

    /// The slice's expectation under the trim and `--min-ac`: the
    /// SHA-256, record and byte counts of `tests/vcf_project.py`'s chain
    /// `project_min_ac(project(project_trim(project_samples(<expected.vcf.gz's
    /// text>, names)), cap), min_ac)` (each step only when set; `names`
    /// empty: the whole cohort), the cap's four counts, the trim's two and
    /// the records left out, printed by
    /// `rust/tools/gen_jc_slice_cap_digests.py`.
    struct OptionDigest {
        names: &'static [&'static str],
        trim: bool,
        cap: usize,
        min_ac: u64,
        sha256: &'static str,
        records: usize,
        bytes: usize,
        counts: [u64; 4],
        trim_counts: [u64; 2],
        left_out: u64,
    }

    const OPTION_DIGESTS: [OptionDigest; 8] = [
        OptionDigest {
            names: &[],
            trim: true,
            cap: 0,
            min_ac: 0,
            sha256: "eff63cb73e2c75e88216c182fd2829fc9fd2bf38b44ad59e27301582d6ffd3a9",
            records: 6112,
            bytes: 1706229,
            counts: [0, 0, 0, 0],
            trim_counts: [176, 273],
            left_out: 0,
        },
        OptionDigest {
            names: &[],
            trim: false,
            cap: 0,
            min_ac: 1,
            sha256: "8ef3c93376c42a07957d2477b4f857868d47d9ffe2e5d2459d702ea250e270d3",
            records: 6111,
            bytes: 1745489,
            counts: [0, 0, 0, 0],
            trim_counts: [0, 0],
            left_out: 1,
        },
        OptionDigest {
            names: &[],
            trim: false,
            cap: 0,
            min_ac: 2,
            sha256: "400975160b727e0705498dc8b7c1ec75061df7762d2d5224941488f7bb15937b",
            records: 4890,
            bytes: 1422149,
            counts: [0, 0, 0, 0],
            trim_counts: [0, 0],
            left_out: 1222,
        },
        OptionDigest {
            names: &[],
            trim: true,
            cap: 0,
            min_ac: 1,
            sha256: "14d951834167b270eebff66f67ac4b24e6ce7dbe2baf74e2360ec3b53b3329a7",
            records: 6111,
            bytes: 1706061,
            counts: [0, 0, 0, 0],
            trim_counts: [176, 273],
            left_out: 1,
        },
        OptionDigest {
            names: &[],
            trim: true,
            cap: 3,
            min_ac: 2,
            sha256: "5b38e0f311cd5e23aea8debd5120c32f0204dbe19868b1f945b187f552c563ae",
            records: 4890,
            bytes: 1381622,
            counts: [18, 28, 29, 20],
            trim_counts: [176, 273],
            left_out: 1222,
        },
        OptionDigest {
            names: &["HG00100", "HG00245"],
            trim: true,
            cap: 0,
            min_ac: 1,
            sha256: "fd78b9ac49999ddbc252b517d20f60f02095e2f5b43a7bca46bee4f1bf7c337d",
            records: 4241,
            bytes: 718286,
            counts: [0, 0, 0, 0],
            trim_counts: [2076, 2321],
            left_out: 1871,
        },
        OptionDigest {
            names: &["HG00099"],
            trim: true,
            cap: 6,
            min_ac: 1,
            sha256: "a2f110d45513cc0f089c3f520faade527decd731fe29a06a1f933d485ea575ab",
            records: 3126,
            bytes: 446495,
            counts: [0, 0, 0, 0],
            trim_counts: [3164, 3485],
            left_out: 2986,
        },
        OptionDigest {
            names: &["HG00101", "HG00102"],
            trim: false,
            cap: 1,
            min_ac: 1,
            sha256: "7661bd55b6ceef86a8de4fadf04fd72e485c7d457088a73e59cbdc4b82ca368f",
            records: 4560,
            bytes: 759467,
            counts: [303, 126, 144, 84],
            trim_counts: [0, 0],
            left_out: 1552,
        },
    ];

    /// The options' header lines, as the driver writes them
    /// (`joint_call.py`'s `vcf_header_lines`): trim, cap, min-ac, each only
    /// when set, before the first `##contig` line.
    fn header_with_options(header: &[String], trim: bool, cap: usize, min_ac: u64) -> Vec<String> {
        let at = header
            .iter()
            .position(|l| l.starts_with("##contig=") || l.starts_with("#CHROM"))
            .unwrap();
        let mut lines = Vec::new();
        if trim {
            lines.push("##mosaic_trim_alt_alleles=true".to_string());
        }
        if cap > 0 {
            lines.push(format!("##mosaic_max_alt_alleles={cap}"));
        }
        if min_ac > 0 {
            lines.push(format!("##mosaic_min_ac={min_ac}"));
        }
        let mut h = header.to_vec();
        h.splice(at..at, lines);
        h
    }

    #[test]
    fn slice_under_trim_and_min_ac_equals_the_projection() {
        let d = data();
        let Ok(text) = std::fs::read_to_string(d.join("slice.json")) else {
            eprintln!("skipped: {} not present", d.display());
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let expected = gunzip(&d.join("expected.vcf.gz"));
        let header: Vec<String> = expected
            .lines()
            .take_while(|l| l.starts_with('#'))
            .map(String::from)
            .collect();
        let c = &v["contig"];
        let r = &v["region"];
        let out_dir = tmpdir("slice-options");
        let all: Vec<Sample> = v["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| Sample {
                id: s["id"].as_i64().unwrap() as i32,
                name: s["name"].as_str().unwrap().to_string(),
            })
            .collect();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        for want in &OPTION_DIGESTS {
            let subset = !want.names.is_empty();
            let samples: Vec<Sample> = all
                .iter()
                .filter(|s| !subset || want.names.contains(&s.name.as_str()))
                .cloned()
                .collect();
            let mut h = header_with_options(&header, want.trim, want.cap, want.min_ac);
            if subset {
                let chrom = h.last_mut().unwrap();
                let cols: Vec<&str> = chrom.split('\t').take(9).collect();
                let names: Vec<&str> = samples.iter().map(|s| s.name.as_str()).collect();
                *chrom = format!("{}\t{}", cols.join("\t"), names.join("\t"));
            }
            let mut m = Manifest {
                output: None,
                header_lines: Some(h),
                samples,
                removed_ids: vec![],
                contigs: vec![Contig {
                    index: c["index"].as_i64().unwrap() as i32,
                    name: c["name"].as_str().unwrap().to_string(),
                    length: c["length"].as_i64().unwrap(),
                    sites: Some(d.join("sites.parquet").display().to_string()),
                    alleles: Some(d.join("alleles.parquet").display().to_string()),
                    sample_site_info: vec![data_file(d.join("sample_site_info.parquet"))],
                    sample_ref_blocks: vec![data_file(d.join("sample_ref_blocks.parquet"))],
                }],
                s3: None,
                window: None,
                region: Some(Region {
                    contig: r["contig"].as_str().unwrap().to_string(),
                    start: r["start"].as_i64().unwrap(),
                    end: r["end"].as_i64().unwrap(),
                }),
                tmpdir: None,
                spill_dir: None,
                stale: false,
                subset,
            };
            m.validate().unwrap();
            for (threads, w) in [(1, 0), (4, 100_000)] {
                let tag = format!(
                    "{:?} trim {} cap {} min-ac {} threads {threads} window {w}",
                    want.names, want.trim, want.cap, want.min_ac
                );
                let out = out_dir.join(format!("opt.t{threads}.vcf.gz"));
                m.output = Some(out.display().to_string());
                let opts = RunOpts {
                    max_alt_alleles: want.cap,
                    trim_alt_alleles: want.trim,
                    min_ac: want.min_ac,
                    ..RunOpts::default()
                };
                let s = run_with(&m, threads, w, &rt, &opts).unwrap();
                let got = gunzip(&out);
                let records = got.lines().filter(|l| !l.starts_with('#')).count();
                assert_eq!((records, got.len()), (want.records, want.bytes), "{tag}");
                assert_eq!(sha256_hex(got.as_bytes()), want.sha256, "{tag}");
                assert_eq!(s["records"].as_u64().unwrap(), want.records as u64, "{tag}");
                if want.cap > 0 {
                    let counts = [
                        "records_capped",
                        "genotypes_no_called",
                        "dropped_alt_observations",
                        "kept_alt_observations_lost",
                    ]
                    .map(|k| s["cap"][k].as_u64().unwrap());
                    assert_eq!(counts, want.counts, "{tag}");
                } else {
                    assert!(s.get("cap").is_none(), "{tag}");
                }
                if want.trim {
                    let t =
                        ["records_trimmed", "alts_dropped"].map(|k| s["trim"][k].as_u64().unwrap());
                    assert_eq!(t, want.trim_counts, "{tag}");
                } else {
                    assert!(s.get("trim").is_none(), "{tag}");
                }
                if want.min_ac > 0 {
                    assert_eq!(s["min_ac"]["min_ac"], want.min_ac, "{tag}");
                    assert_eq!(s["min_ac"]["records_left_out"], want.left_out, "{tag}");
                } else {
                    assert!(s.get("min_ac").is_none(), "{tag}");
                }
            }
        }
        std::fs::remove_dir_all(&out_dir).unwrap();
    }

    /// SHA-256 (FIPS 180-4), for the cap digests above: the crate has no
    /// hash dependency and K3 adds no crate.
    fn sha256_hex(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut msg = data.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        for block in msg.chunks(64) {
            let mut w = [0u32; 64];
            for (i, c) in block.chunks(4).enumerate() {
                w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v = [
                    t1.wrapping_add(t2),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }
            for (x, y) in h.iter_mut().zip(v) {
                *x = x.wrapping_add(y);
            }
        }
        h.iter().map(|x| format!("{x:08x}")).collect()
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let a = vec![b'a'; 1000];
        assert_eq!(
            sha256_hex(&a),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }
}

#[cfg(test)]
mod empty_tests {
    use super::*;
    use crate::testlake::*;

    #[test]
    fn a_contig_without_sites_writes_the_header_only() {
        let dir = tmpdir("empty");
        let sites = sites_export(&dir.join("sites.parquet"), &[], 10);
        let alleles = alleles_export(&dir.join("alleles.parquet"), &[], 10);
        let m = manifest(
            &dir.join("out.vcf.gz"),
            &[(1, "S1")],
            ("chrE", 500),
            sites,
            alleles,
            vec![],
            vec![],
        );
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        for w in [0, 100] {
            run(&m, 1, w, &rt).unwrap();
            let mut text = String::new();
            use std::io::Read;
            noodles_bgzf::io::Reader::new(std::fs::File::open(dir.join("out.vcf.gz")).unwrap())
                .read_to_string(&mut text)
                .unwrap();
            assert_eq!(text.lines().count(), 3);
            assert!(text.lines().all(|l| l.starts_with('#')));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod cap_tests {
    //! The ALT cap (`--max-alt-alleles`) over a crafted lake, the expected
    //! lines derived by hand from the contract (a capped record is a
    //! projection of the uncapped one):
    //! four samples; S4's deletion -ACGT at 100 makes it a `*` carrier (s/s)
    //! at 102, whose VCF alleles are [C, A, G, T, *] with AC 1, 1, 2, 2 (S1
    //! A/G, S2 T/T, S4 */*); S3 is in an all-positive ref block (PL 5, 15,
    //! 30) everywhere, never normalized (the one PL rule, which replaced
    //! the cap's first normalization of a fill at a capped record); 200 has two ALTs
    //! (S1 A/C).
    use super::*;
    use crate::testlake::*;

    fn row(alleles: &[&str], gt: (i32, i32), dp: i32, ad: &[i32], pl: &[i32], id: i32) -> SiteRow {
        let o = |v: &[i32]| Some(v.iter().map(|&x| Some(x)).collect());
        SiteRow {
            sample_id: id,
            alleles: Some(alleles.iter().map(|x| x.to_string()).collect()),
            gt_1: Some(gt.0),
            gt_2: Some(gt.1),
            dp: Some(dp),
            ad: o(ad),
            f1r2: None,
            f2r1: None,
            pl: o(pl),
        }
    }

    fn cap_lake(dir: &std::path::Path) -> Manifest {
        cap_lake_with(dir, &[])
    }

    /// `cap_lake` with `extra` catalog rows (position, allele, count),
    /// merged into the alleles export in its order.
    fn cap_lake_with(dir: &std::path::Path, extra: &[(i32, &'static str, i64)]) -> Manifest {
        let s1 = vec![
            (
                102,
                row(
                    &["C", "xA", "xG", "*"],
                    (1, 2),
                    13,
                    &[2, 5, 6, 0],
                    &[40, 30, 50, 35, 0, 60, 70, 80, 90, 99],
                    1,
                ),
            ),
            (
                200,
                row(
                    &["G", "xA", "xC", "*"],
                    (1, 2),
                    7,
                    &[0, 3, 4, 0],
                    &[50, 20, 40, 10, 0, 30, 60, 60, 60, 60],
                    1,
                ),
            ),
        ];
        let s2 = vec![(
            102,
            row(
                &["C", "xT", "*"],
                (1, 1),
                9,
                &[1, 8, 0],
                &[200, 20, 0, 210, 25, 230],
                2,
            ),
        )];
        let s4 = vec![(
            100,
            row(
                &["A", "-ACGT", "*"],
                (1, 1),
                11,
                &[0, 11, 0],
                &[300, 40, 0, 310, 45, 320],
                4,
            ),
        )];
        let b3 = vec![Block {
            sample_id: 3,
            start: 1,
            end: 1000,
            ad_1: Some(7),
            ad_2: Some(1),
            dp: Some(8),
            pl_1: Some(5),
            pl_2: Some(15),
            pl_3: Some(30),
            sample_counts: Some(1),
        }];
        let ssi = vec![
            site_file(dir, "s1.parquet", 0, &s1, 10),
            site_file(dir, "s2.parquet", 0, &s2, 10),
            site_file(dir, "s4.parquet", 0, &s4, 10),
        ];
        let srb = vec![block_file(dir, "b3.parquet", 0, &b3, 10)];
        let site = |p: i32| {
            (
                p,
                Some(50.0),
                Some(10),
                Some(36000.0),
                Some(10),
                Some([3, 4, 5, 6]),
                None,
                5,
                1,
            )
        };
        let sites = sites_export(
            &dir.join("sites.parquet"),
            &[site(100), site(102), site(200)],
            10,
        );
        let mut catalog = vec![
            (100, "*", 0, None),
            (100, "-ACGT", 2, Some((100, 104))),
            (100, "A", 0, None),
            (102, "*", 0, None),
            (102, "C", 2, None),
            (102, "xA", 1, None),
            (102, "xG", 1, None),
            (102, "xT", 2, None),
            (200, "*", 0, None),
            (200, "G", 2, None),
            (200, "xA", 1, None),
            (200, "xC", 1, None),
        ];
        catalog.extend(extra.iter().map(|&(p, a, c)| (p, a, c, None)));
        catalog.sort_by(|a, b| (a.0, a.1.as_bytes()).cmp(&(b.0, b.1.as_bytes())));
        let alleles = alleles_export(&dir.join("alleles.parquet"), &catalog, 10);
        manifest(
            &dir.join("out.vcf"),
            &[(1, "S1"), (2, "S2"), (3, "S3"), (4, "S4")],
            ("chrC", 1000),
            sites,
            alleles,
            ssi,
            srb,
        )
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    /// The run at `cap` (threads, window): the output text and summary.
    fn call(m: &Manifest, cap: usize, threads: usize, window: i64) -> (String, serde_json::Value) {
        let opts = RunOpts {
            max_alt_alleles: cap,
            ..RunOpts::default()
        };
        let s = run_with(m, threads, window, &rt(), &opts).unwrap();
        (
            std::fs::read_to_string(m.output.as_ref().unwrap()).unwrap(),
            s,
        )
    }

    /// The record at `pos`: (ALT, INFO, the sample columns).
    fn record(text: &str, pos: i32) -> (String, String, Vec<String>) {
        let line = text
            .lines()
            .find(|l| l.split('\t').nth(1) == Some(&pos.to_string()))
            .unwrap_or_else(|| panic!("no record at {pos}"));
        let f: Vec<&str> = line.split('\t').collect();
        (
            f[4].to_string(),
            f[7].to_string(),
            f[9..].iter().map(|x| x.to_string()).collect(),
        )
    }

    fn counts(s: &serde_json::Value) -> [u64; 4] {
        let c = &s["cap"];
        [
            "records_capped",
            "genotypes_no_called",
            "dropped_alt_observations",
            "kept_alt_observations_lost",
        ]
        .map(|k| c[k].as_u64().unwrap())
    }

    #[test]
    fn uncapped_record_is_today_s_and_the_summary_has_no_cap() {
        let dir = tmpdir("cap-zero");
        let m = cap_lake(&dir);
        let (zero, s) = call(&m, 0, 1, 0);
        assert!(s.get("cap").is_none());
        assert_eq!(s["cross_check"]["records"], 3);
        let (alt, info, cols) = record(&zero, 102);
        assert_eq!(alt, "A,G,T,*");
        assert!(
            info.starts_with("AC=1,1,2,2;AF=0.125,0.125,0.25,0.25;AN=8;"),
            "{info}"
        );
        // The uncapped ref-block fill is not normalized (its PL has no 0).
        assert_eq!(
            cols[2],
            "0/0:7,1,1,1,1:8:5,15,30,15,30,30,15,30,30,30,15,30,30,30,30"
        );
        assert_eq!(
            cols[3],
            "4/4:0,0,0,0,11:11:300,310,320,310,320,320,310,320,320,320,40,45,45,45,0"
        );
        // At or above the record's ALT count: the uncapped bytes, the
        // counts zero, every record compared.
        for cap in [4, 5, 100] {
            let (text, s) = call(&m, cap, 2, 50);
            assert_eq!(text, zero, "cap {cap}");
            assert_eq!(counts(&s), [0; 4]);
            assert_eq!(s["cap"]["max_alt_alleles"], cap);
            assert_eq!(s["cross_check"]["records"], 3);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn capped_records_are_projections() {
        let dir = tmpdir("cap-some");
        let m = cap_lake(&dir);
        let (zero, _) = call(&m, 0, 1, 0);
        // N = 3: kept A, T, * (A and G tie at 1: A, the earlier); S1's A/G
        // carries G: ./.; its PL over [C, A, T, *] all > 0: minus 30.
        let (three, s) = call(&m, 3, 1, 0);
        let (alt, info, cols) = record(&three, 102);
        assert_eq!(alt, "A,T,*");
        assert!(
            info.starts_with("AC=0,2,2;AF=0.0,0.333,0.333;AN=6;"),
            "{info}"
        );
        assert_eq!(
            cols,
            [
                "./.:2,5,0,0:13:10,0,20,40,50,69,40,50,69,69",
                "2/2:1,0,8,0:9:200,210,230,20,25,0,210,230,25,230",
                "0/0:7,1,1,1:8:5,15,30,15,30,30,15,30,30,30",
                "3/3:0,0,0,11:11:300,310,320,310,320,320,40,45,45,0",
            ]
        );
        assert_eq!(counts(&s), [1, 1, 1, 1]);
        // Not compared: the cross-check's records are the other two.
        assert_eq!(s["cross_check"]["records"], 2);
        // N = 2: kept T, * (both 2); the fill not normalized (the one PL
        // rule: its PL holds no 0); the star call over [C, T, *].
        let (two, s) = call(&m, 2, 2, 50);
        let (alt, info, cols) = record(&two, 102);
        assert_eq!(alt, "T,*");
        assert!(info.starts_with("AC=2,2;AF=0.333,0.333;AN=6;"), "{info}");
        assert_eq!(
            cols,
            [
                "./.:2,0,0:13:0,30,59,30,59,59",
                "1/1:1,8,0:9:200,20,0,210,25,230",
                "0/0:7,1,1:8:5,15,30,15,30,30",
                "2/2:0,0,11:11:300,310,320,40,45,0",
            ]
        );
        assert_eq!(counts(&s), [1, 1, 2, 0]);
        // A record at or under the cap is the uncapped one, its fill not
        // normalized.
        assert_eq!(record(&two, 200), record(&zero, 200));
        assert_eq!(record(&two, 100), record(&zero, 100));
        assert!(record(&two, 200).2[2].ends_with(":5,15,30,15,30,30"));
        // N = 1: kept T (T and * tie at 2: T); S4's `*` dropped: ./., AD
        // slot 0, PL prices T with its own `*` (300, 310, 320 less 300);
        // 200 is capped too (A, C tie: A), S1's A/C ./. losing its A.
        let (one, s) = call(&m, 1, 4, 1);
        let (alt, info, cols) = record(&one, 102);
        assert_eq!(alt, "T");
        assert!(info.starts_with("AC=2;AF=0.5;AN=4;"), "{info}");
        assert_eq!(
            cols,
            [
                "./.:2,0:13:0,30,59",
                "1/1:1,8:9:200,20,0",
                "0/0:7,1:8:5,15,30",
                "./.:0,0:11:0,10,20",
            ]
        );
        let (alt, info, cols) = record(&one, 200);
        assert_eq!(alt, "A");
        assert!(info.starts_with("AC=0;AF=0.0;AN=2;"), "{info}");
        // S1's PL over [G, A] (50, 20, 40) less 20; S3's fill as it is.
        assert_eq!(cols[0], "./.:0,3:7:30,0,20");
        assert_eq!(cols[2], "0/0:7,1:8:5,15,30");
        assert_eq!(counts(&s), [2, 3, 5, 1]);
        assert_eq!(s["cross_check"]["records"], 1);
        // The site columns other than ALT and INFO's AC / AF / AN, and the
        // record count, are the uncapped record's.
        let rest = |t: &str, p: i32| {
            let (_, info, _) = record(t, p);
            info.split(';').skip(3).collect::<Vec<_>>().join(";")
        };
        for t in [&one, &two, &three] {
            assert_eq!(rest(t, 102), rest(&zero, 102));
            assert_eq!(t.lines().count(), zero.lines().count());
        }
        // The same bytes at every thread count and window.
        for (cap, want) in [(1, &one), (2, &two), (3, &three)] {
            for (threads, window) in [(1, 0), (2, 1), (4, 101), (3, 1000)] {
                assert_eq!(&call(&m, cap, threads, window).0, want, "cap {cap}");
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `m` restricted to the subset `keep` (K4, `joint_call.py --samples`):
    /// the manifest's samples and `#CHROM` columns are the subset's, and
    /// `subset` is set.
    fn subset_of(m: &Manifest, keep: &[(i32, &str)], out: &std::path::Path) -> Manifest {
        let mut s = m.clone();
        s.output = Some(out.display().to_string());
        s.samples = keep
            .iter()
            .map(|(id, name)| crate::manifest::Sample {
                id: *id,
                name: name.to_string(),
            })
            .collect();
        let h = s.header_lines.as_mut().unwrap();
        let names: Vec<&str> = keep.iter().map(|k| k.1).collect();
        *h.last_mut().unwrap() = format!(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}",
            names.join("\t")
        );
        s.subset = true;
        s
    }

    /// The record at `pos`: its FORMAT column.
    fn format_of(text: &str, pos: i32) -> String {
        let line = text
            .lines()
            .find(|l| l.split('\t').nth(1) == Some(&pos.to_string()))
            .unwrap();
        line.split('\t').nth(8).unwrap().to_string()
    }

    /// K4's contract on the crafted lake, the lines derived by hand: a
    /// subset record is the whole-cohort record's projection onto the
    /// subset's columns. `*` (S4's star call at 102) leaves the ALT list
    /// when S4 is not in the subset and stays when it is; a FORMAT key no
    /// subset column carries is dropped; AC / AN / AF are the subset's;
    /// the catalog cross-check is not run (`cross_check` null); the reader
    /// keeps only the subset's rows (the positive mask).
    #[test]
    fn subset_records_are_projections() {
        let dir = tmpdir("subset");
        let m = cap_lake(&dir);
        let (whole, s) = call(&m, 0, 1, 0);
        assert_eq!(s["subset"], false);
        assert_eq!(s["cross_check"]["records"], 3);
        let rest = |t: &str, p: i32| {
            let (_, info, _) = record(t, p);
            info.split(';').skip(3).collect::<Vec<_>>().join(";")
        };
        let (_, _, whole_cols) = record(&whole, 102);

        // {S1, S3}: S4 (the only `*` carrier) is out: `*` dropped. S1's
        // A/G keeps its GT, AD loses `*`'s entry, PL the genotypes over
        // [C, A, G, T] (the first 10; it holds a 0); S3's fill as it is.
        let sub = subset_of(&m, &[(1, "S1"), (3, "S3")], &dir.join("s13.vcf"));
        let (text, s) = call(&sub, 0, 1, 0);
        assert_eq!(s["subset"], true);
        assert_eq!(s["samples"], 2);
        assert!(s["cross_check"].is_null());
        let (alt, info, cols) = record(&text, 102);
        assert_eq!(alt, "A,G,T");
        assert!(
            info.starts_with("AC=1,1,0;AF=0.25,0.25,0.0;AN=4;"),
            "{info}"
        );
        assert_eq!(rest(&text, 102), rest(&whole, 102));
        assert_eq!(format_of(&text, 102), "GT:AD:DP:PL");
        assert_eq!(
            cols,
            [
                "1/2:2,5,6,0:13:40,30,50,35,0,60,70,80,90,99",
                "0/0:7,1,1,1:8:5,15,30,15,30,30,15,30,30,30",
            ]
        );
        // The whole-cohort columns with `*`'s AD entry and genotypes cut.
        assert!(whole_cols[0].starts_with("1/2:2,5,6,0,0:13:40,30,50,35,0,60,70,80,90,99,"));
        assert!(whole_cols[2].starts_with("0/0:7,1,1,1,1:8:5,15,30,15,30,30,15,30,30,30,"));
        // Every cohort site is a record; the site rows merged are the
        // subset's only (S1's two of the four).
        assert_eq!(text.lines().count(), whole.lines().count());
        let merged = |s: &serde_json::Value| s["tables"]["sample_site_info"]["rows_merged"].clone();
        assert_eq!(merged(&s), 2);
        assert_eq!(merged(&call(&m, 0, 1, 0).1), 4);

        // {S2, S4}: `*` stays (S4 */*); the columns are the whole-cohort
        // ones; at 200 neither has evidence: FORMAT GT alone, AN 0.
        let sub = subset_of(&m, &[(2, "S2"), (4, "S4")], &dir.join("s24.vcf"));
        let (text, _) = call(&sub, 0, 2, 50);
        let (alt, info, cols) = record(&text, 102);
        assert_eq!(alt, "A,G,T,*");
        assert!(
            info.starts_with("AC=0,0,2,2;AF=0.0,0.0,0.5,0.5;AN=4;"),
            "{info}"
        );
        assert_eq!(cols, [whole_cols[1].clone(), whole_cols[3].clone()]);
        let (alt, info, cols) = record(&text, 200);
        assert_eq!(alt, "A,C");
        assert!(info.starts_with("AC=0,0;AF=0.0,0.0;AN=0;"), "{info}");
        assert_eq!(format_of(&text, 200), "GT");
        assert_eq!(cols, ["./.", "./."]);

        // {S2}: `*` dropped at 102; T/T's PL over [C, A, G, T] holds its 0.
        let sub = subset_of(&m, &[(2, "S2")], &dir.join("s2.vcf"));
        let (text, _) = call(&sub, 0, 1, 0);
        let (alt, info, cols) = record(&text, 102);
        assert_eq!(alt, "A,G,T");
        assert!(info.starts_with("AC=0,0,2;AF=0.0,0.0,1.0;AN=2;"), "{info}");
        assert_eq!(cols, ["3/3:1,0,0,8:9:200,210,230,210,230,230,20,25,25,0"]);
        assert_eq!(format_of(&text, 100), "GT");
        assert_eq!(format_of(&text, 200), "GT");

        // {S1, S3} under a cap of 2: ranked by the subset's own AC (A 1,
        // G 1, T 0: A, G kept); the fill not normalized.
        let sub = subset_of(&m, &[(1, "S1"), (3, "S3")], &dir.join("s13c.vcf"));
        let (text, s) = call(&sub, 2, 1, 0);
        let (alt, info, cols) = record(&text, 102);
        assert_eq!(alt, "A,G");
        assert!(info.starts_with("AC=1,1;AF=0.25,0.25;AN=4;"), "{info}");
        assert_eq!(
            cols,
            [
                "1/2:2,5,6:13:40,30,50,35,0,60",
                "0/0:7,1,1:8:5,15,30,15,30,30"
            ]
        );
        assert_eq!(counts(&s), [1, 0, 0, 0]);
        assert!(s["cross_check"].is_null());

        // The same bytes at every thread count and window.
        let sub = subset_of(&m, &[(1, "S1"), (3, "S3")], &dir.join("s13t.vcf"));
        let (want, _) = call(&sub, 0, 1, 0);
        for (threads, window) in [(2, 1), (4, 101), (3, 1000)] {
            assert_eq!(call(&sub, 0, threads, window).0, want);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ── The trim and --min-ac ─────────────────────────────────────────────

    /// `cap_lake` plus a catalog allele no sample carries at 102, `xCA`
    /// (ALT CA, the second of [A, CA, G, T, *]: AC 1, 0, 1, 2, 2), with a
    /// catalog count of 1, so the cross-check finds the record differing
    /// on its untrimmed alleles (catalog 1, AC 0).
    fn trim_lake(dir: &std::path::Path) -> Manifest {
        cap_lake_with(dir, &[(102, "xCA", 1)])
    }

    /// The run under `opts` (threads, window): the output text and summary.
    fn call_opts(
        m: &Manifest,
        opts: RunOpts,
        threads: usize,
        window: i64,
    ) -> (String, serde_json::Value) {
        let s = run_with(m, threads, window, &rt(), &opts).unwrap();
        (
            std::fs::read_to_string(m.output.as_ref().unwrap()).unwrap(),
            s,
        )
    }

    fn opts(trim: bool, cap: usize, min_ac: u64) -> RunOpts {
        RunOpts {
            trim_alt_alleles: trim,
            max_alt_alleles: cap,
            min_ac,
            ..RunOpts::default()
        }
    }

    fn trim_counts(s: &serde_json::Value) -> [u64; 2] {
        ["records_trimmed", "alts_dropped"].map(|k| s["trim"][k].as_u64().unwrap())
    }

    fn positions(text: &str) -> Vec<i32> {
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split('\t').nth(1).unwrap().parse().unwrap())
            .collect()
    }

    /// The trim on the crafted lake, the lines derived by hand: CA (AC 0,
    /// in the middle) leaves 102, whose GTs are re-indexed (S1 A/G 1/3 ->
    /// 1/2, S2 T/T 4/4 -> 3/3, S4 */* 5/5 -> 4/4) and whose AD / PL lose
    /// CA's entries: the record is `cap_lake`'s 102, where CA never was.
    /// The other records are unchanged. The cross-check compares 102 on
    /// its untrimmed alleles: the same counts and lines as without the
    /// trim. The summary carries the trim's counts only under the trim.
    #[test]
    fn trimmed_records_are_projections() {
        let dir = tmpdir("trim");
        let m = trim_lake(&dir);
        let (whole, s) = call_opts(&m, opts(false, 0, 0), 1, 0);
        assert!(s.get("trim").is_none() && s.get("min_ac").is_none());
        let (alt, info, cols) = record(&whole, 102);
        assert_eq!(alt, "A,CA,G,T,*");
        assert!(
            info.starts_with("AC=1,0,1,2,2;AF=0.125,0.0,0.125,0.25,0.25;AN=8;"),
            "{info}"
        );
        // S1's PL over [C, A, CA, G, T, *], CA priced with its own `*`.
        assert_eq!(
            cols[0],
            "1/3:2,5,0,6,0,0:13:40,30,50,70,80,99,35,0,90,60,70,80,99,90,99,70,80,99,90,99,99"
        );
        let cc = s["cross_check"].clone();
        assert_eq!(cc["records"], 3);
        assert_eq!(cc["differing"]["other"], 1);
        assert_eq!(cc["first"][0], "chrC:102");

        let base = tmpdir("trim-base");
        let (plain, _) = call(&cap_lake(&base), 0, 1, 0);
        let (trimmed, s) = call_opts(&m, opts(true, 0, 0), 1, 0);
        assert_eq!(record(&trimmed, 102), record(&plain, 102));
        let (alt, info, cols) = record(&trimmed, 102);
        assert_eq!(alt, "A,G,T,*");
        assert!(
            info.starts_with("AC=1,1,2,2;AF=0.125,0.125,0.25,0.25;AN=8;"),
            "{info}"
        );
        assert_eq!(
            cols[0],
            "1/2:2,5,6,0,0:13:40,30,50,35,0,60,70,80,90,99,70,80,90,99,99"
        );
        assert_eq!(cols[1].split(':').next(), Some("3/3"));
        assert_eq!(cols[3].split(':').next(), Some("4/4"));
        for p in [100, 200] {
            assert_eq!(record(&trimmed, p), record(&whole, p), "{p}");
        }
        assert_eq!(trim_counts(&s), [1, 1]);
        assert!(s.get("cap").is_none() && s.get("min_ac").is_none());
        // The cross-check unchanged by the trim.
        assert_eq!(s["cross_check"], cc);
        // The same bytes at every thread count and window.
        for (threads, window) in [(2, 1), (4, 101), (3, 1000)] {
            assert_eq!(call_opts(&m, opts(true, 0, 0), threads, window).0, trimmed);
        }
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// Trim then cap: the cap ranks the trimmed record. At N = 4 the
    /// untrimmed 102 (5 ALTs) is capped (CA dropped: the same text as the
    /// trim's, but counted as capped and not compared), while the trimmed
    /// one (4 ALTs) is not; at N = 2 both keep T, * and the cap's counts
    /// are over the trimmed record.
    #[test]
    fn trim_then_cap() {
        let dir = tmpdir("trim-cap");
        let m = trim_lake(&dir);
        let (trimmed, _) = call_opts(&m, opts(true, 0, 0), 1, 0);
        let (cap4, s) = call_opts(&m, opts(false, 4, 0), 1, 0);
        assert_eq!(record(&cap4, 102), record(&trimmed, 102));
        assert_eq!(counts(&s), [1, 0, 0, 0]);
        assert_eq!(s["cross_check"]["records"], 2);
        let (both4, s) = call_opts(&m, opts(true, 4, 0), 2, 50);
        assert_eq!(both4, trimmed);
        assert_eq!(counts(&s), [0, 0, 0, 0]);
        assert_eq!(trim_counts(&s), [1, 1]);
        assert_eq!(s["cross_check"]["records"], 3);
        assert_eq!(s["cross_check"]["differing"]["other"], 1);
        // N = 2: cap_lake's capped 102 (T, *), with the counts it had.
        let base = tmpdir("trim-cap-base");
        let (plain2, _) = call(&cap_lake(&base), 2, 1, 0);
        let (both2, s) = call_opts(&m, opts(true, 2, 0), 4, 1);
        assert_eq!(record(&both2, 102), record(&plain2, 102));
        assert_eq!(counts(&s), [1, 1, 2, 0]);
        assert_eq!(trim_counts(&s), [1, 1]);
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// --min-ac on the written genotypes: 100 (S4 1/1: 2 observations),
    /// 102 (6), 200 (S1 A/C: 2). At K = 3 only 102 is written; the left-out
    /// records are not counted by the cross-check. At a cap of 1, 200's
    /// only carrier is no-called (A/C, C dropped): its written AC is 0, so
    /// K = 1 leaves it out, which it does not uncapped. The trim's and the
    /// cap's counts include the records left out.
    #[test]
    fn min_ac_leaves_records_out_before_their_calls() {
        let dir = tmpdir("min-ac");
        let m = trim_lake(&dir);
        let (whole, _) = call_opts(&m, opts(false, 0, 0), 1, 0);
        assert_eq!(positions(&whole), [100, 102, 200]);
        let (one, s) = call_opts(&m, opts(false, 0, 1), 1, 0);
        assert_eq!(one, whole);
        assert_eq!(s["min_ac"]["min_ac"], 1);
        assert_eq!(s["min_ac"]["records_left_out"], 0);
        assert_eq!(s["records"], 3);
        let (three, s) = call_opts(&m, opts(false, 0, 3), 2, 50);
        assert_eq!(positions(&three), [102]);
        assert_eq!(record(&three, 102), record(&whole, 102));
        assert_eq!(s["min_ac"]["records_left_out"], 2);
        assert_eq!(s["records"], 1);
        assert_eq!(s["cross_check"]["records"], 1);
        assert!(s.get("trim").is_none() && s.get("cap").is_none());
        // At 6, 102 is still written; at 7 nothing is.
        assert_eq!(positions(&call_opts(&m, opts(false, 0, 6), 1, 0).0), [102]);
        let (none, s) = call_opts(&m, opts(false, 0, 7), 1, 0);
        assert!(positions(&none).is_empty());
        assert_eq!(s["min_ac"]["records_left_out"], 3);
        assert_eq!(s["cross_check"]["records"], 0);
        // Under a cap of 1: 200's carrier no-called, left out at K = 1; 102
        // (T kept: T/T 2) and 100 stay.
        let (c1, _) = call_opts(&m, opts(false, 1, 0), 1, 0);
        assert!(record(&c1, 200).1.starts_with("AC=0;AF=0.0;AN=2;"));
        let (c1k1, s) = call_opts(&m, opts(false, 1, 1), 4, 1);
        assert_eq!(positions(&c1k1), [100, 102]);
        for p in [100, 102] {
            assert_eq!(record(&c1k1, p), record(&c1, p));
        }
        assert_eq!(s["min_ac"]["records_left_out"], 1);
        // The cap's counts include the record left out (102 and 200 capped).
        assert_eq!(
            counts(&s),
            counts(&call_opts(&m, opts(false, 1, 0), 1, 0).1)
        );
        assert_eq!(counts(&s)[0], 2);
        // Trim, cap and min-ac together; the trim's counts include the
        // records left out.
        let (all, s) = call_opts(&m, opts(true, 1, 3), 3, 1000);
        assert_eq!(positions(&all), Vec::<i32>::new());
        assert_eq!(trim_counts(&s), [1, 1]);
        assert_eq!(s["min_ac"]["records_left_out"], 3);
        let (all, s) = call_opts(&m, opts(true, 2, 3), 1, 0);
        assert_eq!(positions(&all), [102]);
        assert!(record(&all, 102).1.starts_with("AC=2,2;"));
        assert_eq!(s["min_ac"]["records_left_out"], 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The trim under a subset: {S3} (a fill only) leaves every ALT of
    /// every record with AC 0: ALT `.`, no AC / AF, the fill's AD / PL
    /// over REF alone (5: its PL held no 0); --min-ac 1 then writes
    /// nothing. {S2, S3}: 102 keeps T alone (S2's T/T re-indexed 4/4 ->
    /// 1/1), `*` already gone with S4.
    #[test]
    fn trim_under_a_subset() {
        let dir = tmpdir("trim-subset");
        let m = trim_lake(&dir);
        let sub = subset_of(&m, &[(3, "S3")], &dir.join("s3.vcf"));
        let (text, s) = call_opts(&sub, opts(true, 0, 0), 1, 0);
        let (alt, info, cols) = record(&text, 102);
        assert_eq!(alt, ".");
        assert!(info.starts_with("AN=2;"), "{info}");
        assert_eq!(cols, ["0/0:7:8:5"]);
        assert_eq!(record(&text, 100).0, ".");
        assert_eq!(record(&text, 200).0, ".");
        assert_eq!(trim_counts(&s), [3, 1 + 4 + 2]);
        assert!(s["cross_check"].is_null());
        let (none, s) = call_opts(&sub, opts(true, 0, 1), 2, 50);
        assert!(positions(&none).is_empty());
        assert_eq!(s["min_ac"]["records_left_out"], 3);
        let sub = subset_of(&m, &[(2, "S2"), (3, "S3")], &dir.join("s23.vcf"));
        let (text, _) = call_opts(&sub, opts(true, 0, 0), 1, 0);
        let (alt, info, cols) = record(&text, 102);
        assert_eq!(alt, "T");
        assert!(info.starts_with("AC=2;AF=0.5;AN=4;"), "{info}");
        assert_eq!(cols, ["1/1:1,8:9:200,20,0", "0/0:7,1:8:5,15,30"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod cross_check_tests {
    //! The catalog cross-check's counting, from
    //! `tests/unit/test_joint_call_info.py::test_cross_check_counts_and_reasons`
    //! and `::test_cross_check_lists_at_most_limit_sites` (ported when the
    //! Python engine was removed), and its two predicates
    //! (`catalog_ac_differs`, `own_call_unresolved`), which the Python
    //! `check()` evaluated itself. The reason labels are the binary's
    //! (`stale`, `unresolved`, `other`); the driver maps them to its
    //! closing lines' text.
    use super::*;

    fn counts(cc: &CrossCheck) -> Vec<(&str, u64)> {
        cc.differing.iter().map(|(k, v)| (k.as_str(), *v)).collect()
    }

    fn alts(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn differs_sets_the_star_alt_aside() {
        // The Python test's first case: the `*` ALT's counts differ, the
        // record does not.
        assert!(!catalog_ac_differs(&alts(&["C", "*"]), &[2, 9], &[2, 1]));
        // A differing non-star ALT.
        assert!(catalog_ac_differs(&alts(&["C", "*"]), &[3, 9], &[2, 9]));
        assert!(catalog_ac_differs(&alts(&["C", "G"]), &[2, 9], &[2, 1]));
        assert!(!catalog_ac_differs(&alts(&["C"]), &[3], &[3]));
        assert!(!catalog_ac_differs(&[], &[], &[]));
    }

    fn call(gt: &str) -> Call {
        Call {
            gt: gt.to_string(),
            ad: None,
            dp: None,
            f1r2: None,
            f2r1: None,
            pl: None,
        }
    }

    fn row(gt_1: Option<i32>, gt_2: Option<i32>) -> SiteRow {
        SiteRow {
            sample_id: 1,
            alleles: Some(alts(&["A", "xC", "*"])),
            gt_1,
            gt_2,
            ..Default::default()
        }
    }

    #[test]
    fn unresolved_is_a_called_row_written_no_call() {
        let called = row(Some(1), Some(2)); // xC/* : `*` is no VCF allele
        let null_gt = row(Some(1), None);
        let outside = row(Some(1), Some(7)); // an index past the list
        // A call written ./. with both alleles in its row.
        assert!(own_call_unresolved(&[call("./.")], &[Some(&called)]));
        // One GT index null, or not naming an allele of the row: not
        // unresolved.
        assert!(!own_call_unresolved(&[call("./.")], &[Some(&null_gt)]));
        assert!(!own_call_unresolved(&[call("./.")], &[Some(&outside)]));
        // A called genotype; a ./. without an own row (a ref-block or
        // missing fill).
        assert!(!own_call_unresolved(&[call("0/1")], &[Some(&called)]));
        assert!(!own_call_unresolved(&[call("./.")], &[None]));
        // Any one column suffices.
        assert!(own_call_unresolved(
            &[call("0/1"), call("./.")],
            &[Some(&called), Some(&called)]
        ));
    }

    #[test]
    fn counts_and_reasons() {
        let mut cc = CrossCheck::default();
        cc.check(None, "chr21:1".into(), false, false); // the `*` aside
        cc.check(None, "chr21:2".into(), true, false);
        cc.check(None, "chr21:3".into(), true, true);
        assert_eq!(cc.records, 3);
        assert_eq!(counts(&cc), [("other", 1), ("unresolved", 1)]);
        assert_eq!(cc.first, ["chr21:2", "chr21:3"]);

        // The run's reason (stale) wins over `unresolved` / `other`.
        let mut stale = CrossCheck::default();
        stale.check(Some("stale"), "chr21:2".into(), true, true);
        assert_eq!(counts(&stale), [("stale", 1)]);

        let mut clean = CrossCheck::default();
        clean.check(None, "chr21:1".into(), false, false);
        assert_eq!((clean.records, clean.differing.len()), (1, 0));
    }

    #[test]
    fn the_run_reason_is_stale_or_none() {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "samples": [{"id": 1, "name": "S1"}, {"id": 2, "name": "S2"}],
            "removed_ids": [3],
            "contigs": [],
        }))
        .unwrap();
        // Tombstoned samples alone give no reason (K1).
        assert_eq!(RunCtx::new(&m).run_reason, None);
        m.stale = true;
        assert_eq!(RunCtx::new(&m).run_reason, Some("stale"));
        m.removed_ids.clear();
        assert_eq!(RunCtx::new(&m).run_reason, Some("stale"));
    }

    #[test]
    fn lists_at_most_limit_sites() {
        let mut cc = CrossCheck::default();
        for p in 0..25 {
            cc.check(None, format!("chr21:{p}"), true, false);
        }
        assert_eq!(cc.first.len(), CROSS_CHECK_LIMIT);
        assert_eq!(CROSS_CHECK_LIMIT, 10);
        assert_eq!(counts(&cc), [("other", 25)]);
        // Units merged in order keep the limit and sum the counts.
        let mut all = CrossCheck::default();
        all.add(&cc);
        all.add(&cc);
        assert_eq!((all.records, all.first.len()), (50, 10));
        assert_eq!(counts(&all), [("other", 50)]);
    }
}
