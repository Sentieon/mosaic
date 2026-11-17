// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The site logic (originally a 1:1 port of the Python joint-call
//! engine's, which `joint_call.py` held):
//! `reconstruct_alleles`, `find_overlapping_deletions`,
//! `calculate_star_samples`, `odds_ratio`, `calculate_info_fields`,
//! `catalog_allele_counts` (the cross-check's AC), `genotype_allele_counts`
//! and `process_variant_site`. The functions keep the Python names; the
//! Python docstrings' rules are the comments. Where the Python engine
//! compares synthesized allele ids (`AlleleIds`), this port compares the
//! allele strings of one site, which is what the ids stand for.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::exports::{CatalogAllele, Site};
use crate::fisher::Fisher;
use crate::pyfmt::py_round;
use crate::rows::SiteRow;

/// A catalog allele with a non-ref prefix (`x`, `+`, `-`, `*`).
pub fn is_nonref(a: &str) -> bool {
    a.starts_with(['x', '+', '-', '*'])
}

/// The result of `reconstruct_alleles`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconstructed {
    /// `(trimmed, local_ref, local_alt)` per parsed catalog allele.
    pub parsed: Vec<(String, String, String)>,
    /// The longest reference sequence (the first of the longest).
    pub max_ref: String,
    /// Database encoding → full VCF allele.
    pub trimmed_to_full: HashMap<String, String>,
    /// The distinct ALT alleles (not `max_ref`, not <NON_REF>), sorted.
    pub final_alts: Vec<String>,
}

/// The full VCF allele of a parsed allele: `local_alt` padded with the
/// part of `max_ref` past `local_ref` (<NON_REF> unpadded).
fn full_allele(local_ref: &str, local_alt: &str, max_ref: &str) -> String {
    if local_alt == "<NON_REF>" {
        return "<NON_REF>".to_string();
    }
    format!(
        "{local_alt}{}",
        max_ref.get(local_ref.len()..).unwrap_or("")
    )
}

/// `reconstruct_alleles(db_alleles, base_ref)`: parse database alleles and
/// reconstruct VCF-compatible allele strings. The base ref itself is
/// `(base_ref, base_ref)`; `x` / `+`: `(base_ref, a[1:])`; `-`:
/// `(a[1:], base_ref)`; `*`: `(base_ref, "<NON_REF>")`; any other string is
/// dropped (the Python `if` chain). `max_ref` is the first longest
/// `local_ref` (`max(key=len)`); `final_alts` the distinct fulls other than
/// `max_ref` and <NON_REF>, sorted (bytewise: Python's `sorted` on ASCII).
pub fn reconstruct_alleles(db_alleles: &[&str], base_ref: &str) -> Reconstructed {
    let mut parsed: Vec<(String, String, String)> = Vec::new();
    for &a in db_alleles {
        let p = if a == base_ref {
            (base_ref, base_ref)
        } else if let Some(rest) = a.strip_prefix('x') {
            (base_ref, rest)
        } else if let Some(rest) = a.strip_prefix('+') {
            (base_ref, rest)
        } else if let Some(rest) = a.strip_prefix('-') {
            (rest, base_ref)
        } else if a.starts_with('*') {
            (base_ref, "<NON_REF>")
        } else {
            continue;
        };
        parsed.push((a.to_string(), p.0.to_string(), p.1.to_string()));
    }
    // Find max reference length for padding (the first of the longest).
    // (The base ref is always parsed, so `parsed` is never empty from
    // `process_variant_site`.)
    let mut max_ref: &str = parsed.first().map_or("", |p| p.1.as_str());
    for (_, r, _) in parsed.iter().skip(1) {
        if r.len() > max_ref.len() {
            max_ref = r;
        }
    }
    let max_ref = max_ref.to_string();
    // Pad alts and build mappings.
    let mut trimmed_to_full = HashMap::new();
    let mut final_alts: Vec<String> = Vec::new();
    for (trimmed, r, alt) in &parsed {
        let full = full_allele(r, alt, &max_ref);
        if full != max_ref && full != "<NON_REF>" {
            final_alts.push(full.clone());
        }
        trimmed_to_full.insert(trimmed.clone(), full);
    }
    // Unique and sort.
    final_alts.sort();
    final_alts.dedup();
    Reconstructed {
        parsed,
        max_ref,
        trimmed_to_full,
        final_alts,
    }
}

/// A catalog deletion allele (the Python `del_alleles` tuple without the
/// chromosome): its site, its string and `[del_start, del_end)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deletion {
    pub pos: i32,
    pub allele: String,
    pub del_start: i32,
    pub del_end: i32,
}

/// The deletions from other sites that span a position.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Overlapping {
    /// `(site, allele)` of every spanning deletion allele.
    pub alleles: Vec<(i32, String)>,
    /// The spanning deletions' sites with their `del_start`, in first-seen
    /// order (the Python dict).
    pub sites: Vec<(i32, i32)>,
}

impl Overlapping {
    /// Is `allele` at `site` a spanning deletion allele
    /// (`aid in overlapping_del_aids`; an id names its site)?
    pub fn contains(&self, site: i32, allele: &str) -> bool {
        self.alleles.iter().any(|(s, a)| *s == site && a == allele)
    }
}

/// `find_overlapping_deletions`: the deletion alleles from other sites
/// that span `pos` (`a_pos != pos and del_start <= pos < del_end`), over
/// the deletions sorted by `del_start` (iteration stops at the first
/// `del_start > pos`).
pub fn find_overlapping_deletions(dels: &[Deletion], pos: i32) -> Overlapping {
    let mut out = Overlapping::default();
    for d in dels {
        // Once del_start > pos, no more deletions can overlap.
        if d.del_start > pos {
            break;
        }
        if d.pos != pos && d.del_start <= pos && pos < d.del_end {
            out.alleles.push((d.pos, d.allele.clone()));
            if !out.sites.iter().any(|(s, _)| *s == d.pos) {
                out.sites.push((d.pos, d.del_start));
            }
        }
    }
    out
}

/// A deletion site's context (the Python `DeletionContext` per site): the
/// catalog's non-ref alleles there and its active samples' rows, in file
/// order.
#[derive(Debug, Default, Clone)]
pub struct DelSite {
    pub nonref: HashSet<String>,
    pub rows: Vec<SiteRow>,
}

/// A star sample: its count (1 het, 2 hom) and its row at the deletion
/// site (`star_sample_data`).
#[derive(Debug, Clone, Copy)]
pub struct Star<'a> {
    pub count: i32,
    pub site: i32,
    pub row: &'a SiteRow,
}

/// `calculate_star_samples`: which samples need a `*` at this position.
/// For the spanning deletions' sites in `del_start` order, for each row at
/// the site: `count` = how many of its `gt_1` / `gt_2` resolve to an
/// allele in the site's non-ref set (every catalog allele there with a
/// non-ref prefix: a sample with any non-ref call at a deletion site is a
/// `*` carrier, the first site that makes it one wins; odd, ported as is).
/// Samples with their own row at the position are then dropped.
/// `column` maps a sample id to its header column (active samples only);
/// `has_own_row[col]` says whether the sample has a row at the position.
pub fn calculate_star_samples<'a>(
    overlapping: &Overlapping,
    del_sites: &'a HashMap<i32, DelSite>,
    column: &HashMap<i32, usize>,
    has_own_row: &[bool],
) -> Vec<Option<Star<'a>>> {
    let mut stars: Vec<Option<Star<'a>>> = vec![None; has_own_row.len()];
    if overlapping.alleles.is_empty() {
        return stars;
    }
    // Process deletion sites in order of position (first site wins); a
    // stable sort on del_start.
    let mut sites = overlapping.sites.clone();
    sites.sort_by_key(|&(_, del_start)| del_start);
    for (site, _) in sites {
        let Some(ctx) = del_sites.get(&site) else {
            continue;
        };
        for row in &ctx.rows {
            let hit = |gt| row.gt_allele(gt).is_some_and(|a| ctx.nonref.contains(a));
            let count = i32::from(hit(row.gt_1)) + i32::from(hit(row.gt_2));
            if count > 0
                && let Some(&c) = column.get(&row.sample_id)
                && stars[c].is_none()
            {
                stars[c] = Some(Star { count, site, row });
            }
        }
    }
    // Filter out samples that already have variant data.
    for (c, s) in stars.iter_mut().enumerate() {
        if has_own_row[c] {
            *s = None;
        }
    }
    stars
}

/// `int / int` as Python's true division: the correctly rounded quotient.
/// Exact for operands up to 2^53 (both converted exactly, one IEEE
/// division); counts beyond that are not met (strand counts of a site).
fn int_div(a: i64, b: i64) -> f64 {
    a as f64 / b as f64
}

/// `odds_ratio(table)`: the symmetric odds ratio for strand bias, on the
/// table with 1 added to each cell.
pub fn odds_ratio(table: [i64; 4]) -> f64 {
    let t0 = table[0] + 1;
    let t1 = table[1] + 1;
    let t2 = table[2] + 1;
    let t3 = table[3] + 1;
    let ratio = int_div(t0 * t3, t1 * t2) + int_div(t1 * t2, t0 * t3);
    let r = int_div(t0.min(t1), t0.max(t1));
    let a = int_div(t2.min(t3), t2.max(t3));
    (ratio * r / a).ln()
}

/// The INFO fields of `calculate_info_fields` (DP, QD, MQ, FS, SOR,
/// InbreedingCoeff), each None when the Python engine does not write it.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Info {
    pub dp: Option<i64>,
    pub qd: Option<f64>,
    pub mq: Option<f64>,
    pub fs: Option<f64>,
    pub sor: Option<f64>,
    pub inbreeding_coeff: Option<f64>,
}

/// `calculate_info_fields`. `total_dp = (dp or 0) + rb_dp`; DP when > 0,
/// then QD when `qual` is not null; MQ when `mqsq_2 > 0` (nulls read as
/// 0); FS and SOR when `sb` is present (an all-zero table included);
/// InbreedingCoeff when `gtc` is present and sums to ≥ 10. Every float
/// through `_round3` (`py_round`: no negative zero).
pub fn calculate_info_fields(site: &Site, fisher: &mut Fisher) -> Info {
    let mut info = Info::default();
    let total_dp = site.dp.unwrap_or(0) + site.rb_dp;
    if total_dp > 0 {
        info.dp = Some(total_dp);
        if let Some(q) = site.qual {
            info.qd = Some(py_round(q / total_dp as f64));
        }
    }
    let mqsq_2 = site.mqsq_2.unwrap_or(0);
    if mqsq_2 > 0 {
        let mqsq_1 = site.mqsq_1.unwrap_or(0.0);
        info.mq = Some(py_round((mqsq_1 / mqsq_2 as f64).sqrt()));
    }
    if let Some(sb) = site.sb {
        info.fs = Some(fisher.fs(sb));
        info.sor = Some(py_round(odds_ratio(sb)));
    }
    if let Some(t) = site.gtc
        && t.iter().map(|&x| i128::from(x)).sum::<i128>() >= 10
    {
        let denom = t[0] + t[1] + t[2];
        if denom > 0 {
            let p = (t[0] as f64 + t[1] as f64 * 0.5) / denom as f64;
            let s = denom as f64 * p * (1.0 - p) * 2.0;
            if s > 0.0 {
                info.inbreeding_coeff = Some(py_round(1.0 - t[1] as f64 / s));
            }
        }
    }
    info
}

/// The AC of `catalog_allele_counts` (the catalog cross-check's input):
/// per ALT, the summed `alleles.count` of the catalog alleles whose full
/// VCF allele it is (`*`: the star count, which the cross-check skips).
/// The Python function's AN / AF are not used by the cross-check and are
/// not ported.
pub fn catalog_allele_counts(
    final_alts: &[String],
    rec: &Reconstructed,
    allele_to_count: &HashMap<&str, i64>,
    star_count: i64,
) -> Vec<i64> {
    final_alts
        .iter()
        .map(|alt| {
            if alt == "*" {
                return star_count;
            }
            rec.parsed
                .iter()
                .filter(|(_, r, a)| &full_allele(r, a, &rec.max_ref) == alt)
                .map(|(t, _, _)| allele_to_count.get(t.as_str()).copied().unwrap_or(0))
                .sum()
        })
        .collect()
}

/// `genotype_allele_counts(gts, n_alts)`: AC / AN / AF from the record's
/// emitted FORMAT/GT strings (`bcftools +fill-tags` semantics): AN the
/// called alleles (`./.` 0, `0/1` 2), AC[i] the calls of ALT i + 1 (`*`
/// the ALT it is), `AF[i] = round(AC[i] / AN, 3)`, 0.0 when AN is 0.
pub fn genotype_allele_counts(gts: &[&str], n_alts: usize) -> (Vec<i64>, i64, Vec<f64>) {
    let mut ac = vec![0i64; n_alts];
    let mut an = 0i64;
    for gt in gts {
        for a in gt.split(['/', '|']) {
            if a == "." || a.is_empty() {
                continue;
            }
            let idx: usize = a.parse().expect("a GT the engine wrote");
            an += 1;
            if idx > 0 {
                ac[idx - 1] += 1;
            }
        }
    }
    let af = ac
        .iter()
        .map(|&c| {
            if an > 0 {
                py_round(int_div(c, an))
            } else {
                0.0
            }
        })
        .collect();
    (ac, an, af)
}

/// What `process_variant_site` reads at a site.
pub struct SiteInputs<'a> {
    /// The catalog's rows at the site, in export order.
    pub catalog: &'a [CatalogAllele],
    /// Per header column: the sample's own row at the site (the last).
    pub own: &'a [Option<&'a SiteRow>],
    /// The active deletions (`del_start` order).
    pub deletions: &'a [Deletion],
    pub del_sites: &'a HashMap<i32, DelSite>,
    pub column: &'a HashMap<i32, usize>,
}

/// What `process_variant_site` decides for a site: the VCF alleles and
/// the mappings the sample calls use.
pub struct SiteCall<'a> {
    pub max_ref: String,
    pub final_alts: Vec<String>,
    pub vcf_alleles: Vec<String>,
    pub catalog: HashSet<&'a str>,
    pub trimmed_to_full: HashMap<String, String>,
    pub full_to_idx: HashMap<String, usize>,
    pub overlapping: Overlapping,
    pub stars: Vec<Option<Star<'a>>>,
    pub info: Info,
    pub catalog_ac: Vec<i64>,
    /// Set when the ALT cap or the trim restricted the record (K3, K5,
    /// `restrict_site`): the unrestricted record's allele mapping;
    /// `final_alts`, `vcf_alleles` and `full_to_idx` above are then the
    /// kept alleles'.
    pub uncapped: Option<Uncapped>,
}

/// The uncapped record's alleles of a capped site (K3): what a call needs
/// to decide as the uncapped call decides (`map_pl_values`' "is any
/// genotype of the full allele list priced"), without its genotypes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uncapped {
    /// The uncapped record's VCF allele count (REF included).
    pub n_alleles: usize,
    /// Full VCF allele → its uncapped VCF index.
    pub full_to_idx: HashMap<String, usize>,
}

impl SiteCall<'_> {
    /// `get_idx`: a row allele (a catalog string at the site) → its VCF
    /// index through `trimmed_to_full` and `full_to_idx` (at a capped
    /// record: its index among the kept alleles; a dropped ALT: None).
    pub fn get_idx(&self, allele: &str) -> Option<usize> {
        if !self.catalog.contains(allele) {
            return None;
        }
        let full = self.trimmed_to_full.get(allele)?;
        self.full_to_idx.get(full).copied()
    }

    /// `get_idx` over the uncapped record's alleles (the same as `get_idx`
    /// at a record the cap did not restrict).
    pub fn get_idx_uncapped(&self, allele: &str) -> Option<usize> {
        let Some(u) = &self.uncapped else {
            return self.get_idx(allele);
        };
        if !self.catalog.contains(allele) {
            return None;
        }
        let full = self.trimmed_to_full.get(allele)?;
        u.full_to_idx.get(full).copied()
    }

    /// The VCF allele count of the uncapped record.
    pub fn n_alleles_uncapped(&self) -> usize {
        self.uncapped
            .as_ref()
            .map_or(self.vcf_alleles.len(), |u| u.n_alleles)
    }
}

/// The cap's counts (K3), summed over records: the records capped, the
/// genotypes the cap wrote `./.` (called in the uncapped record, `./.` in
/// the capped one), the ALT observations on the dropped alleles (their
/// uncapped AC summed) and the kept-allele ALT observations lost with those
/// genotypes. The test side's `tests/vcf_project.py` computes the same four
/// from the uncapped text.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapCounts {
    pub records_capped: u64,
    pub genotypes_no_called: u64,
    pub dropped_alt_observations: u64,
    pub kept_alt_observations_lost: u64,
}

impl CapCounts {
    pub fn add(&mut self, o: &CapCounts) {
        self.records_capped += o.records_capped;
        self.genotypes_no_called += o.genotypes_no_called;
        self.dropped_alt_observations += o.dropped_alt_observations;
        self.kept_alt_observations_lost += o.kept_alt_observations_lost;
    }
}

/// The ranking pass (the ALT cap's): each column's
/// uncapped GT as a pair of VCF allele indices, without building a call.
/// A sample's own row: its two allele lookups (`get_idx`; either
/// unresolved: None, the `./.` `create_variant_call` writes); else a star
/// sample: `0/s` or `s/s` by its count (`s` the index of `*`); else (a ref
/// block or no evidence) None: it counts no ALT. Called by `restrict_site`.
pub fn uncapped_gt_pairs(
    sc: &SiteCall<'_>,
    own: &[Option<&SiteRow>],
) -> Vec<Option<(usize, usize)>> {
    let star_idx = sc.vcf_alleles.iter().position(|a| a == "*");
    own.iter()
        .enumerate()
        .map(|(c, o)| {
            if let Some(row) = o {
                let idx = |gt| row.gt_allele(gt).and_then(|a| sc.get_idx(a));
                match (idx(row.gt_1), idx(row.gt_2)) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None,
                }
            } else if let Some(s) = &sc.stars[c] {
                let si = star_idx.expect("a star sample's record lists *");
                Some(if s.count >= 2 { (si, si) } else { (0, si) })
            } else {
                None
            }
        })
        .collect()
}

/// AC per ALT from the ranking pass's pairs (equal to
/// `genotype_allele_counts` over the uncapped calls' GTs).
pub fn ac_of_pairs(pairs: &[Option<(usize, usize)>], n_alts: usize) -> Vec<i64> {
    let mut ac = vec![0i64; n_alts];
    for &(a, b) in pairs.iter().flatten() {
        for i in [a, b] {
            if i > 0 {
                ac[i - 1] += 1;
            }
        }
    }
    ac
}

/// The kept ALTs (0-based ALT indices): the `cap` with the largest AC, ties
/// to the earlier ALT, in their original order.
pub fn kept_alts(ac: &[i64], cap: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ac.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(ac[i]), i));
    order.truncate(cap);
    order.sort_unstable();
    order
}

/// The trim's counts (K5, `--trim-alt-alleles`), summed over records: the
/// records it changed and the ALTs it dropped. The test side's
/// `tests/vcf_project.py::project_trim` computes the same two.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TrimCounts {
    pub records_trimmed: u64,
    pub alts_dropped: u64,
}

impl TrimCounts {
    pub fn add(&mut self, o: &TrimCounts) {
        self.records_trimmed += o.records_trimmed;
        self.alts_dropped += o.alts_dropped;
    }
}

/// What `restrict_site` is asked for (K3, K5): the trim, the cap (0 = none)
/// and whether the record's written ALT observations are needed
/// (`--min-ac`).
#[derive(Debug, Default, Clone, Copy)]
pub struct Restrict {
    pub trim: bool,
    pub cap: usize,
    pub written_ac: bool,
}

/// What `restrict_site` did to a record.
#[derive(Debug, Default)]
pub struct Restricted {
    /// The trim's counts for this record (zero when it dropped nothing).
    pub trim: TrimCounts,
    /// Set when the cap restricted the (trimmed) record: its counts.
    pub cap: Option<CapCounts>,
    /// The ALT observations the record writes (the sum of its INFO/AC),
    /// when asked for.
    pub written_ac: Option<u64>,
    /// At a record the trim restricted and the cap did not: its
    /// unrestricted ALTs and their AC from the ranking pass, the catalog
    /// cross-check's input (a trimmed record is compared on its untrimmed
    /// alleles, so the trim alone does not move the cross-check).
    pub untrimmed: Option<(Vec<String>, Vec<i64>)>,
}

/// The trim, then the ALT cap, as one restriction of the site's alleles
/// before any call is built (the ALT cap's mechanism, also used by the
/// trim; each a projection of the record before it, in the order trim,
/// cap): one ranking pass
/// (`uncapped_gt_pairs`) gives the unrestricted record's AC. The trim keeps
/// the ALTs whose AC is > 0; the cap then keeps, of those, the `cap` with
/// the largest AC (`kept_alts`, ties to the earlier ALT) when more remain.
/// The site's allele list is restricted to REF and the kept ALTs, so every
/// call is built at the restricted size (a GT allele that is no kept
/// allele resolves to nothing: `./.`; the trim drops no allele a GT names).
/// The written AC follows from the pairs and the kept alleles (a pair with
/// a dropped allele is written `./.`), so a record `--min-ac` leaves out is
/// known before its calls are built. Nothing asked (no trim, no cap over
/// the record's ALTs, no written AC): nothing is computed or changed.
pub fn restrict_site(sc: &mut SiteCall<'_>, own: &[Option<&SiteRow>], o: Restrict) -> Restricted {
    let n_alts = sc.final_alts.len();
    let mut r = Restricted::default();
    if !(o.trim || (o.cap > 0 && n_alts > o.cap) || o.written_ac) {
        return r;
    }
    let pairs = uncapped_gt_pairs(sc, own);
    let ac = ac_of_pairs(&pairs, n_alts);
    // The trim: the ALTs with AC > 0.
    let mut kept: Vec<usize> = (0..n_alts).filter(|&i| !o.trim || ac[i] > 0).collect();
    if kept.len() < n_alts {
        r.trim = TrimCounts {
            records_trimmed: 1,
            alts_dropped: (n_alts - kept.len()) as u64,
        };
    }
    // The cap, ranked on the trimmed record.
    let capped = o.cap > 0 && kept.len() > o.cap;
    if capped {
        let sub: Vec<i64> = kept.iter().map(|&i| ac[i]).collect();
        kept = kept_alts(&sub, o.cap).iter().map(|&j| kept[j]).collect();
    }
    let mut is_kept = vec![false; n_alts + 1];
    is_kept[0] = true;
    for &k in &kept {
        is_kept[k + 1] = true;
    }
    if capped {
        // Over the trimmed record: the ALTs the trim dropped have AC 0 and
        // no genotype names them, so they add nothing here.
        let mut counts = CapCounts {
            records_capped: 1,
            ..CapCounts::default()
        };
        counts.dropped_alt_observations = ac
            .iter()
            .enumerate()
            .filter(|(i, _)| !is_kept[i + 1])
            .map(|(_, &c)| c as u64)
            .sum();
        for &(a, b) in pairs.iter().flatten() {
            if !is_kept[a] || !is_kept[b] {
                counts.genotypes_no_called += 1;
                counts.kept_alt_observations_lost +=
                    [a, b].iter().filter(|&&i| i > 0 && is_kept[i]).count() as u64;
            }
        }
        r.cap = Some(counts);
    }
    if o.written_ac {
        r.written_ac = Some(
            pairs
                .iter()
                .flatten()
                .filter(|&&(a, b)| is_kept[a] && is_kept[b])
                .map(|&(a, b)| u64::from(a > 0) + u64::from(b > 0))
                .sum(),
        );
    }
    if kept.len() < n_alts {
        let full_alts = restrict_to(sc, &kept);
        if !capped {
            r.untrimmed = Some((full_alts, ac));
        }
    }
    r
}

/// Restrict the site's alleles to REF and the `kept` ALTs (0-based, in
/// order), keeping the unrestricted mapping as `uncapped`; returns the
/// unrestricted ALTs.
fn restrict_to(sc: &mut SiteCall<'_>, kept: &[usize]) -> Vec<String> {
    let final_alts: Vec<String> = kept.iter().map(|&k| sc.final_alts[k].clone()).collect();
    let mut vcf_alleles = vec![sc.max_ref.clone()];
    vcf_alleles.extend(final_alts.iter().cloned());
    let full_to_idx = vcf_alleles
        .iter()
        .enumerate()
        .map(|(i, a)| (a.clone(), i))
        .collect();
    let n_alleles = sc.vcf_alleles.len();
    sc.uncapped = Some(Uncapped {
        n_alleles,
        full_to_idx: std::mem::replace(&mut sc.full_to_idx, full_to_idx),
    });
    sc.vcf_alleles = vcf_alleles;
    std::mem::replace(&mut sc.final_alts, final_alts)
}

/// `process_variant_site`: the site's VCF alleles, INFO and call context,
/// or None (and the warning) when its catalog has no base ref.
pub fn process_variant_site<'a>(
    site: &Site,
    inp: &SiteInputs<'a>,
    fisher: &mut Fisher,
) -> Option<SiteCall<'a>> {
    let db_alleles: Vec<&str> = inp.catalog.iter().map(|r| r.allele.as_str()).collect();
    let mut allele_to_count: HashMap<&str, i64> = HashMap::new();
    for r in inp.catalog {
        allele_to_count.insert(r.allele.as_str(), r.count);
    }
    // Find base reference allele.
    let base_ref = db_alleles
        .iter()
        .find(|a| !a.is_empty() && !is_nonref(a))
        .copied()?;
    let rec = reconstruct_alleles(&db_alleles, base_ref);
    let overlapping = find_overlapping_deletions(inp.deletions, site.position);
    let has_own: Vec<bool> = inp.own.iter().map(Option::is_some).collect();
    let stars = calculate_star_samples(&overlapping, inp.del_sites, inp.column, &has_own);
    let mut final_alts = rec.final_alts.clone();
    let star_count: i64 = stars.iter().flatten().map(|s| i64::from(s.count)).sum();
    if stars.iter().any(Option::is_some) {
        final_alts.push("*".to_string());
    }
    let mut vcf_alleles = vec![rec.max_ref.clone()];
    vcf_alleles.extend(final_alts.iter().cloned());
    let mut full_to_idx = HashMap::new();
    for (i, a) in vcf_alleles.iter().enumerate() {
        full_to_idx.insert(a.clone(), i);
    }
    let info = calculate_info_fields(site, fisher);
    let catalog_ac = catalog_allele_counts(&final_alts, &rec, &allele_to_count, star_count);
    Some(SiteCall {
        max_ref: rec.max_ref.clone(),
        final_alts,
        vcf_alleles,
        catalog: db_alleles.into_iter().collect(),
        trimmed_to_full: rec.trimmed_to_full,
        full_to_idx,
        overlapping,
        stars,
        info,
        catalog_ac,
        uncapped: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn reconstruct_pads_sorts_and_drops() {
        let r = reconstruct_alleles(&["A", "-AT", "-ATG", "xC", "+G", "*", "G"], "A");
        // The first longest local ref; "G" (no prefix, not the base) dropped.
        assert_eq!(r.max_ref, "ATG");
        assert_eq!(r.parsed.len(), 6);
        assert_eq!(r.trimmed_to_full["A"], "ATG");
        assert_eq!(r.trimmed_to_full["-AT"], "AG");
        assert_eq!(r.trimmed_to_full["-ATG"], "A");
        assert_eq!(r.trimmed_to_full["xC"], "CTG");
        assert_eq!(r.trimmed_to_full["+G"], "GTG");
        assert_eq!(r.trimmed_to_full["*"], "<NON_REF>");
        assert!(!r.trimmed_to_full.contains_key("G"));
        // Distinct, sorted bytewise, max_ref and <NON_REF> left out.
        assert_eq!(r.final_alts, s(&["A", "AG", "CTG", "GTG"]));
    }

    #[test]
    fn reconstruct_first_longest_ref_wins() {
        let r = reconstruct_alleles(&["A", "-AC", "-AT", "xG"], "A");
        assert_eq!(r.max_ref, "AC");
        assert_eq!(r.trimmed_to_full["-AT"], "A");
        assert_eq!(r.trimmed_to_full["xG"], "GC");
        assert_eq!(r.final_alts, s(&["A", "GC"]));
        // Duplicate fulls collapse.
        let r = reconstruct_alleles(&["C", "xT", "+T"], "C");
        assert_eq!(r.final_alts, s(&["T"]));
    }

    fn del(pos: i32, allele: &str, len: i32) -> Deletion {
        Deletion {
            pos,
            allele: allele.to_string(),
            del_start: pos,
            del_end: pos + len,
        }
    }

    #[test]
    fn overlapping_deletions_skip_own_site_and_stop() {
        let dels = vec![
            del(10, "-ACG", 3),
            del(10, "-ACGTT", 5),
            del(12, "-GT", 2),
            del(20, "-TTT", 3),
        ];
        let o = find_overlapping_deletions(&dels, 12);
        assert_eq!(o.alleles, vec![(10, "-ACG".into()), (10, "-ACGTT".into())]);
        assert_eq!(o.sites, vec![(10, 10)]);
        let o = find_overlapping_deletions(&dels, 13);
        assert_eq!(o.sites, vec![(10, 10), (12, 12)]);
        assert!(o.contains(12, "-GT") && !o.contains(10, "-GT"));
        assert!(find_overlapping_deletions(&dels, 15).alleles.is_empty());
    }

    fn row(sample: i32, alleles: &[&str], gt: (Option<i32>, Option<i32>)) -> SiteRow {
        SiteRow {
            sample_id: sample,
            alleles: Some(s(alleles)),
            gt_1: gt.0,
            gt_2: gt.1,
            ..Default::default()
        }
    }

    #[test]
    fn star_samples_first_site_wins_nonref_set_and_own_row() {
        let mut del_sites = HashMap::new();
        let nonref = |v: &[&str]| v.iter().map(|x| x.to_string()).collect();
        // Site 10: sample 1 het for the deletion, sample 2 hom-ref,
        // sample 3 with a SNP call (a non-ref allele: a star carrier, the
        // odd rule), sample 4 with its `*` called.
        del_sites.insert(
            10,
            DelSite {
                nonref: nonref(&["-ACGTT", "xT", "*"]),
                rows: vec![
                    row(1, &["A", "-ACGTT", "*"], (Some(0), Some(1))),
                    row(2, &["A", "*"], (Some(0), Some(0))),
                    row(3, &["A", "xT", "*"], (Some(1), Some(1))),
                    row(4, &["A", "*"], (Some(0), Some(1))),
                ],
            },
        );
        // Site 11: sample 1 hom for another deletion (the first site wins:
        // its count stays 1), sample 5 hom.
        del_sites.insert(
            11,
            DelSite {
                nonref: nonref(&["-CG", "*"]),
                rows: vec![
                    row(1, &["C", "-CG", "*"], (Some(1), Some(1))),
                    row(5, &["C", "-CG", "*"], (Some(1), Some(1))),
                ],
            },
        );
        let o = Overlapping {
            alleles: vec![(11, "-CG".into()), (10, "-ACGTT".into())],
            sites: vec![(11, 11), (10, 10)],
        };
        let column: HashMap<i32, usize> = (1..=5).map(|i| (i, (i - 1) as usize)).collect();
        let mut own = vec![false; 5];
        let st = calculate_star_samples(&o, &del_sites, &column, &own);
        let counts: Vec<Option<(i32, i32)>> =
            st.iter().map(|x| x.map(|s| (s.count, s.site))).collect();
        assert_eq!(
            counts,
            vec![
                Some((1, 10)),
                None,
                Some((2, 10)),
                Some((1, 10)),
                Some((2, 11))
            ]
        );
        // A sample with its own row at the site is dropped.
        own[2] = true;
        let st = calculate_star_samples(&o, &del_sites, &column, &own);
        assert!(st[2].is_none() && st[0].is_some());
    }

    fn site(sb: Option<[i64; 4]>, gtc: Option<[i64; 3]>) -> Site {
        Site {
            position: 1,
            qual: Some(30.5),
            dp: Some(10),
            mqsq_1: Some(36000.0),
            mqsq_2: Some(10),
            sb,
            gtc,
            rb_dp: 5,
            rb_sample_counts: 1,
        }
    }

    #[test]
    fn info_fields_and_the_zero_sign() {
        let mut f = Fisher::default();
        // FS at p = 1 is -10 * log10(1) = -0.0, written 0.0.
        let i = calculate_info_fields(&site(Some([0, 0, 3, 4]), None), &mut f);
        assert_eq!(i.dp, Some(15));
        assert_eq!(i.qd, Some(py_round(30.5 / 15.0)));
        assert_eq!(i.mq, Some(60.0));
        let fs = i.fs.unwrap();
        assert!(fs == 0.0 && fs.is_sign_positive());
        // An all-zero sb still runs the test (p = 1) and SOR.
        let i = calculate_info_fields(&site(Some([0, 0, 0, 0]), None), &mut f);
        assert!(i.fs == Some(0.0) && i.sor.is_some());
        // No DP: no QD; no mqsq_2: no MQ; no sb: no FS / SOR.
        let mut s0 = site(None, None);
        s0.dp = None;
        s0.rb_dp = 0;
        s0.mqsq_2 = None;
        let i = calculate_info_fields(&s0, &mut f);
        assert_eq!(i, Info::default());
        // SOR cannot be negative: with u = t0 / t1 and v = t3 / t2 (the
        // +1 table), ratio · ref / alt is one of u² + 1/v², v² + 1/u²,
        // (uv)² + 1, 1 + 1/(uv)², all > 1. So no SOR rounds to zero from
        // below (the port's first design expected one could); checked here
        // on a grid.
        for a in 0..30 {
            for b in 0..30 {
                for c in [0, 1, 4, 13, 29] {
                    for d in [0, 2, 7, 30] {
                        assert!(odds_ratio([a, b, c, d]) > 0.0);
                    }
                }
            }
        }
        // An InbreedingCoeff in (-0.0005, 0) rounds to 0.0; Σ gtc < 10:
        // none written.
        let mut found = false;
        'ic: for t0 in 0..200i64 {
            for t1 in 0..200i64 {
                let t2 = 7;
                let denom = (t0 + t1 + t2) as f64;
                let p = (t0 as f64 + t1 as f64 * 0.5) / denom;
                let s = denom * p * (1.0 - p) * 2.0;
                let x = 1.0 - t1 as f64 / s;
                if x < 0.0 && x > -0.0005 {
                    let i = calculate_info_fields(&site(None, Some([t0, t1, t2])), &mut f);
                    let ic = i.inbreeding_coeff.unwrap();
                    assert!(ic == 0.0 && ic.is_sign_positive());
                    found = true;
                    break 'ic;
                }
            }
        }
        assert!(found);
        let i = calculate_info_fields(&site(None, Some([3, 3, 3])), &mut f);
        assert_eq!(i.inbreeding_coeff, None);
        let i = calculate_info_fields(&site(None, Some([5, 0, 5])), &mut f);
        assert_eq!(i.inbreeding_coeff, Some(1.0));
    }

    #[test]
    fn allele_counts_from_genotypes() {
        let (ac, an, af) = genotype_allele_counts(&["0/1", "./.", "1/2", "2/2", "0|1"], 2);
        assert_eq!(ac, vec![3, 3]);
        assert_eq!(an, 8);
        assert_eq!(af, vec![py_round(3.0 / 8.0), py_round(3.0 / 8.0)]);
        let (ac, an, af) = genotype_allele_counts(&["./.", "./."], 1);
        assert_eq!((ac, an, af), (vec![0], 0, vec![0.0]));
        let (ac, an, af) = genotype_allele_counts(&["0/0"], 0);
        assert_eq!((ac.len(), an, af.len()), (0, 2, 0));
    }

    /// Replaces `tests/unit/test_joint_call_info.py::test_genotype_allele_counts`
    /// (every parametrized case, ported when the Python engine was removed):
    /// the GT strings, the ALT count, and AC / AN / AF as the Python engine
    /// computed them (AF the Python values, as literals).
    #[test]
    fn genotype_allele_counts_python_cases() {
        type Case<'a> = (&'a [&'a str], usize, &'a [i64], i64, &'a [f64]);
        let cases: [Case; 12] = [
            (&["./."], 1, &[0], 0, &[0.0]),
            (&["0/0"], 1, &[0], 2, &[0.0]),
            (&["0/1"], 1, &[1], 2, &[0.5]),
            (&["1/1"], 1, &[2], 2, &[1.0]),
            (&["1/2"], 2, &[1, 1], 2, &[0.5, 0.5]),
            (&["0/0", "0/1", "1/1", "./."], 1, &[3], 6, &[0.5]),
            (&["1/2", "2/2", "0/2"], 2, &[1, 4], 6, &[0.167, 0.667]),
            // A star call 0/s and s/s (s = 2: the record's * ALT).
            (&["0/2"], 2, &[0, 1], 2, &[0.0, 0.5]),
            (&["2/2", "0/1"], 2, &[1, 2], 4, &[0.25, 0.5]),
            (&["0|1"], 1, &[1], 2, &[0.5]),
            (&[], 1, &[0], 0, &[0.0]),
            (&[], 0, &[], 0, &[]),
        ];
        for (gts, n_alts, want_ac, want_an, want_af) in cases {
            let (ac, an, af) = genotype_allele_counts(gts, n_alts);
            assert_eq!(ac, want_ac, "{gts:?}");
            assert_eq!(an, want_an, "{gts:?}");
            assert!(ac.iter().sum::<i64>() <= an, "{gts:?}");
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&af), bits(want_af), "{gts:?}");
        }
    }

    /// Replaces `test_joint_call_info.py::test_af_rounded_and_zero_when_no_called_allele`
    /// and `::test_no_call_fill_is_not_counted_in_an` (K1).
    #[test]
    fn af_rounded_zero_without_calls_and_no_call_fill_not_in_an() {
        let (ac, an, af) = genotype_allele_counts(&["0/1", "0/0", "0/0"], 1);
        assert_eq!((ac, an), (vec![1], 6));
        assert_eq!(af[0].to_bits(), 0.167f64.to_bits());
        let (ac, an, af) = genotype_allele_counts(&["./.", "./."], 2);
        assert_eq!((ac, an), (vec![0, 0], 0));
        assert_eq!(af.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), [0; 2]);
        assert_eq!(crate::pyfmt::py_str_float(af[0]), "0.0");
        // Two hom-ref fills, one no-call fill, one het call.
        let (ac, an, af) = genotype_allele_counts(&["0/0", "./.", "0/0", "0/1"], 1);
        assert_eq!((ac, an), (vec![1], 6));
        assert_eq!(af[0].to_bits(), 0.167f64.to_bits());
    }

    /// `calculate_info_fields(qual, dp_si, dp_rb, mqsq, sb, gtc)` of the
    /// Python tests as a `Site`.
    fn py_site(
        qual: Option<f64>,
        dp: i64,
        rb_dp: i64,
        mqsq: (f64, i64),
        sb: Option<[i64; 4]>,
        gtc: Option<[i64; 3]>,
    ) -> Site {
        Site {
            position: 1,
            qual,
            dp: Some(dp),
            mqsq_1: Some(mqsq.0),
            mqsq_2: Some(mqsq.1),
            sb,
            gtc,
            rb_dp,
            rb_sample_counts: 0,
        }
    }

    fn bits(x: Option<f64>) -> Option<u64> {
        x.map(f64::to_bits)
    }

    /// Replaces `test_joint_call_info.py::test_fs_zero_not_negative_zero_on_an_unbiased_table`
    /// and `::test_fs_and_sor_on_a_biased_table` (K1): the same inputs
    /// (`calculate_info_fields(10.0, 5, 5, [100, 1], sb, None)`), every field
    /// the Python engine wrote, as its bits (FS 108.384 is scipy's
    /// `round(-10 * log10(p), 3)` for [[20, 0], [0, 20]], SOR
    /// `round(odds_ratio(sb), 3)`).
    #[test]
    fn info_fields_python_fs_and_sor_cases() {
        let mut f = Fisher::default();
        for (sb, fs, sor) in [
            ([0, 0, 0, 0], 0.0, 0.693),
            ([5, 5, 5, 5], 0.0, 0.693),
            ([20, 0, 0, 20], 108.384, 6.089),
        ] {
            let i = calculate_info_fields(
                &py_site(Some(10.0), 5, 5, (100.0, 1), Some(sb), None),
                &mut f,
            );
            assert_eq!(i.dp, Some(10), "{sb:?}");
            assert_eq!(bits(i.qd), Some(1.0f64.to_bits()), "{sb:?}");
            assert_eq!(bits(i.mq), Some(10.0f64.to_bits()), "{sb:?}");
            assert_eq!(bits(i.fs), Some(f64::to_bits(fs)), "{sb:?}");
            assert_eq!(bits(i.sor), Some(f64::to_bits(sor)), "{sb:?}");
            assert_eq!(i.inbreeding_coeff, None);
            assert_eq!(
                crate::pyfmt::py_str_float(i.fs.unwrap()),
                if fs == 0.0 { "0.0" } else { "108.384" }
            );
        }
    }

    /// Replaces `test_joint_call_info.py::test_signed_info_fields_keep_their_sign`
    /// (K1): `calculate_info_fields(None, 0, 0, [0, 0], None, gtc)`.
    #[test]
    fn info_fields_python_signed_inbreeding_coeff() {
        let mut f = Fisher::default();
        // Excess heterozygosity: InbreedingCoeff < 0.
        let i = calculate_info_fields(
            &py_site(None, 0, 0, (0.0, 0), None, Some([0, 20, 0])),
            &mut f,
        );
        assert_eq!(
            i,
            Info {
                inbreeding_coeff: Some(-1.0),
                ..Info::default()
            }
        );
        // A value that rounds to zero from below is written 0.0.
        let i = calculate_info_fields(
            &py_site(None, 0, 0, (0.0, 0), None, Some([2500, 5001, 2500])),
            &mut f,
        );
        let ic = i.inbreeding_coeff.unwrap();
        assert_eq!(ic.to_bits(), 0.0f64.to_bits());
        assert_eq!(crate::pyfmt::py_str_float(ic), "0.0");
    }

    /// Replaces `tests/unit/test_allele_encoding.py::test_encode_decode_roundtrip`
    /// (every case) and `::test_decode_non_ref_symbolic` (K1): ingest's
    /// encoder (`mosaic_core::alleles::format_alt_allele`, the Rust worker's)
    /// then this decoder must recover `(ref, [alt])`.
    #[test]
    fn encode_decode_roundtrip() {
        use mosaic_core::alleles::format_alt_allele;
        for (r, alt) in [
            ("A", "T"),
            ("C", "G"),
            ("A", "AT"),
            ("G", "GTTT"),
            ("AT", "A"),
            ("ATGC", "A"),
        ] {
            let base_ref = &r[..1]; // the stored ref allele is the first base
            let encoded = format_alt_allele(r, alt);
            let rec = reconstruct_alleles(&[base_ref, encoded.as_str()], base_ref);
            assert_eq!(rec.max_ref, r, "{r} {alt}");
            assert_eq!(rec.final_alts, s(&[alt]), "{r} {alt}");
            assert_eq!(rec.trimmed_to_full[&encoded], alt, "{r} {alt}");
        }
        // The symbolic non-ref decodes to <NON_REF> and is no real ALT.
        let rec = reconstruct_alleles(&["A", "*"], "A");
        assert_eq!(rec.trimmed_to_full["*"], "<NON_REF>");
        assert!(rec.final_alts.is_empty());
    }

    #[test]
    fn catalog_counts_sum_by_full_allele() {
        let r = reconstruct_alleles(&["C", "xT", "+T", "*"], "C");
        let counts: HashMap<&str, i64> = [("C", 5), ("xT", 2), ("+T", 3), ("*", 9)].into();
        let alts = vec!["T".to_string(), "*".to_string()];
        assert_eq!(catalog_allele_counts(&alts, &r, &counts, 4), vec![5, 4]);
    }
}
