// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The per-sample calls (originally a 1:1 port of the Python joint-call
//! engine's, which `joint_call.py` held): `map_pl_values`, `create_variant_call`, `create_star_call`,
//! `create_refblock_call` and `generate_sample_call`, keeping the Python
//! names and, as comments, the docstrings' rules. Allele ids are allele
//! strings here (one site's ids name its strings).

use std::collections::HashSet;

use crate::call::{Overlapping, SiteCall, Star};
use crate::rows::{Block, SiteRow};

/// The PL of a genotype the sample's evidence cannot price: an allele its
/// record neither lists nor can stand in for with its <NON_REF> (`*`)
/// allele, or a ref block without PL (the Python `PL_FILL`).
pub const PL_FILL: i64 = 990;

/// One sample's FORMAT values; None = the key is not written for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub gt: String,
    pub ad: Option<Vec<i64>>,
    pub dp: Option<i64>,
    pub f1r2: Option<Vec<i64>>,
    pub f2r1: Option<Vec<i64>>,
    pub pl: Option<Vec<i64>>,
}

impl Call {
    fn no_call() -> Call {
        Call {
            gt: "./.".to_string(),
            ad: None,
            dp: None,
            f1r2: None,
            f2r1: None,
            pl: None,
        }
    }
}

/// `_fill_nulls(values, fill)`: each null element replaced by `fill`; a
/// null list stays None.
fn fill_nulls(values: &Option<Vec<Option<i32>>>, fill: i64) -> Option<Vec<i64>> {
    values
        .as_ref()
        .map(|v| v.iter().map(|x| x.map_or(fill, i64::from)).collect())
}

/// The PL index of the diploid genotype `(i, j)` (VCF order).
fn gt_index(a: usize, b: usize) -> usize {
    let (lo, hi) = if a > b { (b, a) } else { (a, b) };
    hi * (hi + 1) / 2 + lo
}

/// PL over every VCF genotype from a VCF → local index map: a genotype
/// whose two local indices exist and whose local PL index is inside the
/// row's PL takes that PL, else None.
fn pl_over_vcf(
    n: usize,
    local: impl Fn(usize) -> Option<usize>,
    stored: &[i64],
) -> Vec<Option<i64>> {
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in 0..=j {
            out.push(match (local(i), local(j)) {
                (Some(li), Some(lj)) => stored.get(gt_index(li, lj)).copied(),
                _ => None,
            });
        }
    }
    out
}

/// Nones → `PL_FILL`; when every PL is > 0 the minimum is subtracted.
fn finish_pl(pl: Vec<Option<i64>>) -> Vec<i64> {
    let mut pl: Vec<i64> = pl.into_iter().map(|p| p.unwrap_or(PL_FILL)).collect();
    normalize_pl(&mut pl);
    pl
}

/// When every PL is > 0 the minimum is subtracted.
fn normalize_pl(pl: &mut [i64]) {
    if pl.iter().all(|&p| p > 0)
        && let Some(&m) = pl.iter().min()
    {
        for p in pl {
            *p -= m;
        }
    }
}

/// Whether the uncapped call of a capped record writes a PL (the ALT
/// cap's rule): whether some genotype of the uncapped record's
/// allele list is priced, decided from the row's own alleles, never by
/// enumerating the record's genotypes. Each uncapped VCF allele prices with
/// a local index: the last local allele mapping to it, else `non_ref`
/// (None: not priced). So the locals reachable are the mapped ones, plus
/// `non_ref` when some uncapped VCF allele is not mapped; a genotype is
/// priced iff its two locals' PL index lies inside the row's PL, and every
/// pair of reachable locals is some genotype's (a local with itself
/// included: an allele with itself).
fn uncapped_any_priced(
    alleles: &[String],
    pl_len: usize,
    non_ref: Option<usize>,
    site: &SiteCall<'_>,
) -> bool {
    // Uncapped VCF index → the last local allele mapping there.
    let mut mapped: Vec<(usize, usize)> = Vec::new();
    for (l, a) in alleles.iter().enumerate() {
        if let Some(v) = site.get_idx_uncapped(a) {
            match mapped.iter_mut().find(|(mv, _)| *mv == v) {
                Some(m) => m.1 = l,
                None => mapped.push((v, l)),
            }
        }
    }
    let mut locals: Vec<usize> = mapped.iter().map(|&(_, l)| l).collect();
    if mapped.len() < site.n_alleles_uncapped()
        && let Some(nr) = non_ref
    {
        locals.push(nr);
    }
    locals
        .iter()
        .any(|&a| locals.iter().any(|&b| gt_index(a, b) < pl_len))
}

/// `map_pl_values`: map PL values from local allele space to VCF allele
/// space. The row's allele list is its RECORD's own (ref, its ALTs in the
/// record's order, `*` for its <NON_REF>) and its PL over it, in VCF
/// genotype order. Every lookup is by allele (row string → catalog string
/// → VCF allele), never by position: a VCF allele the record does not list
/// is priced with the record's `*` (its first allele that is a catalog
/// allele equal to `*` or `<NON_REF>`), and with `PL_FILL` when the record
/// has none. None when the row has no PL or no alleles, or when no
/// genotype could be priced. At a capped record (K3) the PL is over the
/// kept alleles only (the kept genotypes' entries, `PL_FILL` where one is
/// not priced, then normalized), and whether there is one is decided as
/// the uncapped call decides it (`uncapped_any_priced`).
pub fn map_pl_values(
    stored_pl: Option<&[i64]>,
    alleles: Option<&[String]>,
    site: &SiteCall<'_>,
) -> Option<Vec<i64>> {
    let pl = stored_pl.filter(|p| !p.is_empty())?;
    let alleles = alleles.filter(|a| !a.is_empty())?;
    // The local index of <NON_REF> or *.
    let non_ref = alleles
        .iter()
        .position(|a| site.catalog.contains(a.as_str()) && (a == "<NON_REF>" || a == "*"));
    // VCF index → local index (the last local allele mapping there).
    let mut vcf_to_local: Vec<Option<usize>> = vec![None; site.vcf_alleles.len()];
    for (l, a) in alleles.iter().enumerate() {
        if let Some(v) = site.get_idx(a) {
            vcf_to_local[v] = Some(l);
        }
    }
    let n = site.vcf_alleles.len();
    let current = pl_over_vcf(n, |v| vcf_to_local[v].or(non_ref), pl);
    let priced = if site.uncapped.is_some() {
        uncapped_any_priced(alleles, pl.len(), non_ref, site)
    } else {
        current.iter().any(Option::is_some)
    };
    if priced {
        Some(finish_pl(current))
    } else {
        None
    }
}

/// `create_variant_call`: a sample with its own row at the site. GT `i/j`
/// from its `gt_1` / `gt_2` through its allele list → `trimmed_to_full` →
/// `full_to_idx` (either unresolved: `./.`); AD always (over the VCF
/// alleles, by allele, zeros elsewhere), F1R2 / F2R1 the same way when the
/// row has them; DP the row's; PL by `map_pl_values`. Nulls: DP `.`, a
/// depth element 0, a PL element `PL_FILL`.
pub fn create_variant_call(row: &SiteRow, site: &SiteCall<'_>) -> Call {
    let dp = row.dp.map(i64::from);
    let ad = fill_nulls(&row.ad, 0);
    let f1r2 = fill_nulls(&row.f1r2, 0);
    let f2r1 = fill_nulls(&row.f2r1, 0);
    let pl = fill_nulls(&row.pl, PL_FILL);
    let idx = |gt| row.gt_allele(gt).and_then(|a| site.get_idx(a));
    let gt = match (idx(row.gt_1), idx(row.gt_2)) {
        (Some(a), Some(b)) => format!("{a}/{b}"),
        _ => "./.".to_string(),
    };
    let n = site.vcf_alleles.len();
    let alleles: &[String] = row.alleles.as_deref().unwrap_or(&[]);
    let by_allele = |values: &Option<Vec<i64>>| -> Vec<i64> {
        let mut cur = vec![0i64; n];
        if let Some(v) = values
            && !alleles.is_empty()
            && !v.is_empty()
        {
            for (a, &count) in alleles.iter().zip(v) {
                if let Some(vi) = site.get_idx(a) {
                    cur[vi] = count;
                }
            }
        }
        cur
    };
    let nonempty = |v: &Option<Vec<i64>>| v.as_ref().is_some_and(|v| !v.is_empty());
    Call {
        gt,
        ad: Some(by_allele(&ad)),
        dp,
        f1r2: nonempty(&f1r2).then(|| by_allele(&f1r2)),
        f2r1: nonempty(&f2r1).then(|| by_allele(&f2r1)),
        pl: map_pl_values(pl.as_deref(), row.alleles.as_deref(), site),
    }
}

/// `create_star_call`: a sample with a `*` (spanning deletion) genotype.
/// GT `s/s` when the count ≥ 2, else `0/s` (`s` the index of `*`). From
/// the sample's row at the deletion site (its record's own list, with AD /
/// F1R2 / F2R1 / PL as the record carries them), over the VCF alleles,
/// each of AD / F1R2 / F2R1 only when the row has it and an allele list:
/// an allele in the overlapping-deletion set goes to the `*` slot as the
/// max, an allele not in the site's non-ref set adds to slot 0, any other
/// is dropped. DP the row's. PL from three local indices found by allele:
/// the ref (the first allele not in the non-ref set), the star (the
/// overlapping deletion allele with the most AD, ties to the earliest;
/// else the record's first non-ref allele other than its `*`) and the
/// record's own `*` (every other VCF allele is priced with it; absent:
/// `PL_FILL`); the same min-subtraction. No PL when the row has none, or
/// no ref or star index.
///
/// At a capped record whose `*` the cap dropped (`vcf_alleles` then
/// lacks `*`): GT `./.`; AD / F1R2 / F2R1
/// keep slot 0 (the `*` slot's value is dropped with it); the PL prices
/// REF with the ref index and every kept ALT with the record's own `*`, as
/// the uncapped call does, and is written exactly when the uncapped call
/// writes one (a ref and a star index). Built at the capped size.
pub fn create_star_call(
    star: &Star<'_>,
    vcf_alleles: &[String],
    overlapping: &Overlapping,
    nonref: &HashSet<String>,
) -> Call {
    // None only at a capped record that dropped `*` (K3).
    let star_idx = vcf_alleles.iter().position(|a| a == "*");
    let gt = match star_idx {
        Some(s) if star.count >= 2 => format!("{s}/{s}"),
        Some(s) => format!("0/{s}"),
        None => "./.".to_string(),
    };
    let row = star.row;
    let dp = row.dp.map(i64::from);
    let s_ad = fill_nulls(&row.ad, 0).filter(|v| !v.is_empty());
    let s_f1r2 = fill_nulls(&row.f1r2, 0).filter(|v| !v.is_empty());
    let s_f2r1 = fill_nulls(&row.f2r1, 0).filter(|v| !v.is_empty());
    let s_pl = fill_nulls(&row.pl, PL_FILL).filter(|v| !v.is_empty());
    let mut call = Call {
        gt,
        ad: None,
        dp,
        f1r2: None,
        f2r1: None,
        pl: None,
    };
    let Some(alleles) = row.alleles.as_deref().filter(|a| !a.is_empty()) else {
        return call;
    };
    let overlaps = |a: &str| overlapping.contains(star.site, a);
    let fold = |values: &[i64]| -> Vec<i64> {
        let mut cur = vec![0i64; vcf_alleles.len()];
        for (i, a) in alleles.iter().enumerate() {
            if i < values.len() {
                if overlaps(a) {
                    if let Some(s) = star_idx {
                        cur[s] = cur[s].max(values[i]);
                    }
                } else if !nonref.contains(a) {
                    cur[0] += values[i];
                }
            }
        }
        cur
    };
    call.ad = s_ad.as_deref().map(fold);
    call.f1r2 = s_f1r2.as_deref().map(fold);
    call.f2r1 = s_f2r1.as_deref().map(fold);
    if let Some(pl) = s_pl.as_deref() {
        // The record's own `*` (<NON_REF>), looked up by allele.
        let local_nonref = alleles.iter().position(|a| a == "*");
        let local_ref = alleles.iter().position(|a| !nonref.contains(a));
        // The best overlapping deletion allele by AD.
        let mut local_star = None;
        let mut best_ad = -1i64;
        for (li, a) in alleles.iter().enumerate() {
            if overlaps(a) {
                let ad_val = s_ad.as_ref().and_then(|v| v.get(li).copied()).unwrap_or(0);
                if ad_val > best_ad {
                    best_ad = ad_val;
                    local_star = Some(li);
                }
            }
        }
        if local_star.is_none() {
            local_star = alleles
                .iter()
                .enumerate()
                .position(|(li, a)| nonref.contains(a) && Some(li) != local_nonref);
        }
        if let (Some(r), Some(s)) = (local_ref, local_star) {
            let local = |v: usize| {
                if v == 0 {
                    Some(r)
                } else if Some(v) == star_idx {
                    Some(s)
                } else {
                    local_nonref
                }
            };
            call.pl = Some(finish_pl(pl_over_vcf(vcf_alleles.len(), local, pl)));
        }
    }
    call
}

/// `create_refblock_call`: a sample in a reference block. GT `0/0`, or
/// `./.` when the block's `sample_counts` is 0 (a gVCF no-call block,
/// whose call is honored; a null is `0/0`). AD `[ad_1, ad_2, ad_2, ...]`,
/// DP the block's, PL `pl_1` for 0/0, `pl_2` for 0/i, `pl_3` for i/j; no
/// F1R2 / F2R1. Nulls: an AD 0, DP `.`, a PL `PL_FILL`.
pub fn create_refblock_call(b: &Block, n: usize) -> Call {
    let no_call = b.sample_counts == Some(0);
    let ad1 = i64::from(b.ad_1.unwrap_or(0));
    let ad2 = i64::from(b.ad_2.unwrap_or(0));
    let mut ad = vec![ad2; n];
    ad[0] = ad1;
    let fill = |p: Option<i32>| p.map_or(PL_FILL, i64::from);
    let mut pl = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in 0..=j {
            pl.push(if i == 0 && j == 0 {
                fill(b.pl_1)
            } else if i == 0 {
                fill(b.pl_2)
            } else {
                fill(b.pl_3)
            });
        }
    }
    Call {
        gt: if no_call { "./." } else { "0/0" }.to_string(),
        ad: Some(ad),
        dp: b.dp.map(i64::from),
        f1r2: None,
        f2r1: None,
        pl: Some(pl),
    }
}

/// `generate_sample_call`: its own row → `create_variant_call`; else a
/// star sample → `create_star_call`; else a ref block →
/// `create_refblock_call`; else `GT=./.` alone. At a capped or trimmed
/// record (K3, K5) every call is over the kept alleles. A ref-block fill's
/// PL is never normalized by itself (since the sample subset; the ALT
/// cap first normalized an all-positive fill at a capped record: the one
/// PL rule of the projections shifts a PL only when its source held a 0, which a variant or star call's always
/// does and a fill's need not); at a restricted record it takes the one PL
/// rule against its unrestricted fill (`restricted_fill_pl`).
pub fn generate_sample_call(
    own: Option<&SiteRow>,
    star: Option<&Star<'_>>,
    block: Option<&Block>,
    site: &SiteCall<'_>,
    star_nonref: Option<&HashSet<String>>,
) -> Call {
    if let Some(row) = own {
        create_variant_call(row, site)
    } else if let Some(s) = star {
        let empty = HashSet::new();
        create_star_call(
            s,
            &site.vcf_alleles,
            &site.overlapping,
            star_nonref.unwrap_or(&empty),
        )
    } else if let Some(b) = block {
        let mut c = create_refblock_call(b, site.vcf_alleles.len());
        if let Some(u) = &site.uncapped
            && let Some(pl) = c.pl.as_mut()
        {
            restricted_fill_pl(pl, b, u.n_alleles);
        }
        c
    } else {
        Call::no_call()
    }
}

/// The one PL rule on a ref-block fill at a restricted record (the trim
/// or the cap): `pl` is the fill over the kept alleles, which are the
/// unrestricted fill's kept genotypes' entries; they are shifted by their
/// minimum when the unrestricted fill (over `n_unrestricted` alleles) held
/// a 0 and every kept entry is > 0. While an ALT is kept the kept entries
/// hold every value of the fill (`pl_1`, `pl_2`, `pl_3`), so this changes
/// nothing; it acts only at a record the trim left without ALTs, whose one
/// genotype (0/0, `pl_1`) loses a 0 the fill held at `pl_2` or `pl_3`.
fn restricted_fill_pl(pl: &mut [i64], b: &Block, n_unrestricted: usize) {
    let fill = |p: Option<i32>| p.map_or(PL_FILL, i64::from);
    let source_zero =
        fill(b.pl_1) == 0 || (n_unrestricted >= 2 && (fill(b.pl_2) == 0 || fill(b.pl_3) == 0));
    if source_zero {
        normalize_pl(pl);
    }
}

#[cfg(test)]
mod tests {
    //! The star-call cases of `tests/unit/test_star_call.py` (E3's worked
    //! example: chr21:9,002,912 `T>C,*`, HG00100's record at the deletion
    //! site 9,002,901 with its own list `[C, -CTTTTTT, -CT, *]`), and
    //! `map_pl_values` by allele.
    use std::collections::HashMap;

    use super::*;
    use crate::call::{Info, Overlapping, SiteCall};

    const DEL_SITE: i32 = 9_002_901;
    const OWN: [&str; 4] = ["C", "-CTTTTTT", "-CT", "*"];
    const AD: [i32; 4] = [7, 14, 7, 0];
    const PL: [i32; 10] = [65, 15, 22, 341, 0, 47, 441, 441, 441, 441];
    const CATALOG_NONREF: [&str; 5] = ["-CTTTTT", "-CTTTTTT", "-CT", "*", "-CTTTTTTTTTTTT"];

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn opt(v: &[i32]) -> Option<Vec<Option<i32>>> {
        Some(v.iter().map(|&x| Some(x)).collect())
    }

    fn star_call(own: &[&str], ad: &[i32], pl: &[i32], count: i32, nonref: &[&str]) -> Call {
        let row = SiteRow {
            sample_id: 1,
            alleles: Some(s(own)),
            gt_1: Some(1),
            gt_2: Some(2),
            dp: Some(42),
            ad: opt(ad),
            f1r2: None,
            f2r1: None,
            pl: opt(pl),
        };
        let star = Star {
            count,
            site: DEL_SITE,
            row: &row,
        };
        let overlapping = Overlapping {
            alleles: vec![(DEL_SITE, "-CTTTTTTTTTTTT".into())],
            sites: vec![(DEL_SITE, DEL_SITE)],
        };
        let nonref: HashSet<String> = nonref.iter().map(|x| x.to_string()).collect();
        create_star_call(&star, &s(&["T", "C", "*"]), &overlapping, &nonref)
    }

    fn gidx(a: usize, b: usize) -> usize {
        gt_index(a, b)
    }

    #[test]
    fn worked_example_star_call_pl() {
        let c = star_call(&OWN, &AD, &PL, 2, &CATALOG_NONREF);
        assert_eq!(c.gt, "2/2");
        assert_eq!(c.pl, Some(vec![50, 426, 426, 0, 426, 7]));
        assert_eq!(c.ad, Some(vec![7, 0, 0])); // ref only: no allele overlaps
        assert_eq!(c.dp, Some(42));
        assert_eq!((c.f1r2, c.f2r1), (None, None));
    }

    #[test]
    fn star_and_nonref_found_by_allele_not_position() {
        let order = [0usize, 3, 1, 2]; // [C, *, -CTTTTTT, -CT]
        let own: Vec<&str> = order.iter().map(|&i| OWN[i]).collect();
        let ad: Vec<i32> = order.iter().map(|&i| AD[i]).collect();
        let mut new_of = [0usize; 4];
        for (new, &old) in order.iter().enumerate() {
            new_of[old] = new;
        }
        let mut pl = vec![0; PL.len()];
        for j in 0..4 {
            for i in 0..=j {
                pl[gidx(new_of[i], new_of[j])] = PL[gidx(i, j)];
            }
        }
        let c = star_call(&own, &ad, &pl, 2, &CATALOG_NONREF);
        assert_eq!(c.pl, Some(vec![50, 426, 426, 0, 426, 7]));
    }

    #[test]
    fn het_deletion_carrier_pl_zero_at_the_called_star_genotype() {
        let c = star_call(
            &["C", "-CTTTTTT", "*"],
            &[7, 14, 0],
            &[65, 0, 400, 341, 410, 441],
            1,
            &CATALOG_NONREF,
        );
        assert_eq!(c.gt, "0/2");
        assert_eq!(c.pl.as_ref().unwrap()[gidx(0, 2)], 0);
        assert_eq!(c.pl, Some(vec![65, 341, 441, 0, 410, 400]));
    }

    #[test]
    fn record_without_nonref_uses_pl_fill() {
        let c = star_call(
            &["C", "-CTTTTTT", "-CT"],
            &[7, 14, 7],
            &[65, 15, 22, 341, 0, 47],
            2,
            &CATALOG_NONREF,
        );
        let pls = c.pl.unwrap();
        assert_eq!(PL_FILL, 990);
        assert_eq!(
            [pls[gidx(0, 1)], pls[gidx(1, 1)], pls[gidx(1, 2)]],
            [990 - 15, 990 - 15, 990 - 15]
        );
        assert!(pls[gidx(0, 2)] == 0 && pls[gidx(0, 0)] == 50);
    }

    #[test]
    fn star_call_overlapping_allele_to_the_star_slot() {
        // The row lists the overlapping deletion: its AD goes to `*` as the
        // max, the ref's to slot 0, other non-ref alleles are dropped; the
        // overlapping allele with the most AD is the star allele.
        let own = ["C", "-CTTTTTTTTTTTT", "-CT", "*"];
        let c = star_call(&own, &[3, 9, 4, 1], &PL, 1, &CATALOG_NONREF);
        assert_eq!(c.ad, Some(vec![3, 0, 9]));
        // ref 0, star 1 (the overlapping allele), every other VCF allele
        // the record's `*` (3): [65, 441, 441, 15, 441, 22] less 15.
        assert_eq!(c.pl, Some(vec![50, 426, 426, 0, 426, 7]));
    }

    /// A site with the given VCF alleles, catalog and mapping.
    fn site<'a>(catalog: &[&'a str], t2f: &[(&str, &str)], vcf: &[&str]) -> SiteCall<'a> {
        SiteCall {
            max_ref: vcf[0].to_string(),
            final_alts: s(&vcf[1..]),
            vcf_alleles: s(vcf),
            catalog: catalog.iter().copied().collect(),
            trimmed_to_full: t2f
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            full_to_idx: vcf
                .iter()
                .enumerate()
                .map(|(i, a)| (a.to_string(), i))
                .collect::<HashMap<_, _>>(),
            overlapping: Overlapping::default(),
            stars: Vec::new(),
            info: Info::default(),
            catalog_ac: Vec::new(),
            uncapped: None,
        }
    }

    const T2F: [(&str, &str); 4] = [("T", "T"), ("xC", "C"), ("xG", "G"), ("*", "<NON_REF>")];

    fn map(own: &[&str], pl: &[i64], vcf: &[&str]) -> Option<Vec<i64>> {
        let sc = site(&["T", "xC", "xG", "*"], &T2F, vcf);
        map_pl_values(Some(pl), Some(&s(own)), &sc)
    }

    #[test]
    fn map_pl_by_allele_with_star_standing_in() {
        let want = vec![30, 0, 40, 50, 60, 70];
        assert_eq!(
            map(
                &["T", "xC", "*"],
                &[30, 0, 40, 50, 60, 70],
                &["T", "C", "G"]
            ),
            Some(want.clone())
        );
        assert_eq!(
            map(
                &["T", "*", "xC"],
                &[30, 50, 70, 0, 60, 40],
                &["T", "C", "G"]
            ),
            Some(want)
        );
    }

    #[test]
    fn map_pl_without_nonref_fills() {
        assert_eq!(
            map(&["T", "xC"], &[30, 0, 40], &["T", "C", "G"]),
            Some(vec![30, 0, 40, 990, 990, 990])
        );
        // Every PL > 0: the minimum subtracted.
        assert_eq!(
            map(&["T", "xC"], &[30, 5, 40], &["T", "C"]),
            Some(vec![25, 0, 35])
        );
        // Nothing priced: no PL; no PL or no alleles: none.
        assert_eq!(map(&["xA", "xT"], &[1, 2, 3], &["T", "C"]), None);
        assert_eq!(map(&["T", "xC"], &[], &["T", "C"]), None);
        let sc = site(&["T"], &T2F, &["T"]);
        assert_eq!(map_pl_values(Some(&[1]), None, &sc), None);
        // A `*` that is not in the catalog at the site does not stand in.
        let sc = site(&["T", "xC"], &T2F, &["T", "C", "G"]);
        assert_eq!(
            map_pl_values(
                Some(&[30, 0, 40, 50, 60, 70]),
                Some(&s(&["T", "xC", "*"])),
                &sc
            ),
            Some(vec![30, 0, 40, 990, 990, 990])
        );
    }

    #[test]
    fn variant_call_gt_ad_and_null_fills() {
        let sc = site(&["T", "xC", "xG", "*"], &T2F, &["T", "C", "G"]);
        let row = SiteRow {
            sample_id: 1,
            alleles: Some(s(&["T", "xG", "*"])),
            gt_1: Some(0),
            gt_2: Some(1),
            dp: None,
            ad: Some(vec![Some(3), None, Some(1)]),
            f1r2: Some(vec![]),
            f2r1: Some(vec![Some(1), Some(2), Some(0)]),
            pl: Some(vec![Some(10), None, Some(0), Some(20), Some(30), Some(40)]),
        };
        let c = create_variant_call(&row, &sc);
        assert_eq!(c.gt, "0/2");
        assert_eq!(c.ad, Some(vec![3, 0, 0])); // the null AD element is 0
        assert_eq!(c.dp, None);
        assert_eq!(c.f1r2, None); // empty: not written
        assert_eq!(c.f2r1, Some(vec![1, 0, 2]));
        // T/T 10, T/C -> T/* 20, C/C -> */* 40, T/G 990 (the null), ...
        assert_eq!(c.pl, Some(vec![10, 20, 40, 990, 30, 0]));
        // A GT allele not among the VCF alleles (`*`): ./.
        let mut r2 = row.clone();
        r2.gt_2 = Some(2);
        assert_eq!(create_variant_call(&r2, &sc).gt, "./.");
        // An index outside the list, a null GT: ./.
        r2.gt_2 = Some(7);
        assert_eq!(create_variant_call(&r2, &sc).gt, "./.");
        r2.gt_2 = None;
        assert_eq!(create_variant_call(&r2, &sc).gt, "./.");
        // No AD: all zeros, still written.
        let mut r3 = row.clone();
        r3.ad = None;
        assert_eq!(create_variant_call(&r3, &sc).ad, Some(vec![0, 0, 0]));
    }

    #[test]
    fn refblock_call_and_the_no_call_block() {
        let b = Block {
            sample_id: 1,
            start: 1,
            end: 10,
            ad_1: Some(5),
            ad_2: None,
            dp: Some(6),
            pl_1: Some(0),
            pl_2: Some(15),
            pl_3: None,
            sample_counts: Some(1),
        };
        let c = create_refblock_call(&b, 3);
        assert_eq!(c.gt, "0/0");
        assert_eq!(c.ad, Some(vec![5, 0, 0]));
        assert_eq!(c.dp, Some(6));
        assert_eq!(c.pl, Some(vec![0, 15, 990, 15, 990, 990]));
        assert_eq!((c.f1r2, c.f2r1), (None, None));
        let no_call = Block {
            sample_counts: Some(0),
            ..b
        };
        let c = create_refblock_call(&no_call, 2);
        assert_eq!(c.gt, "./.");
        assert_eq!(c.pl, Some(vec![0, 15, 990]));
        let null_counts = Block {
            sample_counts: None,
            dp: None,
            ..b
        };
        let c = create_refblock_call(&null_counts, 1);
        assert_eq!((c.gt.as_str(), c.dp), ("0/0", None));
    }

    // ── The null FORMAT values and the ref-block fill of
    // tests/unit/test_joint_call_info.py, ported when the Python engine
    // was removed: the same inputs, through the
    // calls and the writer (`format_record`), whose FORMAT and sample
    // columns are what the Python tests asserted of vcflib's line. A
    // pandas NaN of the Python inputs is a null (None) here.

    /// The Python tests' `_call_data`: VCF alleles [A, G], catalog
    /// {A, xG, *}.
    fn py_site() -> SiteCall<'static> {
        site(
            &["A", "xG", "*"],
            &[("A", "A"), ("xG", "G"), ("*", "<NON_REF>")],
            &["A", "G"],
        )
    }

    /// The FORMAT column and the sample columns of the record line
    /// (chr21:100 A>G, as the Python tests' `_format`).
    fn py_format(calls: &[Call]) -> Vec<String> {
        let alts = s(&["G"]);
        let rec = crate::writer::Record {
            chrom: "chr21",
            pos: 100,
            reference: "A",
            alts: &alts,
            qual: None,
            info: ".",
        };
        let mut out = String::new();
        crate::writer::format_record(&mut out, &rec, calls);
        assert!(!out.to_lowercase().contains("nan"), "{out}");
        out.trim_end_matches('\n')
            .split('\t')
            .skip(8)
            .map(str::to_string)
            .collect()
    }

    fn py_block(v: [Option<i32>; 7]) -> Block {
        Block {
            sample_id: 1,
            start: 1,
            end: 1000,
            ad_1: v[0],
            ad_2: v[1],
            dp: v[2],
            pl_1: v[3],
            pl_2: v[4],
            pl_3: v[5],
            sample_counts: v[6],
        }
    }

    /// Replaces `test_joint_call_info.py::test_variant_call_nulls`: the
    /// record's alleles [A, G, *], GT 0/1, AD with a null G element, DP
    /// null, PL over 3 alleles with a null at G/G (k = 2).
    #[test]
    fn python_variant_call_nulls() {
        let sc = py_site();
        let row = SiteRow {
            sample_id: 1,
            alleles: Some(s(&["A", "xG", "*"])),
            gt_1: Some(0),
            gt_2: Some(1),
            dp: None,
            ad: Some(vec![Some(10), None, Some(0)]),
            f1r2: None,
            f2r1: None,
            pl: Some(vec![Some(0), Some(30), None, Some(40), Some(50), Some(60)]),
        };
        let c = generate_sample_call(Some(&row), None, None, &sc, None);
        assert_eq!(c.gt, "0/1");
        assert_eq!(c.ad, Some(vec![10, 0]));
        assert_eq!(c.dp, None);
        assert_eq!(c.pl, Some(vec![0, 30, PL_FILL]));
        assert_eq!(py_format(&[c]), ["GT:AD:PL", "0/1:10,0:0,30,990"]);
    }

    /// Replaces `test_joint_call_info.py::test_variant_call_null_dp_prints_dot_beside_another_sample`:
    /// a null DP prints `.` when the record writes DP (here because another
    /// sample has it; the Python test set the key to None by hand).
    #[test]
    fn python_variant_call_null_dp_prints_dot_beside_another_sample() {
        let sc = py_site();
        let row = SiteRow {
            sample_id: 1,
            alleles: Some(s(&["A", "xG", "*"])),
            gt_1: Some(0),
            gt_2: Some(1),
            dp: None,
            ad: opt(&[10, 5, 0]),
            f1r2: None,
            f2r1: None,
            pl: opt(&[0, 30, 99, 0, 30, 99]),
        };
        let c = generate_sample_call(Some(&row), None, None, &sc, None);
        assert_eq!(c.dp, None);
        let other = Call {
            dp: Some(5),
            ..c.clone()
        };
        let cols = py_format(&[c, other]);
        assert_eq!(cols[0], "GT:AD:DP:PL");
        assert_eq!(cols[1].split(':').nth(2), Some("."));
    }

    /// Replaces `test_joint_call_info.py::test_refblock_call_nulls`.
    #[test]
    fn python_refblock_call_nulls() {
        let b = py_block([None, Some(5), None, Some(0), None, Some(99), Some(1)]);
        let c = create_refblock_call(&b, 2);
        assert_eq!(c.ad, Some(vec![0, 5]));
        assert_eq!(c.dp, None);
        assert_eq!(c.pl, Some(vec![0, PL_FILL, 99]));
        let c = generate_sample_call(None, None, Some(&b), &py_site(), None);
        assert_eq!(py_format(&[c])[1], "0/0:0,5:0,990,99");
    }

    /// Replaces `test_joint_call_info.py::test_non_null_values_unchanged`.
    #[test]
    fn python_refblock_non_null_values_unchanged() {
        let b = py_block([
            Some(7),
            Some(5),
            Some(12),
            Some(0),
            Some(30),
            Some(99),
            Some(1),
        ]);
        assert_eq!(
            create_refblock_call(&b, 2),
            Call {
                gt: "0/0".to_string(),
                ad: Some(vec![7, 5]),
                dp: Some(12),
                f1r2: None,
                f2r1: None,
                pl: Some(vec![0, 30, 99]),
            }
        );
    }

    /// Replaces `test_joint_call_info.py::test_refblock_no_call_block_writes_dot`:
    /// `sample_counts` 0 (a gVCF ./. <NON_REF> block): GT ./., AD / DP / PL
    /// as the block carries them.
    #[test]
    fn python_refblock_no_call_block_writes_dot() {
        let b = py_block([
            Some(3),
            Some(0),
            Some(3),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
        ]);
        assert_eq!(
            create_refblock_call(&b, 3),
            Call {
                gt: "./.".to_string(),
                ad: Some(vec![3, 0, 0]),
                dp: Some(3),
                f1r2: None,
                f2r1: None,
                pl: Some(vec![0, 0, 0, 0, 0, 0]),
            }
        );
        let c = generate_sample_call(None, None, Some(&b), &py_site(), None);
        assert_eq!(py_format(&[c]), ["GT:AD:DP:PL", "./.:3,0:3:0,0,0"]);
    }

    /// Replaces `test_joint_call_info.py::test_refblock_hom_ref_block_and_null_counts_write_0_0`
    /// (its `sample_counts` 1, NaN and None: NaN and None are both a null
    /// here).
    #[test]
    fn python_refblock_hom_ref_block_and_null_counts_write_0_0() {
        for sc in [Some(1), None] {
            let b = py_block([Some(7), Some(5), Some(12), Some(0), Some(30), Some(99), sc]);
            let c = create_refblock_call(&b, 2);
            assert_eq!(c.gt, "0/0", "{sc:?}");
            assert_eq!(c.ad, Some(vec![7, 5]));
            assert_eq!(c.pl, Some(vec![0, 30, 99]));
        }
    }

    // ── The ALT cap (`--max-alt-alleles`).

    /// A site over [T, C, G, *] whose catalog is {T, xC, xG, *}.
    fn cap_site<'a>() -> SiteCall<'a> {
        site(&["T", "xC", "xG", "*"], &T2F, &["T", "C", "G", "*"])
    }

    fn gt_row(alleles: &[&str], gt: (Option<i32>, Option<i32>), pl: &[i32]) -> SiteRow {
        SiteRow {
            sample_id: 1,
            alleles: Some(s(alleles)),
            gt_1: gt.0,
            gt_2: gt.1,
            dp: Some(5),
            ad: opt(&vec![1; alleles.len()]),
            f1r2: None,
            f2r1: None,
            pl: opt(pl),
        }
    }

    /// The ranking pass gives the AC `genotype_allele_counts`
    /// gives over the uncapped calls: own rows resolved and unresolved, a
    /// het and a hom star sample, a 0/0 block, a ./. block, no evidence.
    #[test]
    fn ranking_pass_equals_the_uncapped_calls_ac() {
        let del = gt_row(&["A", "-ACG", "*"], (Some(1), Some(1)), &[9, 9, 0]);
        let rows = [
            gt_row(&["T", "xC", "*"], (Some(0), Some(1)), &[]),
            gt_row(&["T", "xG", "xC"], (Some(1), Some(2)), &[]),
            gt_row(&["T", "xG", "*"], (Some(2), Some(1)), &[]), // T/* : ./.
            gt_row(&["T", "xC"], (Some(1), None), &[]),         // null: ./.
        ];
        let mut sc = cap_site();
        sc.stars = vec![
            None,
            None,
            None,
            None,
            Some(Star {
                count: 1,
                site: 10,
                row: &del,
            }),
            Some(Star {
                count: 2,
                site: 10,
                row: &del,
            }),
            None,
            None,
            None,
        ];
        let own: Vec<Option<&SiteRow>> = rows
            .iter()
            .map(Some)
            .chain([None, None, None, None, None])
            .collect();
        let block = |sc: i32| Block {
            sample_counts: Some(sc),
            ..Default::default()
        };
        let blocks = [
            None,
            None,
            None,
            None,
            None,
            None,
            Some(block(1)),
            Some(block(0)),
            None,
        ];
        let nonref: HashSet<String> = ["-ACG".to_string(), "*".to_string()].into();
        let calls: Vec<Call> = (0..9)
            .map(|c| {
                generate_sample_call(
                    own[c],
                    sc.stars[c].as_ref(),
                    blocks[c].as_ref(),
                    &sc,
                    Some(&nonref),
                )
            })
            .collect();
        let gts: Vec<&str> = calls.iter().map(|c| c.gt.as_str()).collect();
        assert_eq!(
            gts,
            [
                "0/1", "2/1", "./.", "./.", "0/3", "3/3", "0/0", "./.", "./."
            ]
        );
        let (ac, _, _) = crate::call::genotype_allele_counts(&gts, 3);
        let pairs = crate::call::uncapped_gt_pairs(&sc, &own);
        assert_eq!(crate::call::ac_of_pairs(&pairs, 3), ac);
        assert_eq!(ac, vec![2, 1, 3]);
        // The kept set: ties to the earlier ALT.
        assert_eq!(crate::call::kept_alts(&ac, 1), [2]);
        assert_eq!(crate::call::kept_alts(&ac, 2), [0, 2]);
        assert_eq!(crate::call::kept_alts(&[1, 1, 1], 2), [0, 1]);
        assert_eq!(crate::call::kept_alts(&[0, 0, 5], 2), [0, 2]);
    }

    /// A capped variant call writes a PL exactly when the
    /// uncapped call does, decided from the row's own alleles; when no kept
    /// genotype is priced but a dropped one is, `PL_FILL`s, normalized.
    #[test]
    fn capped_pl_decided_as_the_uncapped_call() {
        // The row lists xG (VCF G) and an allele not in the catalog (xA):
        // only G/G is priced, from the row's (1, 1) = 9.
        let row = gt_row(&["xA", "xG"], (Some(1), Some(1)), &[5, 7, 9]);
        let uncapped = cap_site();
        let c = create_variant_call(&row, &uncapped);
        assert_eq!(c.gt, "2/2");
        // Uncapped: every genotype PL_FILL but G/G's 9: 981 subtracted.
        let pl = c.pl.unwrap();
        assert_eq!(pl.len(), 10);
        assert_eq!(pl[gidx(2, 2)], 0);
        assert!(
            pl.iter()
                .enumerate()
                .all(|(i, &p)| i == gidx(2, 2) || p == 981)
        );
        // Capped to [T, C] (G and * dropped): no kept genotype priced, a
        // dropped one is: PL_FILL x 3, normalized to 0s; the GT ./. .
        let capped = capped_site(&[0]);
        let c = create_variant_call(&row, &capped);
        assert_eq!(c.gt, "./.");
        assert_eq!(c.pl, Some(vec![0, 0, 0]));
        assert_eq!(c.ad, Some(vec![0, 0]));
        // Nothing priced uncapped (no allele of the row maps): no PL either
        // way.
        let none = gt_row(&["xA", "xT"], (Some(0), Some(1)), &[5, 7, 9]);
        assert_eq!(create_variant_call(&none, &uncapped).pl, None);
        assert_eq!(create_variant_call(&none, &capped).pl, None);
        // A row whose `*` stands in for the unmapped alleles: priced either
        // way; capped, the kept genotypes' entries normalized.
        let star = gt_row(
            &["T", "xG", "*"],
            (Some(0), Some(1)),
            &[30, 10, 20, 40, 50, 60],
        );
        let c = create_variant_call(&star, &capped);
        // [T, C]: T/T 30, T/C -> T/* 40, C/C -> */* 60; all > 0: less 30.
        assert_eq!(c.pl, Some(vec![0, 10, 30]));
        assert_eq!(c.gt, "./.");
        // The PL length short of every genotype the row's alleles reach:
        // uncapped, T/T (index 0) is the only one in [7]; capped [T, C]:
        // T/T 7, the others PL_FILL: 0, 983, 983.
        let short = gt_row(&["T", "xG"], (Some(0), Some(0)), &[7]);
        assert_eq!(
            create_variant_call(&short, &capped).pl,
            Some(vec![0, 983, 983])
        );
        assert_eq!(create_variant_call(&short, &capped).gt, "0/0");
    }

    /// `cap_site` restricted to REF and the kept ALTs (0-based ALT indices
    /// of [C, G, *]), as `restrict_site` restricts it.
    fn capped_site<'a>(kept: &[usize]) -> SiteCall<'a> {
        let full = ["C", "G", "*"];
        let mut vcf = vec!["T"];
        vcf.extend(kept.iter().map(|&k| full[k]));
        let mut sc = site(&["T", "xC", "xG", "*"], &T2F, &vcf);
        sc.uncapped = Some(crate::call::Uncapped {
            n_alleles: 4,
            full_to_idx: ["T", "C", "G", "*"]
                .iter()
                .enumerate()
                .map(|(i, a)| (a.to_string(), i))
                .collect(),
        });
        sc
    }

    /// A star call whose `*` the cap dropped is `./.`, AD keeps slot 0, PL
    /// prices the kept ALTs with the record's `*`; with `*` kept it is the
    /// uncapped call restricted. A ref-block fill is never normalized (the
    /// one PL rule; the cap's first version normalized it).
    #[test]
    fn capped_star_call_and_ref_block_fill() {
        // The worked example's record at the deletion site (its
        // overlapping allele -CTTTTTTTTTTTT not listed), count 2.
        let nonref: HashSet<String> = CATALOG_NONREF.iter().map(|x| x.to_string()).collect();
        let row = SiteRow {
            sample_id: 1,
            alleles: Some(s(&OWN)),
            gt_1: Some(1),
            gt_2: Some(2),
            dp: Some(42),
            ad: opt(&AD),
            f1r2: opt(&[1, 2, 3, 4]),
            f2r1: None,
            pl: opt(&PL),
        };
        let star = Star {
            count: 2,
            site: DEL_SITE,
            row: &row,
        };
        let overlapping = Overlapping {
            alleles: vec![(DEL_SITE, "-CTTTTTTTTTTTT".into())],
            sites: vec![(DEL_SITE, DEL_SITE)],
        };
        let full = create_star_call(&star, &s(&["T", "C", "*"]), &overlapping, &nonref);
        assert_eq!(full.pl, Some(vec![50, 426, 426, 0, 426, 7]));
        // `*` dropped: [T, C].
        let c = create_star_call(&star, &s(&["T", "C"]), &overlapping, &nonref);
        assert_eq!(c.gt, "./.");
        assert_eq!(c.ad, Some(vec![7, 0]));
        assert_eq!(c.f1r2, Some(vec![1, 0]));
        assert_eq!(c.dp, Some(42));
        // The uncapped PL's [T/T, T/C, C/C] entries (50, 426, 426: the raw
        // 65, 441, 441 less 15) normalized: the raw ones less 65.
        assert_eq!(c.pl, Some(vec![0, 376, 376]));
        // `*` kept, C dropped: [T, *]: the uncapped PL's T/T, T/*, */*.
        let c = create_star_call(&star, &s(&["T", "*"]), &overlapping, &nonref);
        assert_eq!(c.gt, "1/1");
        assert_eq!(c.pl, Some(vec![50, 0, 7]));
        // The fill: never normalized, at an uncapped or a capped record.
        let b = Block {
            ad_1: Some(7),
            ad_2: Some(1),
            dp: Some(8),
            pl_1: Some(5),
            pl_2: Some(15),
            pl_3: Some(30),
            sample_counts: Some(1),
            ..Default::default()
        };
        let un = generate_sample_call(None, None, Some(&b), &cap_site(), None);
        assert_eq!(un.pl.as_ref().unwrap()[..3], [5, 15, 30]);
        let c = generate_sample_call(None, None, Some(&b), &capped_site(&[0]), None);
        assert_eq!(
            (c.gt.as_str(), c.ad, c.pl),
            ("0/0", Some(vec![7, 1]), Some(vec![5, 15, 30]))
        );
        // A fill holding a 0 is unchanged; a ./. block keeps its GT.
        let z = Block { pl_1: Some(0), ..b };
        let c = generate_sample_call(None, None, Some(&z), &capped_site(&[0]), None);
        assert_eq!(c.pl, Some(vec![0, 15, 30]));
        let nc = Block {
            sample_counts: Some(0),
            ..b
        };
        let c = generate_sample_call(None, None, Some(&nc), &capped_site(&[1]), None);
        assert_eq!((c.gt.as_str(), c.pl), ("./.", Some(vec![5, 15, 30])));
    }

    /// K5: a fill at a record the trim left without ALTs (one genotype,
    /// 0/0) takes the one PL rule against its unrestricted fill: shifted to
    /// 0 when the fill held a 0 at 0/i or i/j and `pl_1` is > 0; as it is
    /// when the fill held no 0, or its 0 at 0/0. While an ALT is kept the
    /// rule changes nothing (the kept entries hold every value of the
    /// fill), and an unrestricted record's fill is never shifted.
    #[test]
    fn fill_at_a_record_trimmed_to_ref() {
        let b = |p: [i32; 3]| Block {
            ad_1: Some(7),
            ad_2: Some(1),
            dp: Some(8),
            pl_1: Some(p[0]),
            pl_2: Some(p[1]),
            pl_3: Some(p[2]),
            sample_counts: Some(1),
            ..Default::default()
        };
        let pl = |blk: &Block, sc: &SiteCall<'_>| {
            generate_sample_call(None, None, Some(blk), sc, None)
                .pl
                .unwrap()
        };
        let to_ref = capped_site(&[]);
        assert_eq!(to_ref.vcf_alleles, ["T"]);
        assert_eq!(pl(&b([5, 0, 30]), &to_ref), [0]);
        assert_eq!(pl(&b([5, 15, 0]), &to_ref), [0]);
        assert_eq!(pl(&b([5, 15, 30]), &to_ref), [5]);
        assert_eq!(pl(&b([0, 15, 30]), &to_ref), [0]);
        // A null pl_2 is PL_FILL (no 0).
        let mut nulls = b([5, 0, 30]);
        nulls.pl_2 = None;
        nulls.pl_3 = None;
        assert_eq!(pl(&nulls, &to_ref), [5]);
        // An ALT kept: the kept entries as they are.
        assert_eq!(pl(&b([5, 0, 30]), &capped_site(&[1])), [5, 0, 30]);
        // Unrestricted: never shifted.
        assert_eq!(pl(&b([5, 15, 30]), &cap_site())[..3], [5, 15, 30]);
        // The AD over REF alone.
        let c = generate_sample_call(None, None, Some(&b([5, 0, 30])), &to_ref, None);
        assert_eq!((c.gt.as_str(), c.ad, c.dp), ("0/0", Some(vec![7]), Some(8)));
    }

    #[test]
    fn sample_call_precedence() {
        let sc = site(&["T", "xC"], &T2F, &["T", "C"]);
        let b = Block {
            sample_counts: Some(0),
            ..Default::default()
        };
        let c = generate_sample_call(None, None, Some(&b), &sc, None);
        assert_eq!(c.gt, "./.");
        assert!(c.ad.is_some());
        let c = generate_sample_call(None, None, None, &sc, None);
        assert_eq!(c, Call::no_call());
    }
}
