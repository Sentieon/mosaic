// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Per-sample QC counters (`sample_qc`): `qc_variant_record` and
//! `qc_ref_block` (originally a port of the Python worker's functions),
//! in `SAMPLE_QC_COLUMNS` order (`mosaic/init_db.py`).

use crate::matrix::{F_DP, F_MIN_DP, GQ_UNKNOWN, RB_HOM_REF, RB_NO_CALL, dict0, samples0};
use crate::pyops::{PyErr, pv, to_int, upper};
use crate::record::Record;
use crate::ref_blocks::I_DP;

pub const N_QC: usize = 19;
pub const QC_N_VAR: usize = 0;
pub const QC_NO_CALL: usize = 1;
pub const QC_HOM_REF: usize = 2;
pub const QC_HET: usize = 3;
pub const QC_HOM_ALT: usize = 4;
pub const QC_HAPLOID: usize = 5;
pub const QC_SNV: usize = 6;
pub const QC_MNP: usize = 7;
pub const QC_INS: usize = 8;
pub const QC_DEL: usize = 9;
pub const QC_TI: usize = 10;
pub const QC_TV: usize = 11;
pub const QC_FILTERED: usize = 12;
pub const QC_DP_SUM: usize = 13;
pub const QC_N_DP: usize = 14;
pub const QC_RB_BASES: usize = 15;
pub const QC_RB_DP_SUM: usize = 16;
pub const QC_RB_GQ20: usize = 17;
pub const QC_RB_NO_CALL: usize = 18;

pub type Qc = [i64; N_QC];

fn is_acgt(s: &str) -> bool {
    matches!(s, "A" | "C" | "G" | "T")
}

fn is_transition(r: &str, a: &str) -> bool {
    matches!((r, a), ("A", "G") | ("G", "A") | ("C", "T") | ("T", "C"))
}

/// `qc_variant_record(q, v, fmts, gt_fields)` (called for a record with a
/// sample).
pub fn qc_variant_record(
    q: &mut Qc,
    v: &Record,
    fmts: &[String],
    gt_fields: Option<&[Option<i64>]>,
) -> Result<(), PyErr> {
    q[QC_N_VAR] += 1;
    if !v.filter.is_empty() && !(v.filter.len() == 1 && v.filter[0] == "PASS") {
        q[QC_FILTERED] += 1;
    }
    let g = match gt_fields {
        Some(g) if !g.iter().all(Option::is_none) => g,
        _ => {
            q[QC_NO_CALL] += 1;
            return Ok(());
        }
    };
    let sample = samples0(v).ok_or_else(|| PyErr::Other("IndexError: no sample".into()))?;
    if let Some(dp) = pv(&sample[F_DP]) {
        q[QC_DP_SUM] += to_int(Some(dp)).map_err(|e| PyErr::Other(e.to_string()))?;
        q[QC_N_DP] += 1;
    }
    let n_fmt = fmts.len() as i64;
    let mut carried: Vec<usize> = Vec::with_capacity(2);
    for x in g.iter().flatten() {
        let x = *x;
        if 0 < x && x < n_fmt {
            let f = fmts[x as usize].as_str();
            if f != "*" && f != "x*" && !carried.contains(&(x as usize)) {
                carried.push(x as usize);
            }
        }
    }
    if g.len() == 1 {
        q[QC_HAPLOID] += 1;
    } else if let (Some(a), Some(b)) = (g[0], g[1]) {
        if a == b {
            q[if a == 0 { QC_HOM_REF } else { QC_HOM_ALT }] += 1;
        } else if !carried.is_empty() {
            q[QC_HET] += 1;
        }
    }
    for x in carried {
        let f = &fmts[x];
        let mut cs = f.chars();
        let c = cs
            .next()
            .ok_or_else(|| PyErr::Other("IndexError: an empty allele".into()))?;
        match c {
            'x' => {
                if f.chars().count() == 2 {
                    q[QC_SNV] += 1;
                    let r = upper(
                        v.ref_
                            .chars()
                            .next()
                            .ok_or_else(|| PyErr::Other("IndexError: an empty REF".into()))?,
                    );
                    let a = upper(cs.next().expect("two characters"));
                    if is_acgt(&r) && is_acgt(&a) {
                        q[if is_transition(&r, &a) { QC_TI } else { QC_TV }] += 1;
                    }
                } else {
                    q[QC_MNP] += 1;
                }
            }
            '+' => q[QC_INS] += 1,
            '-' => q[QC_DEL] += 1,
            _ => {}
        }
    }
    Ok(())
}

/// `qc_ref_block(q, v, kind, gq, start, end, shard_start, shard_end)`.
#[allow(clippy::too_many_arguments)]
pub fn qc_ref_block(
    q: &mut Qc,
    v: &Record,
    kind: u8,
    gq: i64,
    start: i64,
    end: i64,
    shard_start_1based: i64,
    shard_end_1based: i64,
) -> Result<(), PyErr> {
    let length = end.min(shard_end_1based) - start.max(shard_start_1based) + 1;
    if length <= 0 {
        return Ok(());
    }
    if kind == RB_HOM_REF {
        q[QC_RB_BASES] += length;
        let mut dp = dict0(v).and_then(|s| pv(&s[F_MIN_DP]));
        if dp.is_none() {
            dp = pv(&v.info[I_DP]);
        }
        if let Some(d) = dp {
            q[QC_RB_DP_SUM] += to_int(Some(d)).map_err(|e| PyErr::Other(e.to_string()))? * length;
        }
        if gq != GQ_UNKNOWN && gq >= 20 {
            q[QC_RB_GQ20] += length;
        }
    } else if kind == RB_NO_CALL {
        q[QC_RB_NO_CALL] += length;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! `test_genotype_builder.py`'s QC cases.
    use super::*;
    use crate::matrix::tests::{block, var};
    use crate::matrix::{F_GT, parse_gt};
    use std::rc::Rc;

    fn qc(v: &Record, fmts: &[&str]) -> Qc {
        let mut q = [0; N_QC];
        let f: Vec<String> = fmts.iter().map(|s| s.to_string()).collect();
        let g = parse_gt(pv(&v.s0.as_ref().unwrap()[F_GT])).unwrap();
        qc_variant_record(&mut q, v, &f, g.as_deref()).unwrap();
        q
    }

    #[test]
    fn snv_ti_tv() {
        let q = qc(&var(5, "A", "G", ".", "GT:DP", "0/1:12"), &["A", "xG"]);
        assert_eq!((q[QC_SNV], q[QC_TI], q[QC_TV], q[QC_HET]), (1, 1, 0, 1));
        assert_eq!((q[QC_DP_SUM], q[QC_N_DP]), (12, 1));
        let q = qc(&var(5, "a", "T", ".", "GT", "1/1"), &["a", "xT"]);
        assert_eq!((q[QC_TV], q[QC_HOM_ALT]), (1, 1));
    }

    #[test]
    fn indels_mnp_non_ref_and_zygosity() {
        let q = qc(
            &var(5, "AT", "A,ATT", ".", "GT", "1/2"),
            &["A", "-AT", "+AT"],
        );
        assert_eq!((q[QC_DEL], q[QC_INS], q[QC_HET]), (1, 1, 1));
        let q = qc(&var(5, "AC", "GT", ".", "GT", "1/1"), &["A", "xGT"]);
        assert_eq!(q[QC_MNP], 1);
        // <NON_REF> is not carried: 0/1 with fmts[1] = '*' is not het.
        let q = qc(
            &var(5, "A", "<NON_REF>,G", ".", "GT", "0/1"),
            &["A", "*", "xG"],
        );
        assert_eq!(q[QC_HET], 0);
        let q = qc(&var(5, "A", "G", ".", "GT", "./."), &["A", "xG"]);
        assert_eq!(q[QC_NO_CALL], 1);
        let q = qc(&var(5, "A", "G", ".", "GT", "1"), &["A", "xG"]);
        assert_eq!(q[QC_HAPLOID], 1);
        let q = qc(&var(5, "A", "G", ".", "GT", "./1"), &["A", "xG"]);
        assert_eq!((q[QC_HET], q[QC_HOM_ALT], q[QC_NO_CALL]), (0, 0, 0));
        let q = qc(&var(5, "A", "G", ".", "GT", "0/0"), &["A", "xG"]);
        assert_eq!(q[QC_HOM_REF], 1);
        assert_eq!(q[QC_FILTERED], 0);
    }

    #[test]
    fn ref_block_clipping_dp_and_gq20() {
        let b = block(10, 30, "0/0:5,0:7:25:6:0,20,200");
        let mut q = [0; N_QC];
        qc_ref_block(&mut q, &b, RB_HOM_REF, 25, 10, 30, 20, 100).unwrap();
        assert_eq!(
            (q[QC_RB_BASES], q[QC_RB_DP_SUM], q[QC_RB_GQ20]),
            (11, 66, 11)
        );
        qc_ref_block(&mut q, &b, RB_NO_CALL, 25, 10, 30, 1, 100).unwrap();
        assert_eq!(q[QC_RB_NO_CALL], 21);
        let mut q = [0; N_QC];
        qc_ref_block(&mut q, &b, RB_HOM_REF, GQ_UNKNOWN, 10, 30, 50, 100).unwrap();
        assert_eq!(q, [0; N_QC]);
    }

    // ── K6 ports of tests/unit/test_genotype_builder.py's QC cases ──────
    // Python's `Rec(pos, ...)` takes a 0-based pos; POS = pos + 1 here.

    fn qrec(
        pos0: i64,
        r: &str,
        alts: &[&str],
        gt: &str,
        dp: Option<i64>,
        filt: &str,
    ) -> Rc<Record> {
        let (f, s) = match dp {
            Some(d) => ("GT:DP".to_string(), format!("{gt}:{d}")),
            None => ("GT".to_string(), gt.to_string()),
        };
        crate::matrix::tests::rec(&format!(
            "chr21\t{}\t.\t{r}\t{}\t.\t{filt}\t.\t{f}\t{s}",
            pos0 + 1,
            alts.join(",")
        ))
    }

    fn pass(pos0: i64, r: &str, alts: &[&str], gt: &str, dp: Option<i64>) -> Rc<Record> {
        qrec(pos0, r, alts, gt, dp, "PASS")
    }

    /// `qc_of(*records)`: fmts_of + parse_gt per record, one counter set.
    fn qc_of(recs: &[Rc<Record>]) -> Qc {
        let mut q = [0; N_QC];
        for v in recs {
            let mut f = vec![v.ref_.chars().next().unwrap().to_string()];
            f.extend(
                v.alt
                    .iter()
                    .map(|a| mosaic_core::alleles::format_alt_allele(&v.ref_, a)),
            );
            let g = parse_gt(pv(&v.s0.as_ref().unwrap()[F_GT])).unwrap();
            qc_variant_record(&mut q, v, &f, g.as_deref()).unwrap();
        }
        q
    }

    // Replaces tests/unit/test_genotype_builder.py::test_qc_snv_transition_and_transversion (K6).
    #[test]
    fn qc_snv_transition_and_transversion() {
        let q = qc_of(&[
            pass(10, "A", &["G", "<NON_REF>"], "0/1", Some(20)), // Ti, het
            pass(20, "C", &["A", "<NON_REF>"], "1/1", Some(10)), // Tv, hom-alt
            pass(30, "CT", &["TT", "<NON_REF>"], "0/1", Some(5)), // trimmed C>T
        ]);
        assert_eq!(q[QC_N_VAR], 3);
        assert_eq!((q[QC_HET], q[QC_HOM_ALT]), (2, 1));
        assert_eq!((q[QC_SNV], q[QC_TI], q[QC_TV]), (3, 2, 1));
        assert_eq!((q[QC_DP_SUM], q[QC_N_DP]), (35, 3));
    }

    // Replaces tests/unit/test_genotype_builder.py::test_qc_indels_mnp_and_multiallelic (K6).
    #[test]
    fn qc_indels_mnp_and_multiallelic() {
        let q = qc_of(&[
            pass(10, "A", &["AT", "<NON_REF>"], "0/1", None),
            pass(20, "ATG", &["A", "<NON_REF>"], "1/1", None),
            pass(30, "AC", &["GT", "<NON_REF>"], "0/1", None),
            // 1/2: one SNV + one deletion carried; GA (allele 3) is not.
            pass(40, "GAAA", &["G", "TAAA", "GA", "<NON_REF>"], "1/2", None),
        ]);
        assert_eq!((q[QC_INS], q[QC_DEL], q[QC_MNP], q[QC_SNV]), (1, 2, 1, 1));
        assert_eq!((q[QC_HET], q[QC_HOM_ALT]), (3, 1));
        assert_eq!(q[QC_TV], 1); // G>T
    }

    // Replaces tests/unit/test_genotype_builder.py::test_qc_non_ref_is_not_carried (K6).
    #[test]
    fn qc_non_ref_is_not_carried() {
        let q = qc_of(&[pass(10, "A", &["G", "<NON_REF>"], "0/2", None)]);
        assert_eq!(q[QC_N_VAR], 1);
        assert_eq!((q[QC_HET], q[QC_SNV]), (0, 0));
    }

    // Replaces tests/unit/test_genotype_builder.py::test_qc_no_call_haploid_half_call_filtered_hom_ref (K6).
    #[test]
    fn qc_no_call_haploid_half_call_filtered_hom_ref() {
        let q = qc_of(&[
            pass(10, "A", &["G", "<NON_REF>"], "./.", Some(7)),
            pass(20, "A", &["G", "<NON_REF>"], "1", Some(8)),
            pass(30, "A", &["G", "<NON_REF>"], "./1", Some(9)),
            qrec(40, "A", &["G", "<NON_REF>"], "0/0", None, "LowGQ"),
        ]);
        assert_eq!(q[QC_N_VAR], 4);
        assert_eq!(q[QC_NO_CALL], 1);
        assert_eq!(q[QC_HAPLOID], 1);
        assert_eq!(q[QC_HOM_REF], 1);
        assert_eq!((q[QC_HET], q[QC_HOM_ALT]), (0, 0)); // half call: neither
        assert_eq!(q[QC_FILTERED], 1);
        assert_eq!(q[QC_SNV], 2); // haploid 1 and half call ./1 carry G
        assert_eq!((q[QC_DP_SUM], q[QC_N_DP]), (17, 2)); // no-call excluded
    }

    /// Python's `block(pos, end_1based, gt, gq, min_dp)` (+ INFO/DP).
    fn qblk(
        pos0: i64,
        end_1b: i64,
        gt: &str,
        gq: Option<i64>,
        min_dp: Option<i64>,
        info_dp: Option<i64>,
    ) -> Rc<Record> {
        let mut f = vec!["GT".to_string()];
        let mut s = vec![gt.to_string()];
        if let Some(g) = gq {
            f.push("GQ".into());
            s.push(g.to_string());
        }
        if let Some(m) = min_dp {
            f.push("MIN_DP".into());
            s.push(m.to_string());
        }
        let info = match info_dp {
            Some(d) => format!("END={end_1b};DP={d}"),
            None => format!("END={end_1b}"),
        };
        crate::matrix::tests::rec(&format!(
            "chr21\t{}\t.\tA\t<NON_REF>\t.\t.\t{info}\t{}\t{}",
            pos0 + 1,
            f.join(":"),
            s.join(":")
        ))
    }

    // Replaces tests/unit/test_genotype_builder.py::test_qc_ref_block_clipping_dp_and_gq20 (K6).
    #[test]
    fn qc_ref_block_python_case() {
        let mut q = [0; N_QC];
        // [101, 200] hom-ref, GQ 25, MIN_DP 10, clipped to shard [151, 1000].
        let b = qblk(100, 200, "0/0", Some(25), Some(10), None);
        qc_ref_block(&mut q, &b, RB_HOM_REF, 25, 101, 200, 151, 1000).unwrap();
        // [301, 310] hom-ref, GQ 5: not >= 20; INFO DP fallback.
        let b = qblk(300, 310, "0/0", Some(5), None, Some(4));
        qc_ref_block(&mut q, &b, RB_HOM_REF, 5, 301, 310, 151, 1000).unwrap();
        // [401, 405] no-call block.
        let b = qblk(400, 405, "./.", None, None, None);
        qc_ref_block(&mut q, &b, RB_NO_CALL, 0, 401, 405, 151, 1000).unwrap();
        // Entirely outside the shard: ignored.
        let b = qblk(10, 20, "0/0", Some(99), None, None);
        qc_ref_block(&mut q, &b, RB_HOM_REF, 99, 11, 20, 151, 1000).unwrap();
        assert_eq!(q[QC_RB_BASES], 50 + 10);
        assert_eq!(q[QC_RB_DP_SUM], 50 * 10 + 10 * 4);
        assert_eq!(q[QC_RB_GQ20], 50);
        assert_eq!(q[QC_RB_NO_CALL], 5);
    }
}
