// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The Fisher exact test behind INFO/FS: scipy 1.17's
//! `fisher_exact(table)` two-sided branch, ported as it is
//! (`scipy/stats/_stats_py.py` `fisher_exact`, `_binomtest.py`
//! `_binary_search_for_binom_tst`). scipy's hypergeometric pmf / cdf / sf
//! are Boost's; here they are not bit-identical to it, so the contract
//! with the Python engine's output allowed FS to differ by exactly 0.001
//! at a rounding tie, and the test below counts it over the fixture's
//! `sb` tables.
//!
//! **The pmf.** The first design computed every pmf as `exp` of
//! log-factorial differences. Its relative
//! error (~1e-13) is above scipy's equality tolerance of 1e-14, so where
//! the table's own pmf equals the mode's exactly (a distribution with two
//! modes) or a value on the other side of the mode, the branch flipped:
//! 261 of the fixture's 103,658 tables got another FS (e.g. `[0, 1, 75,
//! 74]`: 3.01 against scipy's 0). Here only the mode's pmf comes from the
//! `ln k!` table; every other value is the mode's times the product of the
//! exact term ratios `pmf(k + 1) / pmf(k) = (n1 − k)(n − k) / ((k + 1)(n2 −
//! n + k + 1))` walked out from the mode (each ratio one division of two
//! integer products, exact below 2^53), so mathematically equal values
//! (two modes: a ratio of exactly 1) are equal in floating point, and the
//! relative error grows by one rounding per step from the mode.

use crate::pyfmt::py_round;

/// The test with its `ln k!` table, grown on demand (one `ln` per k, summed
/// in f64), and a scratch buffer of the pmf over the support, so one
/// instance per thread serves every site.
#[derive(Debug, Default)]
pub struct Fisher {
    ln_fact: Vec<f64>,
    pmf: Vec<f64>,
}

/// One table's hypergeometric distribution: `pmf[k - lo]` for `k` in the
/// support `[lo, hi]` (`hypergeom(total, good, draws)`).
struct Dist<'a> {
    lo: i64,
    hi: i64,
    pmf: &'a [f64],
}

impl Dist<'_> {
    /// `hypergeom.pmf(k)`: 0 outside the support.
    fn pmf(&self, k: i64) -> f64 {
        if k < self.lo || k > self.hi {
            0.0
        } else {
            self.pmf[(k - self.lo) as usize]
        }
    }

    /// `hypergeom.cdf(k)`: the pmf summed from the support's low end to `k`.
    fn cdf(&self, k: i64) -> f64 {
        if k < self.lo {
            return 0.0;
        }
        (self.lo..=k.min(self.hi)).map(|x| self.pmf(x)).sum()
    }

    /// `hypergeom.sf(k)`: the pmf summed above `k` to the support's high end.
    fn sf(&self, k: i64) -> f64 {
        if k >= self.hi {
            return 0.0;
        }
        ((k + 1).max(self.lo)..=self.hi).map(|x| self.pmf(x)).sum()
    }
}

impl Fisher {
    fn ensure(&mut self, n: usize) {
        if self.ln_fact.is_empty() {
            self.ln_fact.push(0.0); // ln 0! = 0
        }
        while self.ln_fact.len() <= n {
            let k = self.ln_fact.len();
            let prev = self.ln_fact[k - 1];
            self.ln_fact.push(prev + (k as f64).ln());
        }
    }

    fn ln_choose(&self, n: i64, k: i64) -> f64 {
        let (n, k) = (n as usize, k as usize);
        self.ln_fact[n] - self.ln_fact[k] - self.ln_fact[n - k]
    }

    /// Fill `self.pmf` over the support of `hypergeom(n1 + n2, n1, n)`,
    /// walking out from `mode` (clamped into the support).
    fn fill(&mut self, n1: i64, n2: i64, n: i64, mode: i64) -> (i64, i64) {
        let total = n1 + n2;
        self.ensure(total as usize);
        let lo = 0.max(n - n2);
        let hi = n1.min(n);
        let m = mode.clamp(lo, hi);
        let pm =
            (self.ln_choose(n1, m) + self.ln_choose(n2, n - m) - self.ln_choose(total, n)).exp();
        self.pmf.clear();
        self.pmf.resize((hi - lo + 1) as usize, 0.0);
        let at = |k: i64| (k - lo) as usize;
        self.pmf[at(m)] = pm;
        for k in m..hi {
            // pmf(k + 1) / pmf(k)
            let num = ((n1 - k) * (n - k)) as f64;
            let den = ((k + 1) * (n2 - n + k + 1)) as f64;
            self.pmf[at(k + 1)] = self.pmf[at(k)] * (num / den);
        }
        for k in (lo + 1..=m).rev() {
            // pmf(k - 1) / pmf(k)
            let num = (k * (n2 - n + k)) as f64;
            let den = ((n1 - k + 1) * (n - k + 1)) as f64;
            self.pmf[at(k - 1)] = self.pmf[at(k)] * (num / den);
        }
        (lo, hi)
    }

    /// The two-sided p-value of `[[a, b], [c, d]]` (non-negative counts).
    pub fn two_sided(&mut self, table: [i64; 4]) -> f64 {
        let [a, b, c, d] = table;
        // A zero row or column sum: p = 1 (the odds ratio NaN).
        if a + c == 0 || b + d == 0 || a + b == 0 || c + d == 0 {
            return 1.0;
        }
        let n1 = a + b;
        let n2 = c + d;
        let n = a + c;
        // `int((n + 1) * (n1 + 1) / (n1 + n2 + 2))`: numpy int64 true
        // division (float64), truncated.
        let mode = (((n + 1) * (n1 + 1)) as f64 / (n1 + n2 + 2) as f64) as i64;
        let (lo, hi) = self.fill(n1, n2, n, mode);
        let dist = Dist {
            lo,
            hi,
            pmf: &self.pmf,
        };
        let pexact = dist.pmf(a);
        let pmode = dist.pmf(mode);
        let epsilon = 1e-14;
        let gamma = 1.0 + epsilon;
        if (pexact - pmode).abs() / pexact.max(pmode) <= epsilon {
            return 1.0;
        }
        let p = if a < mode {
            let plower = dist.cdf(a);
            if dist.pmf(n) > pexact * gamma {
                return plower;
            }
            let guess = binary_search(|x| -dist.pmf(x), -pexact * gamma, mode, n);
            plower + dist.sf(guess)
        } else {
            let pupper = dist.sf(a - 1);
            if dist.pmf(0) > pexact * gamma {
                return pupper;
            }
            let guess = binary_search(|x| dist.pmf(x), pexact * gamma, 0, mode);
            pupper + dist.cdf(guess)
        };
        p.min(1.0)
    }

    /// INFO/FS as the Python engine computes it (`calculate_info_fields`):
    /// `round(-10 · log10(max(p, 1e-30)), 3) + 0.0` over the `sb` table
    /// `[[sb1, sb2], [sb3, sb4]]`.
    pub fn fs(&mut self, sb: [i64; 4]) -> f64 {
        let p = self.two_sided(sb);
        py_round(-10.0 * p.max(1e-30).log10())
    }
}

/// `_binary_search_for_binom_tst(a, d, lo, hi)`: the index `i` in
/// `[lo, hi]` with `a(i) <= d < a(i + 1)` over a function ascending there.
fn binary_search(a: impl Fn(i64) -> f64, d: f64, mut lo: i64, mut hi: i64) -> i64 {
    while lo < hi {
        let mid = lo + (hi - lo).div_euclid(2);
        let midval = a(mid);
        if midval < d {
            lo = mid + 1;
        } else if midval > d {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    if a(lo) <= d { lo } else { lo - 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scipy_examples() {
        let mut f = Fisher::default();
        // The docstring's examples.
        let p = f.two_sided([6, 2, 1, 4]);
        assert!((p - 0.10256410256410257).abs() < 1e-15, "{p}");
        let p = f.two_sided([8, 2, 1, 5]);
        assert!((p - 0.034965034965034975).abs() < 1e-15, "{p}");
        assert_eq!(f.two_sided([0, 0, 3, 4]), 1.0);
        assert_eq!(f.two_sided([0, 5, 0, 4]), 1.0);
        assert_eq!(f.two_sided([0, 0, 0, 0]), 1.0);
        // p = 1 → FS 0.0, not -0.0 (no negative zero in INFO).
        assert_eq!(f.fs([0, 0, 0, 0]).to_bits(), 0.0f64.to_bits());
        assert_eq!(f.fs([5, 5, 5, 5]).to_bits(), 0.0f64.to_bits());
    }

    /// Every distinct `sb` table of the six-sample cohort's `site_info`,
    /// with scipy's p and the Python engine's FS
    /// (`rust/tools/gen_fisher_table.py`): the FS values that differ after
    /// rounding are counted and printed; only a difference of exactly
    /// 0.001 is allowed, and on the fixture none is expected. The
    /// checked-in table is 4,000 of the fixture's tables;
    /// `MOSAIC_FISHER_TABLE=<path>` reads another (the full list).
    #[test]
    fn against_scipy() {
        let p = match std::env::var_os("MOSAIC_FISHER_TABLE") {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/fisher_sb.tsv")
            }
        };
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut f = Fisher::default();
        let (mut n, mut fs_diff, mut fs_bad) = (0usize, 0usize, Vec::new());
        let (mut max_abs, mut max_rel) = (0f64, 0f64);
        for line in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let cols: Vec<&str> = line.split('\t').collect();
            let sb: [i64; 4] = std::array::from_fn(|i| cols[i].parse().unwrap());
            let want_p = f64::from_bits(u64::from_str_radix(cols[4], 16).unwrap());
            let want_fs = f64::from_bits(u64::from_str_radix(cols[5], 16).unwrap());
            let got_p = f.two_sided(sb);
            let got_fs = f.fs(sb);
            let d = (got_p - want_p).abs();
            max_abs = max_abs.max(d);
            if want_p > 0.0 {
                max_rel = max_rel.max(d / want_p);
            }
            if got_fs.to_bits() != want_fs.to_bits() {
                fs_diff += 1;
                if ((got_fs - want_fs).abs() - 0.001).abs() > 1e-9 {
                    fs_bad.push(format!("{sb:?}: fs {got_fs} vs {want_fs}"));
                }
            }
            n += 1;
        }
        println!(
            "fisher vs scipy: {n} tables, FS differs at {fs_diff}, max |dp| {max_abs:e}, max rel {max_rel:e}"
        );
        assert!(n > 100, "{n} tables");
        assert!(fs_bad.is_empty(), "{}", fs_bad.join("\n"));
    }
}
