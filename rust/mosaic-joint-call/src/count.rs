// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `--count-rows` (the streaming reader's check): for each unit, the streaming
//! reader over both per-sample tables, counting the rows in the unit after
//! the tombstone mask, the row groups selected, the rows decoded and the
//! bytes read. One stderr line per unit, in unit order; one JSON summary
//! line on stdout.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures::{StreamExt, TryStreamExt};
use serde::Serialize;

use crate::manifest::Manifest;
use crate::reader::{Counters, Mask, Merge, OpenFile, Table};
use crate::store::Stores;

/// One unit: a contig's window `[ws, we]` (1-based, inclusive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub contig: usize,
    pub ws: i64,
    pub we: i64,
}

/// The units of `m` in genomic order: per contig (or the region's one),
/// windows of `window` bp from its start (0: one unit), as the Python
/// engine's window grid.
pub fn plan_units(m: &Manifest, window: i64) -> Vec<Unit> {
    let mut units = Vec::new();
    for (ci, c) in m.contigs.iter().enumerate() {
        let (start, last) = match &m.region {
            Some(r) if r.contig != c.name => continue,
            Some(r) => (r.start, r.end),
            None => (1, c.length),
        };
        if window == 0 {
            units.push(Unit {
                contig: ci,
                ws: start,
                we: last,
            });
            continue;
        }
        let mut ws = start;
        while ws <= last {
            let we = (ws + window - 1).min(last);
            units.push(Unit { contig: ci, ws, we });
            ws += window;
        }
    }
    units
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TableCounts {
    pub rows_in_unit: u64,
    pub row_groups: u64,
    pub rows_decoded: u64,
    pub rows_masked: u64,
    pub bytes_requested: u64,
    pub bytes_fetched: u64,
    pub requests: u64,
}

impl TableCounts {
    fn add(&mut self, o: &TableCounts) {
        self.rows_in_unit += o.rows_in_unit;
        self.row_groups += o.row_groups;
        self.rows_decoded += o.rows_decoded;
        self.rows_masked += o.rows_masked;
        self.bytes_requested += o.bytes_requested;
        self.bytes_fetched += o.bytes_fetched;
        self.requests += o.requests;
    }
}

struct UnitResult {
    tables: [TableCounts; 2],
    secs: f64,
}

async fn count_unit(
    files: [&[Arc<OpenFile>]; 2],
    removed: Arc<HashSet<i32>>,
    ws: i64,
    we: i64,
) -> Result<[TableCounts; 2]> {
    // Both tables' readers start at once; the merges run one after the
    // other (J2 advances them together).
    let counters = [Arc::new(Counters::default()), Arc::new(Counters::default())];
    let mut merges: Vec<Merge> = Table::ALL
        .iter()
        .enumerate()
        .map(|(t, &table)| {
            Merge::start(
                table,
                files[t],
                ws,
                we,
                Mask::Drop(removed.clone()),
                counters[t].clone(),
            )
        })
        .collect();
    let mut out: [TableCounts; 2] = Default::default();
    for (t, &table) in Table::ALL.iter().enumerate() {
        let merge = &mut merges[t];
        let mut n = 0;
        while let Some(r) = merge.next().await? {
            if table.in_unit(r.key, r.end, ws, we) {
                n += 1;
            }
        }
        let (decoded, masked) = merge.decoded_masked();
        let (req, fetched, requests) = counters[t].get();
        out[t] = TableCounts {
            rows_in_unit: n,
            row_groups: merge.row_groups,
            rows_decoded: decoded,
            rows_masked: masked,
            bytes_requested: req,
            bytes_fetched: fetched,
            requests,
        };
    }
    Ok(out)
}

/// Thousands separators, as the Python engine's `{:,}`.
pub fn commas(n: i64) -> String {
    let s = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    if n < 0 { format!("-{out}") } else { out }
}

fn unit_line(i: usize, n: usize, name: &str, u: &Unit, r: &UnitResult) -> String {
    let mut s = format!(
        "Unit {}/{}: {}:{}-{}",
        i + 1,
        n,
        name,
        commas(u.ws),
        commas(u.we)
    );
    for (t, table) in Table::ALL.iter().enumerate() {
        let c = &r.tables[t];
        s.push_str(&format!(
            " | {} rows={} row_groups={} decoded={} masked={} bytes={} fetched={}",
            table.name(),
            c.rows_in_unit,
            c.row_groups,
            c.rows_decoded,
            c.rows_masked,
            c.bytes_requested,
            c.bytes_fetched
        ));
    }
    s.push_str(&format!(" ({:.2} s)", r.secs));
    s
}

#[derive(Serialize)]
struct TableSummary {
    files: usize,
    file_bytes: u64,
    footer_bytes: u64,
    footer_requests: u64,
    #[serde(flatten)]
    counts: TableCounts,
}

/// Run `--count-rows`; the summary line is printed by the caller.
pub fn run(m: &Manifest, threads: usize, window: i64, rt: &tokio::runtime::Runtime) -> Result<()> {
    let t0 = Instant::now();
    let stores = Arc::new(Stores::new(m.s3.clone()));
    // Every data file's footer, once (concurrently, 16 in flight).
    let mut opened: Vec<[Vec<Arc<OpenFile>>; 2]> = Vec::new();
    for c in &m.contigs {
        let mut per: [Vec<Arc<OpenFile>>; 2] = Default::default();
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
        opened.push(per);
    }
    let footer_secs = t0.elapsed().as_secs_f64();

    let units = plan_units(m, window);
    let removed: Arc<HashSet<i32>> = Arc::new(m.removed_ids.iter().copied().collect());
    let handle = rt.handle().clone();
    let t1 = Instant::now();
    let work = |_: &mut (), i: usize| -> Result<UnitResult> {
        let u = &units[i];
        let s = Instant::now();
        let files = [&opened[u.contig][0][..], &opened[u.contig][1][..]];
        let tables = handle.block_on(count_unit(files, removed.clone(), u.ws, u.we))?;
        Ok(UnitResult {
            tables,
            secs: s.elapsed().as_secs_f64(),
        })
    };
    let mut done: BTreeMap<usize, UnitResult> = BTreeMap::new();
    let mut next = 0;
    let mut totals: [TableCounts; 2] = Default::default();
    let mut per_unit = Vec::new();
    let mut error = None;
    let stderr = std::io::stderr();
    let flow = mosaic_core::queue::run_queue(units.len(), threads, work, |i, r| match r {
        Ok(r) => {
            done.insert(i, r);
            while let Some(r) = done.remove(&next) {
                let u = &units[next];
                let name = &m.contigs[u.contig].name;
                let _ = writeln!(
                    stderr.lock(),
                    "{}",
                    unit_line(next, units.len(), name, u, &r)
                );
                for (tot, c) in totals.iter_mut().zip(&r.tables) {
                    tot.add(c);
                }
                per_unit.push((
                    name.clone(),
                    u.ws,
                    u.we,
                    r.tables[0].rows_in_unit,
                    r.tables[1].rows_in_unit,
                    r.secs,
                ));
                next += 1;
            }
            ControlFlow::Continue(())
        }
        Err(e) => {
            let u = &units[i];
            error = Some(e.context(format!(
                "unit {}:{}-{}",
                m.contigs[u.contig].name, u.ws, u.we
            )));
            ControlFlow::Break(())
        }
    });
    if let Some(e) = error {
        return Err(e);
    }
    if flow.is_break() || next != units.len() {
        return Err(anyhow!("{} of {} units finished", next, units.len()));
    }
    let unit_secs = t1.elapsed().as_secs_f64();

    let mut tables = serde_json::Map::new();
    for (t, table) in Table::ALL.iter().enumerate() {
        let files: Vec<&Arc<OpenFile>> = opened.iter().flat_map(|p| p[t].iter()).collect();
        let s = TableSummary {
            files: files.len(),
            file_bytes: files.iter().map(|f| f.size).sum(),
            footer_bytes: files.iter().map(|f| f.footer.1).sum(),
            footer_requests: files.iter().map(|f| f.footer.2).sum(),
            counts: totals[t].clone(),
        };
        tables.insert(table.name().to_string(), serde_json::to_value(s)?);
    }
    let summary = serde_json::json!({
        "kind": "summary",
        "mode": "count_rows",
        "units": units.len(),
        "threads": threads,
        "window": window,
        "removed_ids": m.removed_ids.len(),
        "tables": tables,
        "seconds": {
            "footers": footer_secs,
            "units": unit_secs,
            "total": t0.elapsed().as_secs_f64(),
        },
        "unit_counts": per_unit,
    });
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commas_like_python() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1000), "1,000");
        assert_eq!(commas(46_709_983), "46,709,983");
        assert_eq!(commas(-1234567), "-1,234,567");
    }

    #[test]
    fn units_grid() {
        let text = r#"{"samples": [], "contigs": [
            {"index": 0, "name": "a", "length": 25, "sample_site_info": [], "sample_ref_blocks": []},
            {"index": 1, "name": "b", "length": 7, "sample_site_info": [], "sample_ref_blocks": []}]}"#;
        let mut m: Manifest = serde_json::from_str(text).unwrap();
        let w = |m: &Manifest, w| {
            plan_units(m, w)
                .iter()
                .map(|u| (u.contig, u.ws, u.we))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            w(&m, 10),
            vec![(0, 1, 10), (0, 11, 20), (0, 21, 25), (1, 1, 7)]
        );
        assert_eq!(w(&m, 0), vec![(0, 1, 25), (1, 1, 7)]);
        assert_eq!(w(&m, 100), vec![(0, 1, 25), (1, 1, 7)]);
        m.region = Some(crate::manifest::Region {
            contig: "a".into(),
            start: 5,
            end: 22,
        });
        assert_eq!(w(&m, 10), vec![(0, 5, 14), (0, 15, 22)]);
        assert_eq!(w(&m, 0), vec![(0, 5, 22)]);
    }
}

#[cfg(test)]
mod reader_tests {
    //! The streaming reader over `file://` Parquet fixtures written here
    //! (no store, no network): row groups by statistics, the k-way merge
    //! in `(key, sample_id)` order, the tombstone mask, and the refusals.
    use super::*;
    use crate::manifest::DataFile;
    use crate::reader::Row;
    use arrow::array::{ArrayRef, Int32Array, RecordBatch, new_null_array};
    use arrow::datatypes::SchemaRef;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mosaic-jc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A batch of `schema` with the given Int32 columns, every other
    /// (nullable) column null.
    fn batch(schema: &SchemaRef, n: usize, cols: &[(&str, Vec<i32>)]) -> RecordBatch {
        let arrays: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .map(|f| match cols.iter().find(|(name, _)| name == f.name()) {
                Some((_, v)) => Arc::new(Int32Array::from(v.clone())) as ArrayRef,
                None if f.name() == "chromosome_index" => Arc::new(Int32Array::from(vec![0; n])),
                None => new_null_array(f.data_type(), n),
            })
            .collect();
        RecordBatch::try_new(schema.clone(), arrays).unwrap()
    }

    fn write(
        path: &std::path::Path,
        b: &RecordBatch,
        rows_per_group: usize,
        stats: bool,
    ) -> DataFile {
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_group))
            .set_statistics_enabled(if stats {
                EnabledStatistics::Chunk
            } else {
                EnabledStatistics::None
            })
            .build();
        let f = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(f, b.schema(), Some(props)).unwrap();
        w.write(b).unwrap();
        w.close().unwrap();
        DataFile {
            uri: format!("file://{}", path.display()),
            record_count: b.num_rows() as u64,
            file_size_in_bytes: std::fs::metadata(path).unwrap().len(),
            delete_files: vec![],
        }
    }

    /// Ref blocks of samples 1..=3 over [1, 1000], sorted (start, sample):
    /// sample 1 one long block [1, 500] then short ones; samples 2 / 3
    /// 50 bp blocks (sample 3 removed in the tests).
    fn ref_rows(samples: &[i32]) -> Vec<(i32, i32, i32)> {
        let mut rows = Vec::new();
        for &s in samples {
            if s == 1 {
                rows.push((1, s, 500));
                let mut p = 501;
                while p <= 1000 {
                    rows.push((p, s, p + 9));
                    p += 10;
                }
            } else {
                let mut p = s; // offset starts
                while p <= 1000 {
                    rows.push((p, s, (p + 49).min(1000)));
                    p += 50;
                }
            }
        }
        rows.sort();
        rows
    }

    fn ref_file(
        dir: &std::path::Path,
        name: &str,
        samples: &[i32],
        rpg: usize,
    ) -> (DataFile, Vec<(i32, i32, i32)>) {
        let rows = ref_rows(samples);
        let b = batch(
            &mosaic_core::schemas::sample_ref_blocks(),
            rows.len(),
            &[
                ("position_start", rows.iter().map(|r| r.0).collect()),
                ("sample_id", rows.iter().map(|r| r.1).collect()),
                ("position_end", rows.iter().map(|r| r.2).collect()),
            ],
        );
        (write(&dir.join(name), &b, rpg, true), rows)
    }

    /// Sites: every 7th position for sample s, plus two rows of sample 2
    /// at one position (a second record at one POS).
    fn site_file(
        dir: &std::path::Path,
        name: &str,
        samples: &[i32],
        rpg: usize,
    ) -> (DataFile, Vec<(i32, i32)>) {
        let mut rows = Vec::new();
        for &s in samples {
            let mut p = s;
            while p <= 1000 {
                rows.push((p, s));
                p += 7;
            }
            if s == 2 {
                rows.push((2, 2));
            }
        }
        rows.sort();
        let b = batch(
            &mosaic_core::schemas::sample_site_info(),
            rows.len(),
            &[
                ("position", rows.iter().map(|r| r.0).collect()),
                ("sample_id", rows.iter().map(|r| r.1).collect()),
            ],
        );
        (write(&dir.join(name), &b, rpg, true), rows)
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    fn open_all(
        rt: &tokio::runtime::Runtime,
        table: Table,
        dfs: &[DataFile],
    ) -> Result<Vec<Arc<OpenFile>>> {
        let stores = Stores::new(None);
        dfs.iter()
            .map(|df| {
                rt.block_on(OpenFile::open(&stores, table, df))
                    .map(Arc::new)
            })
            .collect()
    }

    #[test]
    fn counts_equal_brute_force_and_merge_is_ordered() {
        let dir = tmpdir("count");
        let rt = rt();
        let (r1, rows1) = ref_file(&dir, "rb1.parquet", &[1, 3], 16);
        let (r2, rows2) = ref_file(&dir, "rb2.parquet", &[2], 5);
        let (s1, srows1) = site_file(&dir, "si1.parquet", &[1, 3], 8);
        let (s2, srows2) = site_file(&dir, "si2.parquet", &[2], 3);
        let rb = open_all(&rt, Table::RefBlocks, &[r1, r2]).unwrap();
        let si = open_all(&rt, Table::SiteInfo, &[s1, s2]).unwrap();
        let all_rb: Vec<_> = rows1.iter().chain(&rows2).copied().collect();
        let all_si: Vec<_> = srows1.iter().chain(&srows2).copied().collect();
        for removed in [vec![], vec![3]] {
            let rm: Arc<HashSet<i32>> = Arc::new(removed.iter().copied().collect());
            for (ws, we) in [
                (1, 1000),
                (1, 100),
                (301, 400),
                (495, 505),
                (990, 1000),
                (1001, 2000),
                (500, 500),
            ] {
                let got = rt
                    .block_on(count_unit([&si[..], &rb[..]], rm.clone(), ws, we))
                    .unwrap();
                let want_si = all_si
                    .iter()
                    .filter(|r| !rm.contains(&r.1) && (ws..=we).contains(&i64::from(r.0)))
                    .count() as u64;
                let want_rb = all_rb
                    .iter()
                    .filter(|r| !rm.contains(&r.1) && i64::from(r.0) <= we && i64::from(r.2) >= ws)
                    .count() as u64;
                assert_eq!(got[0].rows_in_unit, want_si, "si {ws}-{we} {removed:?}");
                assert_eq!(got[1].rows_in_unit, want_rb, "rb {ws}-{we} {removed:?}");
                if removed.is_empty() {
                    assert_eq!(got[1].rows_masked, 0);
                }
            }
        }
        // The long block [1, 500] of sample 1 sits in row group 0 of rb1:
        // a unit at [301, 400] selects that row group through
        // position_end's statistics, and counts the block.
        let rm = Arc::new(HashSet::new());
        let got = rt
            .block_on(count_unit([&si[..], &rb[..]], rm.clone(), 301, 400))
            .unwrap();
        assert!(got[1].row_groups >= 1);
        assert!(rb[0].select(301, 400).contains(&0));
        // The merged order over both files: (key, sample) non-decreasing,
        // every row once.
        let mut merge = rt.block_on(async {
            Merge::start(
                Table::RefBlocks,
                &rb,
                1,
                1000,
                Mask::Drop(rm.clone()),
                Arc::new(Counters::default()),
            )
        });
        let rows: Vec<Row> = rt.block_on(async {
            let mut v = Vec::new();
            while let Some(r) = merge.next().await.unwrap() {
                // The row's batch holds the row.
                let b = merge.batch(&r);
                let s = b
                    .column_by_name("sample_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                assert_eq!(s.value(r.row), r.sample_id);
                v.push(r);
            }
            v
        });
        let keys: Vec<(i32, i32, i32)> = rows.iter().map(|r| (r.key, r.sample_id, r.end)).collect();
        let mut want = all_rb.clone();
        want.sort();
        assert_eq!(keys, want);
        // The calling run's positive mask (K4): a row is kept iff its
        // sample is in the set; `Keep` of every sample equals no mask, and
        // `Keep({1, 2})` equals `Drop({3})` here (samples 1, 2, 3).
        let merged = |mask: Mask| -> (Vec<(i32, i32, i32)>, u64) {
            rt.block_on(async {
                let mut m = Merge::start(
                    Table::RefBlocks,
                    &rb,
                    1,
                    1000,
                    mask,
                    Arc::new(Counters::default()),
                );
                let mut v = Vec::new();
                while let Some(r) = m.next().await.unwrap() {
                    v.push((r.key, r.sample_id, r.end));
                }
                (v, m.decoded_masked().1)
            })
        };
        let set = |v: &[i32]| Arc::new(v.iter().copied().collect::<HashSet<i32>>());
        assert_eq!(merged(Mask::Keep(set(&[1, 2, 3]))), (want.clone(), 0));
        let (kept, masked) = merged(Mask::Keep(set(&[1, 2])));
        assert_eq!((kept.clone(), masked), merged(Mask::Drop(set(&[3]))));
        let only3: Vec<_> = want.iter().filter(|r| r.1 != 3).copied().collect();
        assert_eq!(kept, only3);
        assert_eq!(masked as usize, want.len() - only3.len());
        let (kept, _) = merged(Mask::Keep(set(&[2])));
        assert!(!kept.is_empty() && kept.iter().all(|r| r.1 == 2));
        // An empty `Keep` keeps nothing.
        assert_eq!(merged(Mask::Keep(set(&[]))).0, vec![]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refusals() {
        let dir = tmpdir("refuse");
        let rt = rt();
        let stores = Stores::new(None);
        // No statistics.
        let rows = ref_rows(&[1]);
        let b = batch(
            &mosaic_core::schemas::sample_ref_blocks(),
            rows.len(),
            &[
                ("position_start", rows.iter().map(|r| r.0).collect()),
                ("sample_id", rows.iter().map(|r| r.1).collect()),
                ("position_end", rows.iter().map(|r| r.2).collect()),
            ],
        );
        let nostats = write(&dir.join("nostats.parquet"), &b, 8, false);
        let e = rt
            .block_on(OpenFile::open(&stores, Table::RefBlocks, &nostats))
            .err()
            .unwrap();
        assert!(
            format!("{e:#}").contains("no min / max statistics"),
            "{e:#}"
        );
        // A missing file.
        let mut missing = nostats.clone();
        missing.uri = format!("file://{}", dir.join("none.parquet").display());
        let e = rt
            .block_on(OpenFile::open(&stores, Table::RefBlocks, &missing))
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("none.parquet"), "{e:#}");
        // A record count that disagrees with the footer.
        let (mut good, _) = ref_file(&dir, "good.parquet", &[1], 8);
        good.record_count += 1;
        let e = rt
            .block_on(OpenFile::open(&stores, Table::RefBlocks, &good))
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("in the manifest"), "{e:#}");
        // The other table's file (no `position` column).
        good.record_count -= 1;
        let e = rt
            .block_on(OpenFile::open(&stores, Table::SiteInfo, &good))
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("no column"), "{e:#}");
        // An s3:// file without s3 settings.
        let mut s3 = good.clone();
        s3.uri = "s3://bucket/key.parquet".into();
        let e = rt
            .block_on(OpenFile::open(&stores, Table::RefBlocks, &s3))
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("no s3 settings"), "{e:#}");
        // Rows out of the sort order.
        let b = batch(
            &mosaic_core::schemas::sample_site_info(),
            3,
            &[("position", vec![5, 3, 9]), ("sample_id", vec![1, 1, 1])],
        );
        let unsorted = write(&dir.join("unsorted.parquet"), &b, 8, true);
        let si = open_all(&rt, Table::SiteInfo, &[unsorted]).unwrap();
        let e = rt
            .block_on(count_unit([&si[..], &[]], Arc::new(HashSet::new()), 1, 10))
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("not sorted"), "{e:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
