// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `--dump-groups` (one JSON line per grouper yield; it was the
//! differential against the Python grouper this one was ported from) and
//! `--count-records` (the parse-only pass).

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use crate::grouper::{Grouper, Source};
use crate::reader::{Input, ShardReader};
use crate::record::{FMT_KEYS, INFO_KEYS, Record};
use crate::value::Value;

impl Source for ShardReader {
    type Rec = Record;
    fn next_rec(&mut self) -> Result<Option<Record>> {
        self.next_record()
    }
}

/// `"f64:<16 hex digits of the IEEE 754 bits>"`.
fn put_f64(out: &mut Vec<u8>, x: f64) {
    write!(out, "\"f64:{:016x}\"", x.to_bits()).expect("vec write");
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    serde_json::to_writer(&mut *out, s).expect("vec write");
}

fn put_list<T>(out: &mut Vec<u8>, xs: &[T], mut f: impl FnMut(&mut Vec<u8>, &T)) {
    out.push(b'[');
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        f(out, x);
    }
    out.push(b']');
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Int(i) => write!(out, "{i}").expect("vec write"),
        Value::Float(x) => put_f64(out, *x),
        Value::Str(s) => put_str(out, s),
        Value::Bools(xs) => put_list(out, xs, |o, b| {
            o.extend_from_slice(if *b { b"true" } else { b"false" })
        }),
        Value::Ints(xs) => put_list(out, xs, |o, i| write!(o, "{i}").expect("vec write")),
        Value::Floats(xs) => put_list(out, xs, |o, x| put_f64(o, *x)),
        Value::Strs(xs) => put_list(out, xs, |o, s| put_str(o, s)),
    }
}

fn put_fields(out: &mut Vec<u8>, keys: &[&str], vals: &[Option<Value>]) {
    out.push(b'{');
    let mut first = true;
    for (k, v) in keys.iter().zip(vals) {
        if let Some(v) = v {
            if !first {
                out.push(b',');
            }
            first = false;
            put_str(out, k);
            out.push(b':');
            put_value(out, v);
        }
    }
    out.push(b'}');
}

pub fn put_record(out: &mut Vec<u8>, r: &Record) {
    write!(out, "{{\"pos\":{},\"end\":{},\"ref\":", r.pos, r.end).expect("vec write");
    put_str(out, &r.ref_);
    out.extend_from_slice(b",\"alt\":");
    put_list(out, &r.alt, |o, s| put_str(o, s));
    out.extend_from_slice(b",\"qual\":");
    match r.qual {
        Some(q) => put_f64(out, q),
        None => out.extend_from_slice(b"null"),
    }
    out.extend_from_slice(b",\"filter\":");
    put_list(out, &r.filter, |o, s| put_str(o, s));
    out.extend_from_slice(b",\"info\":");
    put_fields(out, &INFO_KEYS, &r.info);
    out.extend_from_slice(b",\"s0\":");
    match &r.s0 {
        Some(s0) => put_fields(out, &FMT_KEYS, s0),
        None => out.extend_from_slice(b"null"),
    }
    write!(out, ",\"homref\":{}}}", r.homref).expect("vec write");
}

/// A shard to read: chromosome, 0-based start, exclusive end.
pub type ShardRange = (String, i64, i64);

/// Per-run totals.
#[derive(Debug, Default)]
pub struct Totals {
    pub shards: u64,
    pub yields: u64,
    pub records: u64,
    pub records_dropped: u64,
}

/// Dump every shard's group sequence to `out`: a `{"shard":[c,s,e]}` line,
/// then one line per yield.
pub fn dump_groups(
    inputs: &[Arc<Input>],
    shards: &[ShardRange],
    out: &mut dyn Write,
) -> Result<Totals> {
    let mut readers: Vec<Option<ShardReader>> = inputs
        .iter()
        .map(|i| ShardReader::new(i.clone()).map(Some))
        .collect::<Result<_>>()?;
    let mut tot = Totals::default();
    let mut line = Vec::with_capacity(4096);
    for (chrom, start, end) in shards {
        for r in readers.iter_mut().flatten() {
            r.start_shard(chrom, *start, *end);
        }
        line.clear();
        line.extend_from_slice(b"{\"shard\":[");
        put_str(&mut line, chrom);
        writeln!(line, ",{start},{end}]}}")?;
        out.write_all(&line)?;
        let mut g = Grouper::new(std::mem::take(&mut readers))?;
        while let Some(pos) = g.next_group()? {
            line.clear();
            write!(line, "{{\"pos\":{pos},\"grp\":[")?;
            for (i, slot) in g.grp.iter().enumerate() {
                if i > 0 {
                    line.push(b',');
                }
                match slot {
                    Some(r) => put_record(&mut line, r),
                    None => line.extend_from_slice(b"null"),
                }
            }
            line.extend_from_slice(b"],\"ovl\":[");
            for (i, (k, v, j)) in g.ovl.iter().enumerate() {
                if i > 0 {
                    line.push(b',');
                }
                write!(line, "[{k},")?;
                put_record(&mut line, v);
                write!(line, ",{j}]")?;
            }
            line.extend_from_slice(b"]}\n");
            out.write_all(&line)?;
            tot.yields += 1;
        }
        let mut back = g.into_sources();
        let mut shard_dropped = 0;
        for r in back.iter_mut().flatten() {
            shard_dropped += r.records_dropped;
            r.records_dropped = 0;
        }
        tot.records_dropped += shard_dropped;
        tot.shards += 1;
        readers = back;
    }
    Ok(tot)
}

/// Count the shards' records through the reader alone (no grouper).
pub fn count_records(inputs: &[Arc<Input>], shards: &[ShardRange]) -> Result<Totals> {
    let mut tot = Totals::default();
    let mut readers: Vec<ShardReader> = inputs
        .iter()
        .map(|i| ShardReader::new(i.clone()))
        .collect::<Result<_>>()?;
    for (chrom, start, end) in shards {
        for r in readers.iter_mut() {
            r.start_shard(chrom, *start, *end);
            while r.next_record()?.is_some() {
                tot.records += 1;
            }
            tot.records_dropped += r.records_dropped;
            r.records_dropped = 0;
        }
        tot.shards += 1;
    }
    Ok(tot)
}

/// Time a closure.
pub fn timed<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    let t = Instant::now();
    let v = f()?;
    Ok((v, t.elapsed().as_secs_f64()))
}
