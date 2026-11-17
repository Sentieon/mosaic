// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The shard range reader: bgzf + the tabix / CSI index query, then
//! records in file order with `vcflib`'s `VCFReader` semantics.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use noodles_bgzf as bgzf;

use crate::header::Header;
use crate::record::{ParseError, Record, Tokenizer};

/// One reference sequence of the index, as `Tabix.load` holds it: bins
/// (bin id -> (loffset, chunks)) and the linear index.
#[derive(Debug, Default, Clone)]
struct RefIndex {
    bins: HashMap<usize, (u64, Vec<(u64, u64)>)>,
    intvs: Vec<u64>,
}

/// The index of one input, loaded once per batch and shared.
#[derive(Debug, Clone)]
pub struct Index {
    min_shift: u32,
    max_shift: u32,
    refs: HashMap<Vec<u8>, RefIndex>,
}

impl Index {
    /// `Tabix.load`: `<path>.csi` if it exists, else `<path>.tbi`, parsed
    /// as vcflib parses them (a port: noodles-tabix refuses the
    /// `input_gvcfs4` indexes vcflib wrote, whose `col_end` is 2).
    pub fn load(path: &Path) -> Result<Index> {
        let csi = PathBuf::from(format!("{}.csi", path.display()));
        let tbi = PathBuf::from(format!("{}.tbi", path.display()));
        let (idxf, is_csi) = if csi.exists() {
            (csi, true)
        } else {
            (tbi, false)
        };
        let mut data = Vec::new();
        bgzf::io::Reader::new(BufReader::new(
            File::open(&idxf).with_context(|| format!("open {}", idxf.display()))?,
        ))
        .read_to_end(&mut data)
        .with_context(|| format!("read {}", idxf.display()))?;
        Index::parse(&data, is_csi).with_context(|| format!("{}", idxf.display()))
    }

    fn parse(data: &[u8], is_csi: bool) -> Result<Index> {
        let mut off = 0usize;
        let u32_ = |off: &mut usize| -> Result<u32> {
            let b = data
                .get(*off..*off + 4)
                .ok_or_else(|| anyhow!("truncated index"))?;
            *off += 4;
            Ok(u32::from_le_bytes(b.try_into().expect("4 bytes")))
        };
        let magic = u32_(&mut off)?;
        let want = if is_csi { 0x0149_5343 } else { 0x0149_4254 };
        if magic != want {
            bail!("Not a tabix file");
        }
        let (min_shift, depth, n_ref, names);
        if !is_csi {
            min_shift = 14;
            depth = 5;
            n_ref = u32_(&mut off)? as usize;
            off += 24; // the six-int header
            let l_nm = u32_(&mut off)? as usize;
            names = data
                .get(off..off + l_nm)
                .ok_or_else(|| anyhow!("truncated index"))?
                .to_vec();
            off += l_nm;
        } else {
            min_shift = u32_(&mut off)?;
            depth = u32_(&mut off)?;
            let l_aux = u32_(&mut off)? as usize;
            if l_aux < 28 {
                bail!("Invalid header");
            }
            off += 24;
            let l_nm = u32_(&mut off)? as usize;
            names = data
                .get(off..off + l_nm)
                .ok_or_else(|| anyhow!("truncated index"))?
                .to_vec();
            off += l_nm;
            off += l_aux - (28 + l_nm);
            n_ref = u32_(&mut off)? as usize;
        }
        let names: Vec<&[u8]> = names.split(|&c| c == 0).collect();
        if names.len() != n_ref + 1 || !names[n_ref].is_empty() {
            bail!("Header sequence name length mismatch");
        }
        let u64_ = |off: &mut usize| -> Result<u64> {
            let b = data
                .get(*off..*off + 8)
                .ok_or_else(|| anyhow!("truncated index"))?;
            *off += 8;
            Ok(u64::from_le_bytes(b.try_into().expect("8 bytes")))
        };
        let mut refs = HashMap::new();
        for name in names.iter().take(n_ref) {
            let mut ri = RefIndex::default();
            let n_bin = u32_(&mut off)?;
            for _ in 0..n_bin {
                let bin = u32_(&mut off)? as usize;
                let loffset = if is_csi { u64_(&mut off)? } else { 0 };
                let n_chunk = u32_(&mut off)?;
                let mut chunks = Vec::with_capacity(n_chunk as usize);
                for _ in 0..n_chunk {
                    let s = u64_(&mut off)?;
                    let e = u64_(&mut off)?;
                    chunks.push((s, e));
                }
                ri.bins.insert(bin, (loffset, chunks));
            }
            if !is_csi {
                let n_intv = u32_(&mut off)?;
                for _ in 0..n_intv {
                    ri.intvs.push(u64_(&mut off)?);
                }
                if n_intv == 0 {
                    ri.intvs.push(0);
                }
            }
            refs.insert(name.to_vec(), ri);
        }
        Ok(Index {
            min_shift,
            max_shift: min_shift + depth * 3,
            refs,
        })
    }

    /// `Tabix.query(c, s, e)`: the virtual-offset ranges to read, merged
    /// at block granularity (`Tabix.merge(ranges, 16)`).
    pub fn query(&self, chrom: &str, s: i64, e: i64) -> Vec<(u64, u64)> {
        let Some(ci) = self.refs.get(chrom.as_bytes()) else {
            return Vec::new();
        };
        let s = s.max(0);
        let i = (s >> self.min_shift) as usize;
        let mut minoff = if ci.intvs.is_empty() {
            0
        } else {
            ci.intvs[i.min(ci.intvs.len() - 1)]
        };
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut shift = self.max_shift as i64;
        while shift > self.min_shift as i64 - 3 {
            let bo = ((1i64 << (self.max_shift as i64 - shift)) - 1) / 7;
            let bs = bo + (s >> shift);
            let be = bo + ((e - 1) >> shift);
            if ci.intvs.is_empty() {
                let mut bi = bs;
                while bi > bo - 1 {
                    if let Some(b) = usize::try_from(bi).ok().and_then(|x| ci.bins.get(&x)) {
                        minoff = minoff.max(b.0);
                        break;
                    }
                    bi -= 1;
                }
            }
            let mut bi = bs;
            while bi <= be {
                if let Some(b) = usize::try_from(bi).ok().and_then(|x| ci.bins.get(&x)) {
                    ranges.extend_from_slice(&b.1);
                }
                bi += 1;
            }
            shift -= 3;
        }
        if minoff > 0 {
            ranges = ranges
                .into_iter()
                .filter(|r| r.1 > minoff)
                .map(|(a, b)| (a.max(minoff), b))
                .collect();
        }
        merge(ranges, 16)
    }
}

/// `Tabix.merge`.
fn merge(mut ranges: Vec<(u64, u64)>, shift: u32) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    let mut p: Option<(u64, u64)> = None;
    for r in ranges {
        p = Some(match p {
            None => r,
            Some(q) if (r.0 >> shift) > (q.1 >> shift) => {
                out.push(q);
                r
            }
            Some(q) => (q.0, q.1.max(r.1)),
        });
    }
    out.extend(p);
    out
}

/// One gVCF input: its path, header and index, shared by the threads.
#[derive(Debug)]
pub struct Input {
    pub path: PathBuf,
    pub header: Header,
    pub index: Index,
}

impl Input {
    pub fn open(path: &Path) -> Result<Arc<Input>> {
        let mut r = bgzf::io::Reader::new(BufReader::new(
            File::open(path).with_context(|| format!("open {}", path.display()))?,
        ));
        let mut lines = Vec::new();
        let mut buf = Vec::new();
        loop {
            buf.clear();
            if r.read_until(b'\n', &mut buf)? == 0 || buf.first() != Some(&b'#') {
                break;
            }
            let s = String::from_utf8(buf.clone())
                .with_context(|| format!("{}: a header line is not UTF-8", path.display()))?;
            lines.push(s.trim_end().to_string());
        }
        let header = Header::parse(&lines).with_context(|| format!("{}", path.display()))?;
        let index = Index::load(path)?;
        Ok(Arc::new(Input {
            path: path.to_path_buf(),
            header,
            index,
        }))
    }
}

/// A bgzf reader of one input, re-targeted per shard (`VCFReader`).
pub struct ShardReader {
    input: Arc<Input>,
    bgzf: bgzf::io::Reader<BufReader<File>>,
    tok: Tokenizer,
    line: Vec<u8>,
    chrom: String,
    start: i64,
    end: i64,
    /// `VCFReader.ranges`: Some while looking for the first record.
    ranges: Option<Vec<(u64, u64)>>,
    range_i: usize,
    need_seek: bool,
    done: bool,
    /// Records dropped for a malformed needed field (`records_dropped`).
    pub records_dropped: u64,
    /// The first few drop messages (for the log).
    pub drop_messages: Vec<String>,
}

enum Line {
    Eof,
    Rec(Box<Record>, bool),
    Dropped(String),
}

impl ShardReader {
    pub fn new(input: Arc<Input>) -> Result<ShardReader> {
        let f =
            File::open(&input.path).with_context(|| format!("open {}", input.path.display()))?;
        Ok(ShardReader {
            bgzf: bgzf::io::Reader::new(BufReader::with_capacity(1 << 17, f)),
            input,
            tok: Tokenizer::default(),
            line: Vec::with_capacity(1024),
            chrom: String::new(),
            start: 0,
            end: 0,
            ranges: None,
            range_i: 0,
            need_seek: false,
            done: true,
            records_dropped: 0,
            drop_messages: Vec::new(),
        })
    }

    /// `__shard__((chrom, start, end))` + `iter()`: 0-based start,
    /// exclusive end.
    pub fn start_shard(&mut self, chrom: &str, start: i64, end: i64) {
        self.chrom = chrom.to_string();
        self.start = start;
        self.end = end;
        self.ranges = Some(self.input.index.query(chrom, start, end));
        self.range_i = 0;
        self.need_seek = true;
        self.done = false;
    }

    fn tell(&self) -> u64 {
        u64::from(self.bgzf.virtual_position())
    }

    /// `VCF.__next__`: the next non-`#` line, parsed.
    fn read_line(&mut self) -> Result<Line> {
        loop {
            self.line.clear();
            let n = self
                .bgzf
                .read_until(b'\n', &mut self.line)
                .with_context(|| format!("{}: read", self.input.path.display()))?;
            if n == 0 {
                return Ok(Line::Eof);
            }
            if self.line[0] != b'#' {
                break;
            }
        }
        match self.tok.parse(&self.line, &self.input.header) {
            Ok((chrom, rec)) => {
                let same = chrom == self.chrom.as_bytes();
                Ok(Line::Rec(Box::new(rec), same))
            }
            Err(ParseError::Malformed(m)) => Ok(Line::Dropped(m)),
            Err(ParseError::Fatal(m)) => bail!("{}: {m}", self.input.path.display()),
        }
    }

    fn note_drop(&mut self, m: String) {
        self.records_dropped += 1;
        eprintln!(
            "WARNING: skipping malformed VCF record: {m} ({})",
            self.input.path.display()
        );
        if self.drop_messages.len() < 10 {
            self.drop_messages.push(m);
        }
    }

    /// The next record of the shard, or None at its end. A malformed
    /// record is dropped and counted (grouper's `qfill`), and the read
    /// continues.
    pub fn next_record(&mut self) -> Result<Option<Record>> {
        if self.done {
            return Ok(None);
        }
        if self.ranges.is_some() {
            return self.first();
        }
        loop {
            match self.read_line()? {
                Line::Eof => {
                    self.done = true;
                    return Ok(None);
                }
                Line::Dropped(m) => self.note_drop(m),
                Line::Rec(v, same) => {
                    if !same || v.pos >= self.end {
                        self.done = true;
                        return Ok(None);
                    }
                    return Ok(Some(*v));
                }
            }
        }
    }

    /// `VCFReader.first`.
    fn first(&mut self) -> Result<Option<Record>> {
        let ranges = self.ranges.take().unwrap_or_default();
        while self.range_i < ranges.len() {
            let r = ranges[self.range_i];
            if self.need_seek {
                self.bgzf
                    .seek(bgzf::VirtualPosition::from(r.0))
                    .with_context(|| format!("{}: seek", self.input.path.display()))?;
                self.need_seek = false;
            }
            loop {
                match self.read_line()? {
                    Line::Eof => {
                        self.done = true;
                        return Ok(None);
                    }
                    Line::Dropped(m) => {
                        // ranges[i] = (tell, r[1]) or i += 1; the caller
                        // (qfill) calls again: keep going here.
                        self.note_drop(m);
                        if self.tell() >= r.1 {
                            self.range_i += 1;
                            self.need_seek = true;
                            break;
                        }
                    }
                    Line::Rec(v, same) => {
                        if !same || v.pos >= self.end {
                            self.done = true;
                            return Ok(None);
                        }
                        if v.end <= self.start {
                            if self.tell() >= r.1 {
                                self.range_i += 1;
                                self.need_seek = true;
                                break;
                            }
                            continue;
                        }
                        return Ok(Some(*v));
                    }
                }
            }
        }
        self.done = true;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_at_block_granularity() {
        // Same block (>> 16 equal) or overlapping: merged; a later block:
        // separate.
        let r = merge(
            vec![
                (5 << 16, 6 << 16),
                (6 << 16 | 10, 7 << 16),
                (9 << 16, 10 << 16),
            ],
            16,
        );
        assert_eq!(r, vec![(5 << 16, 7 << 16), (9 << 16, 10 << 16)]);
        assert!(merge(vec![], 16).is_empty());
    }
}
