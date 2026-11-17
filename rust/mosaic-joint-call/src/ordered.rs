// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The ordered output with bounded memory: the units run on several
//! threads and their output is laid down in unit order. The **head** unit
//! (`next`, the first unit not fully written) writes its blocks to the
//! output file as it produces them; every other unit appends them to a
//! memory buffer of its own and, once that buffer exceeds the bound
//! (`--unit-buffer-bytes`), moves it to a spill file of its own and appends
//! there from then on. When the head ends, its thread advances `next` over
//! the following units that already finished, copying each one's buffer or
//! spill file to the output, and makes the first unit still running (or not
//! yet started) the head, flushing what that unit buffered so far.
//!
//! **Locking.** Each unit has a slot (its buffer or spill file, and two
//! flags: `head`, `done`) behind a mutex of its own, which only that unit's
//! thread takes, except at the one moment the advancing thread promotes it
//! (or copies it once it is done). The output file is behind its own mutex,
//! taken by the head's writes and the advancing thread's copies, always
//! after a slot's (the one lock order: slot, then output). Nothing is
//! compressed under a lock. The promotion and a unit's end are atomic under
//! the unit's slot lock: either the unit ends first (it is not the head, so
//! it is left `done` and the advancer copies it) or the advancer promotes it
//! first (the unit then sees `head` at its end and advances itself), so no
//! finished unit is missed.
//!
//! **The gate** (J3's `UnitGate`, unchanged: unit `i` starts only while
//! `i < g + 2 × threads`) is advanced by this writer to a unit `g` only once
//! `g` holds no buffer (it is the head, flushed, or copied), so the units
//! that may buffer lie in `(g, g + 2 × threads)`: at most `2 × threads − 1`
//! buffers, each at most the bound plus one chunk's blocks, and at most as
//! many spill files. At one thread every unit is promoted before it starts:
//! nothing is buffered or spilled.

use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::run::UnitGate;
use crate::writer::Output;

/// The default bound on a non-head unit's memory buffer (64 MiB).
pub const DEFAULT_UNIT_BUFFER_BYTES: u64 = 64 << 20;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A non-head unit's spill file.
struct Spill {
    path: PathBuf,
    file: File,
    bytes: u64,
}

/// A unit's output not yet written, and its state.
#[derive(Default)]
struct Slot {
    /// The unit writes to the output itself.
    head: bool,
    /// The unit ended.
    done: bool,
    /// Its blocks (or text) buffered in memory.
    buf: Vec<u8>,
    /// Its spill file, once the buffer exceeded the bound.
    spill: Option<Spill>,
    /// The uncompressed bytes of `buf` / `spill`.
    text_bytes: u64,
}

/// What the writer measured.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WriterStats {
    /// The most finished units awaiting the head at once.
    pub peak_units: usize,
    /// The peak of the bytes buffered in memory over all units.
    pub peak_bytes: u64,
    /// Units that moved their buffer to a spill file.
    pub spilled_units: u64,
    /// Bytes written to spill files.
    pub spilled_bytes: u64,
}

/// The output file, written in unit order (see the module comment).
pub struct OrderedWriter {
    out: Mutex<Option<Output>>,
    slots: Vec<Mutex<Slot>>,
    gate: UnitGate,
    bgzf: bool,
    bound: u64,
    spill_dir: PathBuf,
    /// Whether this run created `spill_dir` (and has not removed it).
    spill_created: Mutex<bool>,
    /// The first unit not fully written (the gate's value).
    next: AtomicUsize,
    buffered: AtomicU64,
    peak_buffered: AtomicU64,
    held: AtomicUsize,
    peak_held: AtomicUsize,
    spilled_units: AtomicU64,
    spilled_bytes: AtomicU64,
    write_ns: AtomicU64,
    copy_ns: AtomicU64,
}

impl OrderedWriter {
    /// A writer of `units` units onto `out` (its header already written),
    /// gated at `gate_bound` units past the first unwritten one, spilling a
    /// non-head unit's buffer once it exceeds `bound` bytes into
    /// `spill_dir` (created at the first spill).
    pub fn new(
        out: Output,
        units: usize,
        gate_bound: usize,
        bound: u64,
        spill_dir: PathBuf,
    ) -> OrderedWriter {
        let bgzf = out.is_bgzf();
        let slots: Vec<Mutex<Slot>> = (0..units)
            .map(|i| {
                Mutex::new(Slot {
                    head: i == 0,
                    ..Slot::default()
                })
            })
            .collect();
        OrderedWriter {
            out: Mutex::new(Some(out)),
            slots,
            gate: UnitGate::new(gate_bound),
            bgzf,
            bound,
            spill_dir,
            spill_created: Mutex::new(false),
            next: AtomicUsize::new(0),
            buffered: AtomicU64::new(0),
            peak_buffered: AtomicU64::new(0),
            held: AtomicUsize::new(0),
            peak_held: AtomicUsize::new(0),
            spilled_units: AtomicU64::new(0),
            spilled_bytes: AtomicU64::new(0),
            write_ns: AtomicU64::new(0),
            copy_ns: AtomicU64::new(0),
        }
    }

    /// Wait until unit `i` may start (the gate); an error once closed.
    pub fn admit(&self, i: usize) -> Result<()> {
        self.gate.admit(i)
    }

    /// Stop the run: no unit starts, nothing more is copied.
    pub fn close(&self) {
        self.gate.close();
    }

    /// The first unit not fully written.
    pub fn next(&self) -> usize {
        self.next.load(Ordering::SeqCst)
    }

    pub fn stats(&self) -> WriterStats {
        WriterStats {
            peak_units: self.peak_held.load(Ordering::SeqCst),
            peak_bytes: self.peak_buffered.load(Ordering::SeqCst),
            spilled_units: self.spilled_units.load(Ordering::SeqCst),
            spilled_bytes: self.spilled_bytes.load(Ordering::SeqCst),
        }
    }

    /// Seconds spent writing the output (the heads' blocks and the copies)
    /// and, of those, copying held buffers and spill files.
    pub fn seconds(&self) -> (f64, f64) {
        let s = |a: &AtomicU64| a.load(Ordering::SeqCst) as f64 / 1e9;
        (s(&self.write_ns), s(&self.copy_ns))
    }

    fn add_ns(a: &AtomicU64, t: Instant) {
        a.fetch_add(t.elapsed().as_nanos() as u64, Ordering::SeqCst);
    }

    /// Append one chunk of unit `i`'s text: compressed here (a `.gz`
    /// output), outside any lock; then written to the output when the unit
    /// is the head, else buffered or spilled.
    pub fn push(&self, i: usize, text: &[u8]) -> Result<()> {
        let blocks: Cow<'_, [u8]> = if self.bgzf {
            Cow::Owned(Output::compress(text)?)
        } else {
            Cow::Borrowed(text)
        };
        let n = blocks.len() as u64;
        let mut s = lock(&self.slots[i]);
        if s.head {
            let t = Instant::now();
            let r = lock(&self.out)
                .as_mut()
                .expect("open while units run")
                .write_raw(&blocks, text.len() as u64);
            Self::add_ns(&self.write_ns, t);
            return r.context("write the output");
        }
        s.text_bytes += text.len() as u64;
        if let Some(sp) = s.spill.as_mut() {
            sp.file
                .write_all(&blocks)
                .with_context(|| format!("{}: write the spill file", sp.path.display()))?;
            sp.bytes += n;
            self.spilled_bytes.fetch_add(n, Ordering::SeqCst);
            return Ok(());
        }
        s.buf.extend_from_slice(&blocks);
        let now = self.buffered.fetch_add(n, Ordering::SeqCst) + n;
        self.peak_buffered.fetch_max(now, Ordering::SeqCst);
        if s.buf.len() as u64 > self.bound {
            self.spill(i, &mut s)?;
        }
        Ok(())
    }

    /// Move unit `i`'s buffer to its spill file (created here).
    fn spill(&self, i: usize, s: &mut Slot) -> Result<()> {
        self.ensure_spill_dir()?;
        let path = self.spill_dir.join(format!("unit-{i}.spill"));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("{}: create the spill file", path.display()))?;
        let buf = std::mem::take(&mut s.buf);
        let n = buf.len() as u64;
        file.write_all(&buf)
            .with_context(|| format!("{}: write the spill file", path.display()))?;
        drop(buf);
        self.buffered.fetch_sub(n, Ordering::SeqCst);
        self.spilled_units.fetch_add(1, Ordering::SeqCst);
        self.spilled_bytes.fetch_add(n, Ordering::SeqCst);
        s.spill = Some(Spill {
            path,
            file,
            bytes: n,
        });
        Ok(())
    }

    fn ensure_spill_dir(&self) -> Result<()> {
        let mut created = lock(&self.spill_created);
        // Once the run is stopped its spill directory is removed (or
        // about to be): a unit still running must not bring it back.
        // `remove_spill_dir` takes the same lock, so after it no later
        // spill creates the directory again.
        if self.gate.is_closed() {
            bail!("the run stopped");
        }
        if !*created {
            std::fs::create_dir_all(&self.spill_dir).with_context(|| {
                format!("{}: create the spill directory", self.spill_dir.display())
            })?;
            *created = true;
        }
        Ok(())
    }

    /// Remove the spill directory if this run created it (best effort; a
    /// no-op when nothing spilled or it is already removed).
    pub fn remove_spill_dir(&self) {
        let mut created = lock(&self.spill_created);
        if !*created {
            return;
        }
        // Another thread may still be writing a spill file into it (the
        // exit path): retry a few times.
        for _ in 0..3 {
            match std::fs::remove_dir_all(&self.spill_dir) {
                Ok(()) => break,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        *created = false;
    }

    /// The spill directory.
    pub fn spill_dir(&self) -> &Path {
        &self.spill_dir
    }

    /// Unit `i` ended: if it is the head, advance over the finished units
    /// after it; else it is held for that advance.
    pub fn end_unit(&self, i: usize) -> Result<()> {
        let head = {
            let mut s = lock(&self.slots[i]);
            s.done = true;
            if !s.head {
                let now = self.held.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak_held.fetch_max(now, Ordering::SeqCst);
            }
            s.head
        };
        if head { self.advance(i + 1) } else { Ok(()) }
    }

    /// Copy every finished unit from `cur` on, in order, then promote the
    /// first unit not finished (flushing what it buffered).
    fn advance(&self, mut cur: usize) -> Result<()> {
        loop {
            if self.gate.is_closed() {
                return Ok(());
            }
            if cur == self.slots.len() {
                self.next.store(cur, Ordering::SeqCst);
                self.gate.advance(cur);
                return Ok(());
            }
            let mut s = lock(&self.slots[cur]);
            self.drain(&mut s)?;
            // `cur` holds no buffer now (copied, or the head).
            self.next.store(cur, Ordering::SeqCst);
            self.gate.advance(cur);
            if s.done {
                self.held.fetch_sub(1, Ordering::SeqCst);
                drop(s);
                cur += 1;
                continue;
            }
            s.head = true;
            return Ok(());
        }
    }

    /// Write a slot's buffer or spill file to the output and free it.
    fn drain(&self, s: &mut Slot) -> Result<()> {
        let text_bytes = std::mem::take(&mut s.text_bytes);
        if s.spill.is_none() && s.buf.is_empty() && text_bytes == 0 {
            return Ok(());
        }
        let t = Instant::now();
        let mut out = lock(&self.out);
        let out = out.as_mut().expect("open while units run");
        if let Some(mut sp) = s.spill.take() {
            sp.file
                .seek(SeekFrom::Start(0))
                .with_context(|| format!("{}: rewind the spill file", sp.path.display()))?;
            out.copy_from(&mut sp.file, sp.bytes, text_bytes)
                .with_context(|| format!("{}: copy the spill file", sp.path.display()))?;
            drop(sp.file);
            std::fs::remove_file(&sp.path)
                .with_context(|| format!("{}: remove the spill file", sp.path.display()))?;
        } else {
            let buf = std::mem::take(&mut s.buf);
            out.write_raw(&buf, text_bytes)
                .context("write the output")?;
            self.buffered.fetch_sub(buf.len() as u64, Ordering::SeqCst);
        }
        Self::add_ns(&self.write_ns, t);
        Self::add_ns(&self.copy_ns, t);
        Ok(())
    }

    /// The output, once every unit is written (the caller appends the EOF
    /// block and flushes).
    pub fn take_output(&self) -> Option<Output> {
        lock(&self.out).take()
    }
}

impl Drop for OrderedWriter {
    fn drop(&mut self) {
        self.remove_spill_dir();
    }
}

#[cfg(test)]
mod tests {
    //! The writer alone, without a lake: units of chunks of varied sizes
    //! pushed from several threads, finishing out of order.
    use std::io::Read;
    use std::ops::ControlFlow;
    use std::time::Duration;

    use super::*;
    use crate::testlake::tmpdir;

    /// Unit `i`'s chunks: a deterministic mix of sizes (some empty units).
    fn chunks(i: usize) -> Vec<String> {
        let k = (i * 7 + 3) % 5;
        (0..k)
            .map(|j| {
                let len = ((i * 131 + j * 977) % 50_000) + 1;
                let c = (b'a' + ((i + j) % 26) as u8) as char;
                let mut s = format!("u{i}c{j}:");
                s.extend(std::iter::repeat_n(c, len));
                s.push('\n');
                s
            })
            .collect()
    }

    /// Run `units` units on `threads` threads through the writer, unit `i`
    /// sleeping a little in reverse order so later units tend to finish
    /// first (`slow_head`: unit 0 sleeps 300 ms, so the later units are
    /// held); returns the stats and whether the spill directory existed
    /// before the end.
    fn write_all(
        path: &Path,
        spill: &Path,
        units: usize,
        threads: usize,
        bound: u64,
        slow_head: bool,
    ) -> (WriterStats, bool) {
        let mut out = Output::create(path.to_str().unwrap()).unwrap();
        out.write_text(b"##header\n").unwrap();
        let w = OrderedWriter::new(
            out,
            units,
            crate::run::HELD_PER_THREAD * threads,
            bound,
            spill.to_path_buf(),
        );
        let saw_dir = std::sync::atomic::AtomicBool::new(false);
        let flow = mosaic_core::queue::run_queue(
            units,
            threads,
            |_: &mut (), i| {
                w.admit(i)?;
                if slow_head && i == 0 {
                    std::thread::sleep(Duration::from_millis(300));
                } else {
                    std::thread::sleep(Duration::from_micros(((units - i) % 7) as u64 * 300));
                }
                for c in chunks(i) {
                    w.push(i, c.as_bytes())?;
                    if spill.exists() {
                        saw_dir.store(true, Ordering::SeqCst);
                    }
                }
                w.end_unit(i)
            },
            |_, r| {
                r.unwrap();
                ControlFlow::Continue(())
            },
        );
        assert!(flow.is_continue());
        assert_eq!(w.next(), units);
        w.take_output().unwrap().finish().unwrap();
        let stats = w.stats();
        w.remove_spill_dir();
        (stats, saw_dir.load(Ordering::SeqCst))
    }

    fn expected(units: usize) -> String {
        let mut s = "##header\n".to_string();
        for i in 0..units {
            for c in chunks(i) {
                s.push_str(&c);
            }
        }
        s
    }

    fn read_text(path: &Path) -> String {
        let bytes = std::fs::read(path).unwrap();
        if path.extension().is_some_and(|e| e == "gz") {
            let mut s = String::new();
            noodles_bgzf::io::Reader::new(&bytes[..])
                .read_to_string(&mut s)
                .unwrap();
            s
        } else {
            String::from_utf8(bytes).unwrap()
        }
    }

    #[test]
    fn units_in_order_with_and_without_spilling() {
        let dir = tmpdir("ordered");
        let units = 40;
        let want = expected(units);
        let largest_chunk = (0..units)
            .flat_map(chunks)
            .map(|c| c.len() as u64 + 1024)
            .max()
            .unwrap();
        for name in ["out.vcf", "out.vcf.gz"] {
            let path = dir.join(name);
            let mut first: Option<Vec<u8>> = None;
            for threads in [1, 2, 4] {
                for (bound, slow) in [(1 << 30, false), (1 << 30, true), (1, false), (1, true)] {
                    let spill = dir.join(format!("spill-{name}-{threads}-{bound}-{slow}"));
                    let (st, saw) = write_all(&path, &spill, units, threads, bound, slow);
                    let tag = format!("{name} threads {threads} bound {bound} slow {slow}");
                    assert_eq!(read_text(&path), want, "{tag}");
                    // The bytes do not depend on the threads or the bound.
                    let bytes = std::fs::read(&path).unwrap();
                    match &first {
                        None => first = Some(bytes),
                        Some(f) => assert!(*f == bytes, "{tag}"),
                    }
                    assert!(!spill.exists(), "{tag}: the spill directory is left");
                    let held_bound = (crate::run::HELD_PER_THREAD * threads - 1) as u64;
                    assert!(
                        st.peak_bytes <= held_bound * (bound.min(1 << 20) + largest_chunk),
                        "{tag}: {st:?}"
                    );
                    if threads == 1 {
                        // Every unit is the head from its start.
                        assert_eq!(
                            st,
                            WriterStats::default(),
                            "{tag}: one thread buffers nothing"
                        );
                        assert!(!saw, "{tag}");
                    }
                    if bound > 1 {
                        // Nothing spills under a large bound, and the
                        // directory is never created.
                        assert_eq!((st.spilled_units, st.spilled_bytes), (0, 0), "{tag}");
                        assert!(!saw, "{tag}");
                    }
                    if slow && threads > 1 {
                        // The units after the slow head finish first: held.
                        assert_eq!(st.peak_units as u64, held_bound, "{tag}");
                        if bound == 1 {
                            assert!(st.spilled_units > 0 && saw, "{tag}: {st:?}");
                        } else {
                            assert!(st.peak_bytes > 0, "{tag}");
                        }
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_non_head_chunk_spills_at_bound_one() {
        // Two units on two threads: unit 0 waits until unit 1 pushed all
        // its chunks and ended (held), so every chunk of unit 1 went to its
        // spill file; unit 0 then copies it.
        let dir = tmpdir("ordered-one");
        let path = dir.join("out.vcf");
        let spill = dir.join("spill");
        let mut out = Output::create(path.to_str().unwrap()).unwrap();
        out.write_text(b"#h\n").unwrap();
        let w = OrderedWriter::new(out, 2, 4, 1, spill.clone());
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let rx = Mutex::new(rx);
        std::thread::scope(|sc| {
            sc.spawn(|| {
                w.push(1, b"b1\n").unwrap();
                assert_eq!(w.stats().spilled_units, 1);
                assert_eq!(w.stats().spilled_bytes, 3);
                w.push(1, b"b22\n").unwrap();
                assert_eq!(w.stats().spilled_bytes, 7);
                // The buffer moved to the spill file: nothing in memory.
                assert_eq!(w.buffered.load(Ordering::SeqCst), 0);
                assert!(spill.join("unit-1.spill").exists());
                w.end_unit(1).unwrap();
                tx.send(()).unwrap();
            });
            sc.spawn(|| {
                rx.lock().unwrap().recv().unwrap();
                w.push(0, b"a1\n").unwrap();
                w.end_unit(0).unwrap();
            });
        });
        assert_eq!(w.next(), 2);
        assert!(!spill.join("unit-1.spill").exists());
        w.take_output().unwrap().finish().unwrap();
        let st = w.stats();
        assert_eq!((st.peak_units, st.peak_bytes), (1, 3));
        drop(w);
        assert!(!spill.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "#h\na1\nb1\nb22\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_unit_promoted_while_running_flushes_its_buffer_first() {
        // Unit 1 buffers a chunk while unit 0 runs; unit 0 ends and
        // promotes unit 1, flushing its buffer; unit 1's next chunk goes
        // straight to the output.
        let dir = tmpdir("ordered-promote");
        let path = dir.join("out.vcf");
        let spill = dir.join("spill");
        let out = Output::create(path.to_str().unwrap()).unwrap();
        let w = OrderedWriter::new(out, 3, 4, 1 << 20, spill.clone());
        w.push(1, b"b1\n").unwrap();
        w.push(0, b"a1\n").unwrap();
        w.push(2, b"c1\n").unwrap();
        w.end_unit(2).unwrap();
        assert_eq!(w.stats().peak_units, 1);
        w.end_unit(0).unwrap();
        // Unit 1 is the head now; unit 2 still held.
        assert_eq!(w.next(), 1);
        assert_eq!(w.buffered.load(Ordering::SeqCst), 3);
        w.push(1, b"b2\n").unwrap();
        w.end_unit(1).unwrap();
        assert_eq!(w.next(), 3);
        assert_eq!(w.buffered.load(Ordering::SeqCst), 0);
        w.take_output().unwrap().finish().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a1\nb1\nb2\nc1\n");
        assert_eq!(w.stats().peak_bytes, 6);
        assert!(!spill.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn nothing_spills_once_the_run_is_stopped() {
        // The exit path: the run is closed and its spill directory
        // removed while units still run. A unit that would spill for
        // the first time gets an error and does not create the
        // directory again; a unit that already has a spill file keeps
        // appending to it (unlinked), also without creating anything.
        let dir = tmpdir("ordered-stopped");
        let spill = dir.join("spill");
        let out = Output::create(dir.join("out.vcf").to_str().unwrap()).unwrap();
        let w = OrderedWriter::new(out, 3, 4, 0, spill.clone());
        w.push(1, b"b1\n").unwrap();
        assert!(spill.join("unit-1.spill").exists());
        w.close();
        w.remove_spill_dir();
        assert!(!spill.exists());
        let e = w.push(2, b"c1\n").unwrap_err();
        assert!(format!("{e:#}").contains("the run stopped"), "{e:#}");
        assert!(!spill.exists());
        w.push(1, b"b2\n").unwrap();
        assert!(!spill.exists());
        drop(w);
        assert!(!spill.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_spill_error_is_the_units_error() {
        // The spill directory cannot be created (its parent is a file).
        let dir = tmpdir("ordered-err");
        let blocker = dir.join("file");
        std::fs::write(&blocker, b"x").unwrap();
        let out = Output::create(dir.join("out.vcf").to_str().unwrap()).unwrap();
        let w = OrderedWriter::new(out, 2, 4, 0, blocker.join("spill"));
        let e = w.push(1, b"b\n").unwrap_err();
        assert!(
            format!("{e:#}").contains("create the spill directory"),
            "{e:#}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
