// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `grouper`: the heap merge of the input streams by position
//! (originally a 1:1 port of the Python ingest worker's `grouper`).

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::rc::Rc;

use anyhow::Result;

/// What grouper reads of a record.
pub trait GroupRec {
    fn pos(&self) -> i64;
    fn end(&self) -> i64;
    fn alt(&self) -> &[String];
}

impl GroupRec for crate::record::Record {
    fn pos(&self) -> i64 {
        self.pos
    }
    fn end(&self) -> i64 {
        self.end
    }
    fn alt(&self) -> &[String] {
        &self.alt
    }
}

/// A record stream (a `VCFReader` with qfill's drop handling inside).
pub trait Source {
    type Rec: GroupRec;
    fn next_rec(&mut self) -> Result<Option<Self::Rec>>;
}

/// A heap entry `(pos, end, k, j, v, i)`; `live` is "i is not None".
struct Entry<R> {
    pos: i64,
    end: i64,
    k: usize,
    j: u64,
    v: Rc<R>,
    live: bool,
}

impl<R> Entry<R> {
    fn key(&self) -> (i64, i64, usize, u64) {
        (self.pos, self.end, self.k, self.j)
    }
}

impl<R> PartialEq for Entry<R> {
    fn eq(&self, o: &Self) -> bool {
        self.key() == o.key()
    }
}
impl<R> Eq for Entry<R> {}
impl<R> PartialOrd for Entry<R> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<R> Ord for Entry<R> {
    fn cmp(&self, o: &Self) -> Ordering {
        // (pos, end, k, j) is unique among the live entries (Python would
        // compare the records next, and raise).
        self.key().cmp(&o.key())
    }
}

fn sorted_alt(a: &[String]) -> Vec<&String> {
    let mut v: Vec<&String> = a.iter().collect();
    v.sort();
    v
}

/// The grouper state: call `next_group` for each yield; `grp` and `ovl`
/// then hold the group (the Python generator's `(pos, grp, ovl)`).
pub struct Grouper<S: Source> {
    sources: Vec<Option<S>>,
    /// The stream still has an iterator (Python: `i` is not None).
    alive: Vec<bool>,
    q: BinaryHeap<Reverse<Entry<S::Rec>>>,
    pub grp: Vec<Option<Rc<S::Rec>>>,
    /// `(k, v, j)` of the records that lost the same-position rule.
    pub ovl: Vec<(usize, Rc<S::Rec>, u64)>,
}

impl<S: Source> Grouper<S> {
    /// `grouper(*vcfs)`: `None` streams are skipped.
    pub fn new(sources: Vec<Option<S>>) -> Result<Grouper<S>> {
        let n = sources.len();
        let alive = sources.iter().map(Option::is_some).collect();
        let mut g = Grouper {
            sources,
            alive,
            q: BinaryHeap::new(),
            grp: vec![None; n],
            ovl: Vec::new(),
        };
        for k in 0..n {
            g.qfill(k, 0)?;
        }
        Ok(g)
    }

    /// The streams back (exhausted ones included), for the next shard.
    pub fn into_sources(self) -> Vec<Option<S>> {
        self.sources
    }

    fn qfill(&mut self, k: usize, j: u64) -> Result<()> {
        if !self.alive[k] {
            return Ok(());
        }
        let Some(src) = self.sources[k].as_mut() else {
            return Ok(());
        };
        match src.next_rec()? {
            Some(v) => self.q.push(Reverse(Entry {
                pos: v.pos(),
                end: v.end(),
                k,
                j,
                v: Rc::new(v),
                live: true,
            })),
            None => self.alive[k] = false,
        }
        Ok(())
    }

    /// The next yield's position, or None when the streams are done.
    pub fn next_group(&mut self) -> Result<Option<i64>> {
        // After the previous yield: re-queue the losers, with no iterator.
        for (k, v, j) in std::mem::take(&mut self.ovl) {
            self.q.push(Reverse(Entry {
                pos: v.pos(),
                end: v.end(),
                k,
                j,
                v,
                live: false,
            }));
        }
        let Some(Reverse(top)) = self.q.pop() else {
            return Ok(None);
        };
        let pos = top.pos;
        for slot in self.grp.iter_mut() {
            if slot.as_ref().is_some_and(|r| r.end() <= pos) {
                *slot = None;
            }
        }
        self.grp[top.k] = Some(top.v);
        if top.live {
            self.qfill(top.k, top.j + 1)?;
        }
        while self.q.peek().is_some_and(|Reverse(e)| e.pos == pos) {
            let Reverse(e) = self.q.pop().expect("peeked");
            let k = e.k;
            let mut loser: Option<Rc<S::Rec>> = Some(e.v);
            match &self.grp[k] {
                Some(cur) if cur.pos() == pos => {
                    let v = loser.as_ref().expect("set");
                    if e.end < cur.end()
                        || (e.end == cur.end() && sorted_alt(v.alt()) < sorted_alt(cur.alt()))
                    {
                        let won = loser.take().expect("set");
                        loser = self.grp[k].replace(won);
                    }
                }
                _ => {
                    // None or carried forward: overwrite, no re-queue.
                    self.grp[k] = loser.take();
                }
            }
            if let Some(v) = loser {
                self.ovl.push((k, v, e.j));
            }
            if e.live {
                self.qfill(k, e.j + 1)?;
            }
        }
        Ok(Some(pos))
    }
}

#[cfg(test)]
mod tests {
    //! The five tests of `tests/unit/test_grouper.py`, same streams.
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Rec {
        pos: i64,
        end: i64,
        alt: Vec<String>,
    }

    fn rec(pos: i64, end: i64) -> Rec {
        rec_alt(pos, end, &["A"])
    }

    fn rec_alt(pos: i64, end: i64, alt: &[&str]) -> Rec {
        Rec {
            pos,
            end,
            alt: alt.iter().map(|s| s.to_string()).collect(),
        }
    }

    impl GroupRec for Rec {
        fn pos(&self) -> i64 {
            self.pos
        }
        fn end(&self) -> i64 {
            self.end
        }
        fn alt(&self) -> &[String] {
            &self.alt
        }
    }

    struct VecSource(std::vec::IntoIter<Rec>);

    impl Source for VecSource {
        type Rec = Rec;
        fn next_rec(&mut self) -> Result<Option<Rec>> {
            Ok(self.0.next())
        }
    }

    type Group = (i64, Vec<Option<Rc<Rec>>>, Vec<(usize, Rc<Rec>, u64)>);

    fn collect(streams: Vec<Option<Vec<Rec>>>) -> Vec<Group> {
        let srcs = streams
            .into_iter()
            .map(|s| s.map(|v| VecSource(v.into_iter())))
            .collect();
        let mut g = Grouper::new(srcs).unwrap();
        let mut out = Vec::new();
        while let Some(pos) = g.next_group().unwrap() {
            out.push((pos, g.grp.clone(), g.ovl.clone()));
        }
        out
    }

    fn positions(gs: &[Group]) -> Vec<i64> {
        gs.iter().map(|g| g.0).collect()
    }

    #[test]
    fn single_stream_passthrough() {
        let out = collect(vec![Some(vec![rec(10, 11), rec(20, 21)])]);
        assert_eq!(positions(&out), vec![10, 20]);
        assert_eq!(out[0].1[0].as_ref().unwrap().pos, 10);
        assert_eq!(out[1].1[0].as_ref().unwrap().pos, 20);
        assert_eq!(out[1].1[0].as_ref().unwrap().end, 21);
    }

    #[test]
    fn two_streams_interleaved() {
        let out = collect(vec![Some(vec![rec(10, 11)]), Some(vec![rec(15, 16)])]);
        assert_eq!(positions(&out), vec![10, 15]);
        assert!(out[0].1[0].is_some() && out[0].1[1].is_none());
        assert!(out[1].1[0].is_none() && out[1].1[1].is_some());
    }

    #[test]
    fn overlap_carry_forward() {
        let out = collect(vec![
            Some(vec![rec(10, 30)]),
            Some(vec![rec(15, 16), rec(20, 21)]),
        ]);
        assert_eq!(positions(&out), vec![10, 15, 20]);
        let block = out[0].1[0].clone().unwrap();
        for g in &out {
            assert!(Rc::ptr_eq(g.1[0].as_ref().unwrap(), &block));
        }
        assert_eq!(out[1].1[1].as_ref().unwrap().pos, 15);
        assert_eq!(out[2].1[1].as_ref().unwrap().pos, 20);
    }

    #[test]
    fn same_position_prefers_shorter_span() {
        let out = collect(vec![
            Some(vec![rec_alt(10, 50, &["AAAA"])]),
            Some(vec![rec_alt(10, 12, &["A"])]),
        ]);
        assert_eq!(out[0].0, 10);
        let actives: Vec<&Rc<Rec>> = out[0].1.iter().flatten().collect();
        assert!(actives.iter().any(|r| ***r == rec_alt(10, 12, &["A"])));
    }

    #[test]
    fn empty_streams() {
        assert!(collect(vec![Some(vec![]), Some(vec![])]).is_empty());
        let out = collect(vec![None, Some(vec![rec(5, 6)])]);
        assert_eq!(out[0].0, 5);
        assert_eq!(out[0].1.len(), 2);
    }

    #[test]
    fn same_stream_same_position_ovl_and_requeue() {
        // Within one stream: the shorter record wins the slot, the longer
        // goes to ovl and comes back, alone, at the same position.
        let out = collect(vec![Some(vec![
            rec_alt(10, 20, &["T"]),
            rec_alt(10, 12, &["G"]),
            rec(30, 31),
        ])]);
        assert_eq!(positions(&out), vec![10, 10, 30]);
        assert_eq!(out[0].1[0].as_ref().unwrap().end, 12);
        assert_eq!(out[0].2.len(), 1);
        assert_eq!(out[0].2[0].1.end, 20);
        assert_eq!(out[0].2[0].2, 1); // the popped (winning) entry's j
        assert_eq!(out[1].1[0].as_ref().unwrap().end, 20);
        assert!(out[1].2.is_empty());
        // Equal spans: ties by sorted(alt).
        let out = collect(vec![Some(vec![
            rec_alt(10, 12, &["T", "C"]),
            rec_alt(10, 12, &["G", "A"]),
        ])]);
        assert_eq!(out[0].1[0].as_ref().unwrap().alt, vec!["G", "A"]);
        assert_eq!(out[1].1[0].as_ref().unwrap().alt, vec!["T", "C"]);
    }
}
