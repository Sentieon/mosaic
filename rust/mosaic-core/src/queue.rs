// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The engines' thread model: a queue of items (the ingest worker's shards, the joint
//! caller's units) taken in order from a shared counter by `threads`
//! scoped threads, each with its own state (its readers), and the results
//! handed to the calling thread in completion order over a channel. Std
//! only.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use anyhow::{Result, anyhow};

/// Run `work(&mut state, i)` for every `i` in `0..n` on `threads` threads
/// (at least 1, at most `n`). Each thread takes the next index from a
/// shared counter, so the items start in index order, and keeps its own
/// `state` (`C::default()` at thread start) across its items. `sink(i,
/// result)` runs on the calling thread, once per finished item, in
/// completion order; a panic in `work` reaches it as an error. When the
/// sink returns `Break` no further item is started, the threads finish the
/// item they hold (their results are dropped) and `Break` is returned.
pub fn run_queue<C, T, W, S>(n: usize, threads: usize, work: W, mut sink: S) -> ControlFlow<()>
where
    C: Default,
    T: Send,
    W: Fn(&mut C, usize) -> Result<T> + Sync,
    S: FnMut(usize, Result<T>) -> ControlFlow<()>,
{
    let threads = threads.clamp(1, n.max(1));
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<(usize, Result<T>)>();
    std::thread::scope(|sc| {
        for _ in 0..threads.min(n) {
            let tx = tx.clone();
            let (next, stop, work) = (&next, &stop, &work);
            sc.spawn(move || {
                let mut state = C::default();
                while !stop.load(Ordering::SeqCst) {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        break;
                    }
                    let r = catch_unwind(AssertUnwindSafe(|| work(&mut state, i)))
                        .unwrap_or_else(|p| Err(anyhow!("panic: {}", panic_message(&p))));
                    let failed = r.is_err();
                    // The receiver is gone only after a Break.
                    if tx.send((i, r)).is_err() || failed {
                        // A thread whose item failed takes no further item
                        // (its state may be inconsistent).
                        break;
                    }
                }
            });
        }
        drop(tx);
        for (i, r) in rx.iter() {
            if sink(i, r).is_break() {
                stop.store(true, Ordering::SeqCst);
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    })
}

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "(non-string payload)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn every_item_exactly_once() {
        for threads in [1, 2, 3, 4, 8, 64] {
            for n in [0, 1, 2, 7, 50] {
                let mut seen: HashMap<usize, usize> = HashMap::new();
                let flow = run_queue(
                    n,
                    threads,
                    |calls: &mut usize, i| {
                        *calls += 1;
                        // Uneven item times, so completion order differs.
                        std::thread::sleep(Duration::from_micros(((i * 7919) % 13) as u64 * 50));
                        Ok(i * 10)
                    },
                    |i, r| {
                        assert_eq!(r.unwrap(), i * 10);
                        *seen.entry(i).or_default() += 1;
                        ControlFlow::Continue(())
                    },
                );
                assert!(flow.is_continue());
                assert_eq!(seen.len(), n, "threads {threads} n {n}");
                assert!(seen.values().all(|&c| c == 1));
            }
        }
    }

    #[test]
    fn each_thread_takes_rising_items_with_its_own_state() {
        let starts = Mutex::new(Vec::new());
        let threads_seen = Mutex::new(HashMap::<std::thread::ThreadId, Vec<usize>>::new());
        let flow = run_queue(
            40,
            4,
            |mine: &mut Vec<usize>, i| {
                starts.lock().unwrap().push(i);
                mine.push(i);
                // The state is this thread's own: its items so far, rising.
                assert!(mine.windows(2).all(|w| w[0] < w[1]));
                threads_seen
                    .lock()
                    .unwrap()
                    .insert(std::thread::current().id(), mine.clone());
                std::thread::sleep(Duration::from_micros(200));
                Ok(())
            },
            |_, r| {
                r.unwrap();
                ControlFlow::Continue(())
            },
        );
        assert!(flow.is_continue());
        // Every item started once (the order of the pushes across threads
        // is not the order of the counter: not asserted).
        let mut starts = starts.into_inner().unwrap();
        starts.sort_unstable();
        assert_eq!(starts, (0..40).collect::<Vec<_>>());
        let per = threads_seen.into_inner().unwrap();
        assert!(per.len() <= 4);
        let mut all: Vec<usize> = per.into_values().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn an_error_or_panic_reaches_the_sink_and_break_stops() {
        for panic in [false, true] {
            let started = AtomicUsize::new(0);
            let mut errs = Vec::new();
            let flow = run_queue(
                1000,
                3,
                |_: &mut (), i| {
                    started.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_micros(100));
                    if i == 5 {
                        if panic {
                            panic!("boom {i}");
                        }
                        anyhow::bail!("bad {i}");
                    }
                    Ok(())
                },
                |i, r| match r {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(e) => {
                        errs.push((i, format!("{e}")));
                        ControlFlow::Break(())
                    }
                },
            );
            assert!(flow.is_break());
            let want = if panic { "panic: boom 5" } else { "bad 5" };
            assert_eq!(errs, vec![(5, want.to_string())]);
            // Stopped well before the end of the queue.
            assert!(started.load(Ordering::SeqCst) < 100);
        }
    }
}
