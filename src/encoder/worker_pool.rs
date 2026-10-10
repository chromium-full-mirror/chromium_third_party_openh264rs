// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

// Copyright (c) 2013, Cisco Systems
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice, this
//   list of conditions and the following disclaimer in the documentation and/or
//   other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#![deny(unsafe_code)]
// The C++ counterpart is `codec/common/inc/WelsThreadPool.h` /
// `codec/common/src/WelsThreadPool.cpp`: threads created once, fed tasks from a
// queue, signalling when a frame's task count reaches zero.

//! # The persistent worker pool
//!
//! [`WorkerPool`] keeps `N` threads alive for the encoder's lifetime and offers
//! `std::thread::scope`'s API — `pool.scope(|s| { s.spawn(..) .. h.join() })` — to
//! the encoder's three slice forks (`EncodeFixedSlicesForked`,
//! `EncodeSizeLimitedSlicesForked`, `UpdateMbMapForked` in
//! `slice_multi_threading.rs`), which would otherwise create and destroy one OS
//! thread per worker per frame.
//!
//! # Shape
//!
//! * `N` worker threads, created by [`WorkerPool::new`], parked when idle.
//! * One shared FIFO of type-erased jobs behind a `Mutex`, with an atomic
//!   `pending` count beside it so an idle worker can poll for work without the
//!   lock.
//! * Per scope: an outstanding-job counter and the owning thread's handle, in an
//!   `Arc` so a worker can finish its bookkeeping after the scope has returned.
//! * Per job: a packet holding the result (`Ok(T)` or the panic payload) and a
//!   `done` flag.
//!
//! The calling thread helps: while it waits in [`JobHandle::join`] or at the end of
//! [`WorkerPool::scope`] it runs any job still in the queue itself, so a frame never
//! waits on a worker's wake-up latency for a job nobody has started, and a pool of
//! zero workers runs everything inline. The pool decides which thread runs a job,
//! never how the slices are grouped into jobs.
//!
//! Parking and waking are `std::thread::park`/`unpark`; every other piece of
//! synchronisation is `std::sync`.
//!
//! # The one `unsafe`, and why it is sound
//!
//! A job borrows the frame's stack (`'scope`), and a persistent thread is
//! `'static`, so the closure has to cross that boundary: it is boxed as
//! `Box<dyn FnOnce() + Send + 'scope>` and its lifetime is erased to `'static`
//! at the one `unsafe` site in this module (`Scope::spawn`). The argument is
//! `std::thread::scope`'s:
//!
//! 1. **The scope does not return, and does not unwind past its own frame, until
//!    every job it spawned has completed.** `ScopeData::running` is incremented
//!    at spawn and decremented by the thread that ran the job; `scope` waits for
//!    it to reach zero before returning `f`'s result, and the wait is
//!    unconditional — `f` runs under `catch_unwind`, so a panic in the scope body
//!    also waits, then resumes unwinding. The wait's `Acquire` load against the
//!    worker's `Release` decrement is the happens-before edge that publishes
//!    everything the job wrote to whatever reads it after the scope; the
//!    reconstruction seam (`rec_view.rs`) relies on that edge.
//! 2. **The decrement is the last thing a job does with anything borrowed.** The
//!    user's closure is consumed by its call, so its captures are dropped when it
//!    returns; the result `T` (which may itself borrow `'scope`) is stored in the
//!    packet and the running thread's reference to the packet is dropped before
//!    the decrement, so if the handle was already gone the `T` is dropped then
//!    too. After the decrement the thread touches only the `Arc<ScopeData>` it
//!    owns, which is heap, and its own queue.
//! 3. **`'scope` cannot shrink.** `Scope` is invariant in `'scope` (the
//!    `PhantomData<&'scope mut &'scope ()>` marker), and `spawn` demands
//!    `F: 'scope`, so a job cannot capture a local that dies before `f` returns.
//!
//! So the erased `'static` is never relied on: the closure is called, and its
//! captures dropped, strictly before the borrows it holds can end. Panics inside
//! a job are caught (`catch_unwind`), stored, and returned from `join` as `Err`,
//! which the forks map to `ENC_RETURN_UNEXPECTED`; a job that panicked and was
//! never joined makes `scope` panic at its end, as `std` does.
//!
//! # Ownership and thread-safety
//!
//! The pool's threads are owned by [`WorkerPool`] and joined in its `Drop`.
//! `WorkerPool` is `Sync` by its fields (atomics, a mutex, thread handles), never
//! by hand, which is what lets `&sWelsEncCtx` — which holds the pool through
//! `pSliceThreading` — cross a spawn. `Scope` and `JobHandle` are `!Sync`/`!Send`:
//! a handle is joined on the thread that owns the scope, because that is the thread
//! a completing job unparks.
//!
//! # Invariants the encoder relies on
//!
//! * The pool is built by `RequestMtResource`, with `uiThreadBsBufferNum`
//!   workers, only when `iMultipleThreadIdc > 1`; single-threaded encoding never
//!   constructs one.
//! * It lives in `SSliceThreading`, which the context drops at
//!   `WelsUninitEncoderExt`, so the threads end with the encoder and none outlives
//!   it.
//! * A worker panic is reported through `join` and leaves the pool usable.

use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, Thread};

/// A job as the queue holds it: type-erased and lifetime-erased; the module header
/// carries the soundness argument for the `'static`.
type Job = ScopedJob<'static>;
type ScopedJob<'a> = Box<dyn FnOnce() + Send + 'a>;
/// A panic payload, as `std::thread::Result` carries it.
pub type Payload = Box<dyn Any + Send + 'static>;

/// A parked worker's mailbox: its thread handle, registered by the worker itself
/// before it first parks, and the flag a producer claims to wake it.
struct WorkerSlot {
    thread: OnceLock<Thread>,
    /// `true` while the worker is parked or about to park. A producer that wants
    /// this worker swaps it to `false` and unparks; the worker leaving the park
    /// loop on its own clears it too.
    parked: AtomicBool,
}

/// What the pool's threads and its owner share.
struct Shared {
    queue: Mutex<VecDeque<Job>>,
    /// Jobs pushed and not yet popped. Maintained under `queue`'s lock; read
    /// without it by idle threads polling for work.
    pending: AtomicUsize,
    stop: AtomicBool,
    workers: Vec<WorkerSlot>,
    /// Worker threads that have not yet exited their loop. The `Drop` test reads
    /// it after the join.
    live: AtomicUsize,
}

impl Shared {
    fn push(&self, job: Job) {
        {
            let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.push_back(job);
            self.pending.fetch_add(1, Ordering::SeqCst);
        }
        self.wake_one();
    }

    fn try_pop(&self) -> Option<Job> {
        if self.pending.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let job = q.pop_front();
        if job.is_some() {
            self.pending.fetch_sub(1, Ordering::SeqCst);
        }
        job
    }

    /// Wakes one parked worker, if there is one.
    ///
    /// No wake-up is lost: the producer increments `pending` then reads `parked`,
    /// the worker sets `parked` then reads `pending`, and all four are `SeqCst`, so
    /// at least one side sees the other's write — either the worker skips the park
    /// or the producer claims and unparks it. An `unpark` before the `park` keeps
    /// its token.
    fn wake_one(&self) {
        for slot in &self.workers {
            if slot.parked.load(Ordering::SeqCst)
                && slot
                    .parked
                    .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                if let Some(t) = slot.thread.get() {
                    t.unpark();
                }
                return;
            }
        }
    }

    /// A worker's idle wait: park until a producer claims this slot, `stop` is
    /// raised, or work shows up.
    fn wait_for_work(&self, slot: &WorkerSlot) {
        let has_work =
            || self.pending.load(Ordering::SeqCst) > 0 || self.stop.load(Ordering::SeqCst);
        slot.parked.store(true, Ordering::SeqCst);
        loop {
            if has_work() {
                slot.parked.store(false, Ordering::SeqCst);
                return;
            }
            thread::park();
            if !slot.parked.load(Ordering::SeqCst) {
                // Claimed by a producer (or by `Drop`); the wake carries no job
                // by itself, the loop above pops one.
                return;
            }
            // A spurious wake-up: still parked, check again.
        }
    }

    /// The calling thread's wait for `done`: park until every completing job
    /// unparks the scope's owner, which is this thread.
    fn wait_caller(&self, done: impl Fn() -> bool) {
        while !done() {
            thread::park();
        }
    }

    fn worker_main(&self, slot: &WorkerSlot) {
        // Registration before the first park. Its readers (`wake_one`, `Drop`)
        // reach it only through an RMW on `parked` that read this thread's own
        // later write of it, which is the edge that makes the handle visible.
        let _ = slot.thread.set(thread::current());
        loop {
            if let Some(job) = self.try_pop() {
                // The job is a `Scope::spawn` wrapper: it catches its own panics
                // and does all the scope bookkeeping, so nothing here can unwind.
                job();
                continue;
            }
            if self.stop.load(Ordering::SeqCst) {
                break;
            }
            self.wait_for_work(slot);
        }
        self.live.fetch_sub(1, Ordering::Release);
    }
}

/// `N` persistent worker threads and the queue that feeds them. See the module
/// header.
pub struct WorkerPool {
    shared: Arc<Shared>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl WorkerPool {
    /// An inline-only pool with 0 worker threads; infallible and spawns no OS threads.
    pub fn empty() -> Self {
        Self {
            shared: Arc::new(Shared {
                queue: Mutex::new(VecDeque::with_capacity(2)),
                pending: AtomicUsize::new(0),
                stop: AtomicBool::new(false),
                workers: Vec::new(),
                live: AtomicUsize::new(0),
            }),
            threads: Vec::new(),
        }
    }

    /// A pool of `workers` threads. `0` is valid: every job then runs on the
    /// calling thread inside `scope`. Falls back to an inline 0-worker pool if OS
    /// thread creation fails; [`WorkerPool::try_new`] reports the OS error instead.
    pub fn new(workers: usize) -> Self {
        Self::try_new(workers).unwrap_or_else(|_| Self::empty())
    }

    /// [`WorkerPool::new`], returning the OS error of a failed thread creation.
    /// The threads already created are joined before the error is returned.
    pub fn try_new(workers: usize) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::with_capacity(workers.max(1) * 2)),
            pending: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            workers: (0..workers)
                .map(|_| WorkerSlot {
                    thread: OnceLock::new(),
                    parked: AtomicBool::new(false),
                })
                .collect(),
            live: AtomicUsize::new(workers),
        });
        let mut pool = Self {
            shared,
            threads: Vec::with_capacity(workers),
        };
        for k in 0..workers {
            let shared = Arc::clone(&pool.shared);
            let spawned = thread::Builder::new()
                .name(format!("openh264-worker-{k}"))
                .spawn(move || shared.worker_main(&shared.workers[k]));
            match spawned {
                Ok(h) => pool.threads.push(h),
                Err(e) => {
                    // The slots past `k` never got a thread; `live` counts only
                    // the ones that did, so the drop's join is exact.
                    pool.shared.live.fetch_sub(workers - k, Ordering::Release);
                    drop(pool);
                    return Err(e);
                }
            }
        }
        Ok(pool)
    }

    /// How many worker threads the pool owns.
    pub fn workers(&self) -> usize {
        self.threads.len()
    }

    /// Runs `f` with a scope; every job spawned on it has finished when this
    /// returns, whether `f` returned or panicked.
    ///
    /// The signature is `std::thread::scope`'s: jobs may borrow anything that
    /// outlives `'env`, and `'scope` is invariant so it cannot be shrunk to
    /// outlive a local of `f`. Scopes may nest — `f` may call `scope` again on
    /// the same pool, and so may a job — because the calling thread helps with
    /// queued jobs while it waits, so an inner scope never waits on a worker that
    /// is itself waiting.
    ///
    /// # Panics
    /// Resumes a panic of `f` after waiting for the jobs; resumes unwind with
    /// "a job spawned on the worker pool panicked" if a job panicked and its
    /// handle was never joined, as `std::thread::scope` does.
    #[track_caller]
    pub fn scope<'env, F, R>(&'env self, f: F) -> R
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> R,
    {
        let scope = Scope {
            shared: &self.shared,
            data: Arc::new(ScopeData {
                running: AtomicUsize::new(0),
                unhandled_panics: AtomicUsize::new(0),
                owner: thread::current(),
            }),
            scope: PhantomData,
            env: PhantomData,
            not_sync: PhantomData,
        };

        // `f` under `catch_unwind` so that the wait below runs on the panic path
        // too: the frame's locals die when this function unwinds, and a job may
        // still be borrowing them.
        let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));

        // The wait, on both paths: what makes the erased `'static` in `spawn` sound
        // (module header, point 1).
        scope.wait_all();

        match result {
            Err(e) => resume_unwind(e),
            Ok(_) if scope.data.unhandled_panics.load(Ordering::Relaxed) != 0 => {
                resume_unwind(Box::new("a job spawned on the worker pool panicked"))
            }
            Ok(r) => r,
        }
    }

    /// The shared block, for the tests that watch the threads exit.
    #[cfg(test)]
    fn shared_for_tests(&self) -> Arc<Shared> {
        Arc::clone(&self.shared)
    }
}

impl Drop for WorkerPool {
    /// Raises `stop`, wakes every worker and joins them. No scope can be live
    /// here — `scope` borrows the pool for its whole duration — so the queue is
    /// empty and each worker exits at the top of its loop.
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        for slot in &self.shared.workers {
            // A `swap`, not a `store`: the read half of the RMW is what
            // synchronises with the worker's own write of `parked`, and through it
            // with the registration of `thread` before that write. A plain store
            // has no read, so the handle could read as unset here and the worker
            // would stay parked. A worker that has not parked yet reads `stop`
            // (`SeqCst`, stored above) before it would.
            let was_parked = slot.parked.swap(false, Ordering::SeqCst);
            if let Some(t) = slot.thread.get() {
                t.unpark();
            } else {
                debug_assert!(!was_parked, "a parked worker has registered its thread");
            }
        }
        for h in self.threads.drain(..) {
            // A worker cannot panic out of its loop (jobs catch their own), so
            // `Err` here would be a bug in this module, not something to act on
            // during a drop.
            let _ = h.join();
        }
    }
}

impl fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerPool")
            .field("workers", &self.threads.len())
            .field("pending", &self.shared.pending.load(Ordering::Relaxed))
            .finish()
    }
}

/// One scope's bookkeeping, `Arc`-owned so a job can finish its decrement after
/// `scope` has returned.
struct ScopeData {
    /// Jobs spawned and not yet completed.
    running: AtomicUsize,
    /// Jobs that panicked and whose `Err` no `join` has taken yet.
    unhandled_panics: AtomicUsize,
    /// The thread that called `scope`, which every completion unparks.
    owner: Thread,
}

/// A scope to spawn jobs in. See [`WorkerPool::scope`].
pub struct Scope<'scope, 'env: 'scope> {
    shared: &'env Arc<Shared>,
    data: Arc<ScopeData>,
    /// Invariance over `'scope`: without it a job could spawn a job borrowing one
    /// of its own locals.
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
    /// `!Sync` (and `!Send`): jobs are spawned and joined on the owning thread,
    /// the one a completing job unparks.
    not_sync: PhantomData<*mut ()>,
}

impl<'scope, 'env> Scope<'scope, 'env> {
    /// Hands the job to the pool and returns a handle for its result. The job
    /// runs on the first idle worker, or on the calling thread while it waits
    /// in [`JobHandle::join`] or at the end of the scope, whichever comes first.
    /// The job may borrow anything `'env` outlives.
    pub fn spawn<F, T>(&'scope self, f: F) -> JobHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        let packet = Arc::new(Packet {
            result: Mutex::new(None),
            done: AtomicBool::new(false),
        });
        let their_packet = Arc::clone(&packet);
        let data = Arc::clone(&self.data);
        self.data.running.fetch_add(1, Ordering::SeqCst);

        let wrapper = move || {
            // `f` is consumed by the call: its captures are dropped by the time
            // this returns, panic or not.
            let result = catch_unwind(AssertUnwindSafe(f));
            if result.is_err() {
                // Before `done`, so a `join` that observes the `Err` always finds
                // this count already raised.
                data.unhandled_panics.fetch_add(1, Ordering::Relaxed);
            }
            *their_packet
                .result
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(result);
            their_packet.done.store(true, Ordering::Release);
            // The result may borrow `'scope`; if the handle is already gone this
            // drops it, and it happens before the decrement (module header,
            // point 2).
            drop(their_packet);
            data.running.fetch_sub(1, Ordering::Release);
            data.owner.unpark();
            // `data` is an `Arc` — heap, not the frame — and is dropped here.
        };

        let boxed: ScopedJob<'scope> = Box::new(wrapper);
        // SAFETY: lifetime erasure only — the same trait object with `'scope` read
        // as `'static`, an identical fat-pointer layout. Sound because the closure
        // is called, and every capture and result dropped, before `'scope` can end:
        // `scope` waits for `ScopeData::running` to reach zero on both its return
        // and its unwind path, and the wrapper above decrements only after `f` has
        // been consumed and its result handed over. See the module header.
        #[allow(unsafe_code)]
        let job: Job = unsafe { std::mem::transmute::<ScopedJob<'scope>, Job>(boxed) };
        self.shared.push(job);

        JobHandle {
            packet,
            shared: self.shared,
            data: &self.data,
            not_send: PhantomData,
        }
    }

    /// Waits until every job of this scope has completed, running queued ones on
    /// this thread meanwhile.
    fn wait_all(&self) {
        let done = || self.data.running.load(Ordering::Acquire) == 0;
        loop {
            if done() {
                return;
            }
            if let Some(job) = self.shared.try_pop() {
                job();
                continue;
            }
            self.shared.wait_caller(done);
        }
    }
}

impl fmt::Debug for Scope<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope")
            .field("running", &self.data.running.load(Ordering::Relaxed))
            .field(
                "unhandled_panics",
                &self.data.unhandled_panics.load(Ordering::Relaxed),
            )
            .finish()
    }
}

/// A job's result slot.
struct Packet<T> {
    result: Mutex<Option<Result<T, Payload>>>,
    /// Set, with `Release`, after the result is stored.
    done: AtomicBool,
}

/// An owned permission to wait for one job; see [`Scope::spawn`].
pub struct JobHandle<'scope, T> {
    packet: Arc<Packet<T>>,
    shared: &'scope Arc<Shared>,
    data: &'scope ScopeData,
    /// `!Send`: joined on the scope's owning thread.
    not_send: PhantomData<*mut ()>,
}

impl<T> JobHandle<'_, T> {
    /// Waits for this job; `Err` carries the panic payload if it panicked, like
    /// `ScopedJoinHandle::join`. While the job is still queued, or another job
    /// is, this thread runs them itself rather than waiting.
    pub fn join(self) -> Result<T, Payload> {
        let done = || self.packet.done.load(Ordering::Acquire);
        loop {
            if done() {
                break;
            }
            if let Some(job) = self.shared.try_pop() {
                job();
                continue;
            }
            self.shared.wait_caller(done);
        }
        let result = self
            .packet
            .result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("a finished job has a result, and a handle joins once");
        if result.is_err() {
            self.data.unhandled_panics.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    /// Whether the job has completed (not blocking).
    pub fn is_finished(&self) -> bool {
        self.packet.done.load(Ordering::Acquire)
    }
}

impl<T> fmt::Debug for JobHandle<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobHandle")
            .field("done", &self.is_finished())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    /// Jobs borrow the frame's locals — a `Vec` and a counter — and their writes
    /// are visible after the scope. Repeated so the drop of the locals after each
    /// scope is checked against a job that might still hold them.
    #[test]
    fn jobs_borrow_locals_and_publish_their_writes() {
        let pool = WorkerPool::new(3);
        for round in 0..if cfg!(miri) { 4 } else { 200 } {
            let input: Vec<u32> = (0..16).map(|i| i + round).collect();
            let mut output = [0u32; 16];
            let hits = AtomicU32::new(0);
            pool.scope(|s| {
                let mut handles = Vec::new();
                for (k, out) in output.chunks_mut(4).enumerate() {
                    let input = &input;
                    let hits = &hits;
                    handles.push(s.spawn(move || {
                        for (j, o) in out.iter_mut().enumerate() {
                            *o = input[k * 4 + j] * 2;
                        }
                        hits.fetch_add(1, Ordering::Relaxed);
                        k
                    }));
                }
                for (k, h) in handles.into_iter().enumerate() {
                    assert_eq!(h.join().unwrap(), k);
                }
            });
            assert_eq!(hits.load(Ordering::Relaxed), 4);
            assert!(output.iter().zip(&input).all(|(o, i)| *o == i * 2));
        }
    }

    /// A job's result may itself borrow `'scope`.
    #[test]
    fn results_come_back_through_join_and_may_borrow_the_scope() {
        let pool = WorkerPool::new(2);
        let text = String::from("hello, worker");
        let (a, b) = pool.scope(|s| {
            let ha = s.spawn(|| &text[..5]);
            let hb = s.spawn(|| text.len());
            (ha.join().unwrap(), hb.join().unwrap())
        });
        assert_eq!(a, "hello");
        assert_eq!(b, 13);
    }

    /// A panicking job is an `Err` from `join` carrying the payload, and the pool
    /// runs the next scope as if nothing happened.
    #[test]
    #[cfg(panic = "unwind")]
    fn a_panicking_job_is_an_err_and_does_not_poison_the_pool() {
        let pool = WorkerPool::new(2);
        let r = pool.scope(|s| {
            let bad = s.spawn(|| -> u32 { panic!("slice {} failed", 7) });
            let good = s.spawn(|| 42u32);
            (bad.join(), good.join())
        });
        let payload = r.0.expect_err("the panic must come back as Err");
        // A formatted panic's payload is a `String`, unless the compiler folded
        // the literal arguments and it is a `&str`; either way the text is there.
        let text = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        assert_eq!(text, Some("slice 7 failed"));
        assert_eq!(r.1.unwrap(), 42);
        // The pool is intact: same workers, another scope, all jobs run.
        let n = AtomicU32::new(0);
        pool.scope(|s| {
            for _ in 0..6 {
                s.spawn(|| n.fetch_add(1, Ordering::Relaxed));
            }
        });
        assert_eq!(n.load(Ordering::Relaxed), 6);
    }

    /// A job that panicked and was never joined makes the scope panic at its end
    /// — `std::thread::scope`'s rule — and the pool is still usable after.
    #[test]
    #[cfg(panic = "unwind")]
    fn an_unjoined_panic_surfaces_at_the_end_of_the_scope() {
        let pool = WorkerPool::new(1);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|s| {
                s.spawn(|| panic!("dropped on the floor"));
            })
        }));
        let payload = r.expect_err("the scope must panic");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("a job spawned on the worker pool panicked")
        );
        assert_eq!(pool.scope(|s| s.spawn(|| 5).join().unwrap()), 5);
    }

    /// A panic in the scope body with a job in flight: the scope must not unwind
    /// until that job has finished, because the job borrows the body's frame.
    #[test]
    #[cfg(panic = "unwind")]
    fn a_panic_in_the_scope_body_still_waits_for_the_jobs() {
        let pool = WorkerPool::new(2);
        let finished = AtomicBool::new(false);
        let gate = Barrier::new(2);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|s| {
                s.spawn(|| {
                    gate.wait();
                    // Long enough that an unwinding scope which did not wait would
                    // have returned by now.
                    thread::sleep(Duration::from_millis(if cfg!(miri) { 5 } else { 50 }));
                    finished.store(true, Ordering::Release);
                });
                gate.wait();
                panic!("body failed with a job running");
            });
        }));
        assert!(r.is_err());
        assert!(
            finished.load(Ordering::Acquire),
            "scope unwound before its job completed"
        );
    }

    /// A job's result with a destructor that touches the frame: when the handle
    /// is dropped without a join, the worker drops the result before its
    /// decrement, so by the end of the scope the borrow is gone.
    #[test]
    fn an_unjoined_result_is_dropped_before_the_scope_ends() {
        struct Tally<'a>(&'a AtomicU32);
        impl Drop for Tally<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Release);
            }
        }
        let pool = WorkerPool::new(2);
        let dropped = AtomicU32::new(0);
        pool.scope(|s| {
            for _ in 0..3 {
                let h = s.spawn(|| Tally(&dropped));
                drop(h);
            }
        });
        assert_eq!(dropped.load(Ordering::Acquire), 3);
    }

    /// Scopes nest on the same thread: the inner scope's jobs run while the
    /// outer scope's are outstanding, and both complete.
    #[test]
    fn scopes_nest_from_the_same_thread() {
        let pool = WorkerPool::new(2);
        let outer_hits = AtomicU32::new(0);
        let inner_sum = pool.scope(|s| {
            for _ in 0..3 {
                s.spawn(|| outer_hits.fetch_add(1, Ordering::Relaxed));
            }
            pool.scope(|inner| {
                let hs: Vec<_> = (0..4u32).map(|k| inner.spawn(move || k * k)).collect();
                hs.into_iter().map(|h| h.join().unwrap()).sum::<u32>()
            })
        });
        assert_eq!(inner_sum, 0 + 1 + 4 + 9);
        assert_eq!(outer_hits.load(Ordering::Relaxed), 3);
    }

    /// A job may itself open a scope on the pool (the pool is `Sync`): with one
    /// worker, that worker owns the inner scope and helps with its jobs, so this
    /// cannot deadlock.
    #[test]
    fn a_job_may_open_a_nested_scope_on_the_pool() {
        let pool = WorkerPool::new(1);
        let total = pool.scope(|s| {
            let pool = &pool;
            let h = s.spawn(move || {
                pool.scope(|inner| {
                    let a = inner.spawn(|| 20);
                    let b = inner.spawn(|| 22);
                    a.join().unwrap() + b.join().unwrap()
                })
            });
            h.join().unwrap()
        });
        assert_eq!(total, 42);
    }

    /// The calling thread helps: with one worker blocked in job A until job B has
    /// run, `join(B)` runs B on the caller. Without helping this test hangs.
    #[test]
    fn the_caller_runs_queued_jobs_while_it_waits() {
        let pool = WorkerPool::new(1);
        let b_done = AtomicBool::new(false);
        let a_started = Barrier::new(2);
        pool.scope(|s| {
            let a = s.spawn(|| {
                a_started.wait();
                while !b_done.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                "a"
            });
            a_started.wait();
            let b = s.spawn(|| {
                b_done.store(true, Ordering::Release);
                thread::current().name().map(str::to_owned)
            });
            let ran_on = b.join().unwrap();
            assert_eq!(a.join().unwrap(), "a");
            assert_ne!(
                ran_on.as_deref(),
                Some("openh264-worker-0"),
                "the only worker was busy in A, so B must have run on the caller"
            );
        });
    }

    /// More jobs than workers, and their results in spawn order.
    #[test]
    fn more_jobs_than_workers() {
        let pool = WorkerPool::new(2);
        let n = if cfg!(miri) { 12 } else { 200 };
        let results: Vec<usize> = pool.scope(|s| {
            let hs: Vec<_> = (0..n).map(|k| s.spawn(move || k * 3)).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(results, (0..n).map(|k| k * 3).collect::<Vec<_>>());
    }

    /// A scope with no jobs returns at once with `f`'s value.
    #[test]
    fn zero_jobs() {
        let pool = WorkerPool::new(2);
        assert_eq!(pool.scope(|_| 17), 17);
    }

    /// A pool with no workers runs everything on the calling thread.
    #[test]
    fn zero_workers_run_inline() {
        let pool = WorkerPool::new(0);
        assert_eq!(pool.workers(), 0);
        let me = thread::current().id();
        let ids = pool.scope(|s| {
            let hs: Vec<_> = (0..3).map(|_| s.spawn(|| thread::current().id())).collect();
            hs.into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(ids.iter().all(|id| *id == me));
    }

    /// Jobs must not be re-run or lost across many small scopes — the encoder's
    /// shape, one scope per frame.
    #[test]
    fn many_frames_of_four_jobs() {
        let pool = WorkerPool::new(4);
        let frames = if cfg!(miri) { 6 } else { 2000 };
        let mut acc = 0u64;
        for frame in 0..frames as u64 {
            let bufs: Vec<Vec<u64>> = (0..4).map(|k| vec![frame + k; 8]).collect();
            let sum: u64 = pool.scope(|s| {
                let hs: Vec<_> = bufs
                    .iter()
                    .map(|b| s.spawn(move || b.iter().sum::<u64>()))
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).sum()
            });
            assert_eq!(sum, (0..4).map(|k| (frame + k) * 8).sum::<u64>());
            acc += sum;
        }
        assert!(acc > 0);
    }

    /// `Drop` joins the threads: the counter each worker decrements on exit reads
    /// zero afterwards, and it is read through the shared block that outlives the
    /// pool.
    #[test]
    fn dropping_the_pool_ends_its_threads() {
        let pool = WorkerPool::new(3);
        let shared = pool.shared_for_tests();
        assert_eq!(shared.live.load(Ordering::Acquire), 3);
        // Park the workers first, so the drop has to wake them.
        pool.scope(|s| {
            s.spawn(|| 1).join().unwrap();
        });
        thread::sleep(Duration::from_millis(if cfg!(miri) { 1 } else { 5 }));
        drop(pool);
        assert_eq!(shared.live.load(Ordering::Acquire), 0);
        assert!(shared.stop.load(Ordering::Acquire));
        assert_eq!(shared.pending.load(Ordering::Acquire), 0);
    }

    /// The pool is `Sync` (the context that holds it crosses the spawn as
    /// `&sWelsEncCtx`), and `Send`.
    #[test]
    fn the_pool_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WorkerPool>();
    }
}
