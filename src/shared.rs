//! Picture tables shared between the threads decoding one picture.
//!
//! This is the only module of the crate with `unsafe` code. It exists so
//! that several CTB rows of one picture can be decoded at the same time
//! (wavefront parallel processing): each row writes its own part of the
//! sample planes and per-picture tables while reading what the rows above
//! have already finished.
//!
//! # The tables
//!
//! A [`SharedTable`] is either the *owner* of a buffer (the picture state's
//! table, used like a `Vec` / slice) or a *view* of an owner's buffer (one
//! per worker thread, created by [`SharedTable::view`]). A view supports
//! only element and range indexing, which create references to exactly the
//! elements accessed and never to the whole buffer. Two threads accessing
//! disjoint elements through their views therefore never hold overlapping
//! references. Whole-buffer operations (`Deref` to a slice) panic on a view.
//!
//! # The protocol
//!
//! Views are sound as long as no element is written by one thread while
//! another thread accesses it. The wavefront schedule guarantees that:
//!
//! - every element belongs to one CTB (by its position in the picture);
//! - a worker writes only elements of the CTB it is decoding;
//! - a worker reads elements of CTBs in its own CTB row, or of CTBs in rows
//!   above that are finished: row `r` decodes CTB `c` only after row `r - 1`
//!   has finished CTB `c + 1` (published through [`RowProgress`]).
//!
//! In test builds each view checks every access against that rule (see
//! [`AccessCheck`]), so the decode path cannot silently step outside it:
//! the threaded decoding tests run every WPP fixture with checking on.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A buffer that is either owned (behaves like a `Vec<T>` / `[T]`) or a
/// view of another table's buffer for one worker thread. See the module
/// documentation for the rules.
pub struct SharedTable<T> {
    ptr: *mut T,
    len: usize,
    /// Capacity of the allocation when owned; `None` for a view.
    cap: Option<usize>,
    /// Access checking for a view, in test builds.
    #[cfg(test)]
    check: Option<(Arc<AccessCheck>, Geometry)>,
    _owns: PhantomData<T>,
}

// SAFETY: an owned table is a `Vec<T>`, which is `Send` / `Sync` when `T`
// is. Views share the buffer between threads; the wavefront protocol (see
// the module documentation) ensures no element is accessed by one thread
// while another writes it, so sharing views is sound for `T: Send + Sync`.
unsafe impl<T: Send> Send for SharedTable<T> {}
unsafe impl<T: Send + Sync> Sync for SharedTable<T> {}

impl<T> SharedTable<T> {
    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether this table owns its buffer (false for a worker view).
    #[inline]
    pub fn is_owner(&self) -> bool {
        self.cap.is_some()
    }

    /// The owned buffer as a `Vec`. Panics on a view.
    pub fn into_vec(self) -> Vec<T> {
        let cap = self.cap.expect("into_vec on a worker view");
        let (ptr, len) = (self.ptr, self.len);
        std::mem::forget(self);
        // SAFETY: `ptr`, `len` and `cap` come from `Vec::into_raw_parts`
        // equivalent in `From<Vec<T>>` and were not changed since.
        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    /// A view of this table's buffer for a worker thread.
    ///
    /// # Safety
    ///
    /// The view must be dropped before this table is accessed, moved out
    /// of, or dropped again, and every access made through views of the
    /// same table must follow the wavefront protocol (module
    /// documentation): no element accessed by one thread while another
    /// writes it.
    pub unsafe fn view(&self) -> SharedTable<T> {
        SharedTable {
            ptr: self.ptr,
            len: self.len,
            cap: None,
            #[cfg(test)]
            check: None,
            _owns: PhantomData,
        }
    }

    /// Attach access checking to a view (test builds only).
    #[cfg(test)]
    pub fn set_check(&mut self, check: Arc<AccessCheck>, geometry: Geometry) {
        debug_assert!(!self.is_owner());
        self.check = Some((check, geometry));
    }

    #[inline(always)]
    fn check_read(&self, _range: Range<usize>) {
        #[cfg(test)]
        if let Some((c, g)) = &self.check {
            c.check(g, _range, false);
        }
    }

    #[inline(always)]
    fn check_write(&self, _range: Range<usize>) {
        #[cfg(test)]
        if let Some((c, g)) = &self.check {
            c.check(g, _range, true);
        }
    }

    /// Element `i`, or `None` when out of range.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&T> {
        if i < self.len {
            self.check_read(i..i + 1);
            // SAFETY: in bounds; the protocol rules out a concurrent write.
            Some(unsafe { &*self.ptr.add(i) })
        } else {
            None
        }
    }

    /// Element `i` for writing, or `None` when out of range.
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        if i < self.len {
            self.check_write(i..i + 1);
            // SAFETY: in bounds; the protocol rules out any concurrent access.
            Some(unsafe { &mut *self.ptr.add(i) })
        } else {
            None
        }
    }

    #[inline(always)]
    fn range_bounds(&self, r: &Range<usize>) {
        if r.start > r.end || r.end > self.len {
            range_fail(r, self.len);
        }
    }
}

#[cold]
#[inline(never)]
fn range_fail(r: &Range<usize>, len: usize) -> ! {
    panic!(
        "range {}..{} out of bounds for table of length {len}",
        r.start, r.end
    )
}

#[cold]
#[inline(never)]
fn index_fail(i: usize, len: usize) -> ! {
    panic!("index {i} out of bounds for table of length {len}")
}

impl<T> Default for SharedTable<T> {
    fn default() -> Self {
        Vec::new().into()
    }
}

impl<T> From<Vec<T>> for SharedTable<T> {
    fn from(v: Vec<T>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        SharedTable {
            ptr: v.as_mut_ptr(),
            len: v.len(),
            cap: Some(v.capacity()),
            #[cfg(test)]
            check: None,
            _owns: PhantomData,
        }
    }
}

impl<T> From<SharedTable<T>> for Vec<T> {
    /// The owned buffer. Panics on a view.
    fn from(t: SharedTable<T>) -> Self {
        t.into_vec()
    }
}

impl<T> Drop for SharedTable<T> {
    fn drop(&mut self) {
        if let Some(cap) = self.cap {
            // SAFETY: an owner's raw parts come from a `Vec` (see `From`).
            drop(unsafe { Vec::from_raw_parts(self.ptr, self.len, cap) });
        }
    }
}

impl<T: Clone> Clone for SharedTable<T> {
    /// Clones an owned table into a new owned table. Panics on a view.
    fn clone(&self) -> Self {
        self.deref().to_vec().into()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for SharedTable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_owner() {
            self.deref().fmt(f)
        } else {
            write!(f, "SharedTable view of {} elements", self.len)
        }
    }
}

impl<T> Deref for SharedTable<T> {
    type Target = [T];
    /// The whole buffer. Owned tables only: a view panics, since a
    /// reference to the whole buffer would overlap other threads' rows.
    #[inline]
    fn deref(&self) -> &[T] {
        assert!(self.is_owner(), "whole-table access through a worker view");
        // SAFETY: an owner has no live views while it is accessed (see
        // `view`), and its raw parts describe its allocation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for SharedTable<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        assert!(self.is_owner(), "whole-table access through a worker view");
        // SAFETY: as for `deref`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Index<usize> for SharedTable<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        if i >= self.len {
            index_fail(i, self.len);
        }
        self.check_read(i..i + 1);
        // SAFETY: in bounds; the protocol rules out a concurrent write.
        unsafe { &*self.ptr.add(i) }
    }
}

impl<T> IndexMut<usize> for SharedTable<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        if i >= self.len {
            index_fail(i, self.len);
        }
        self.check_write(i..i + 1);
        // SAFETY: in bounds; the protocol rules out any concurrent access.
        unsafe { &mut *self.ptr.add(i) }
    }
}

impl<T> Index<Range<usize>> for SharedTable<T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: Range<usize>) -> &[T] {
        self.range_bounds(&r);
        self.check_read(r.clone());
        // SAFETY: in bounds; the protocol rules out a concurrent write to
        // any element of the range.
        unsafe { std::slice::from_raw_parts(self.ptr.add(r.start), r.end - r.start) }
    }
}

impl<T> IndexMut<Range<usize>> for SharedTable<T> {
    #[inline(always)]
    fn index_mut(&mut self, r: Range<usize>) -> &mut [T] {
        self.range_bounds(&r);
        self.check_write(r.clone());
        // SAFETY: in bounds; the protocol rules out any concurrent access
        // to the elements of the range.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(r.start), r.end - r.start) }
    }
}

/// Whole-table and open-ended ranges: owned tables only (through `Deref`).
macro_rules! owner_range_index {
    ($($r:ty),*) => {$(
        impl<T> Index<$r> for SharedTable<T> {
            type Output = [T];
            #[inline]
            fn index(&self, r: $r) -> &[T] {
                &self.deref()[r]
            }
        }
        impl<T> IndexMut<$r> for SharedTable<T> {
            #[inline]
            fn index_mut(&mut self, r: $r) -> &mut [T] {
                &mut self.deref_mut()[r]
            }
        }
    )*};
}
owner_range_index!(
    std::ops::RangeFrom<usize>,
    std::ops::RangeTo<usize>,
    std::ops::RangeFull,
    std::ops::RangeInclusive<usize>
);

// ---- Row progress ------------------------------------------------------

/// How far each CTB row of the picture is decoded, shared by the workers:
/// `done(r)` is the number of leading CTBs of row `r` that are finished
/// (the row width once the row is complete).
pub struct RowProgress {
    done: Vec<AtomicUsize>,
    /// Set when decoding stops early (an error or a slice that ends before
    /// its last row): waiting rows give up instead of waiting forever.
    abort: AtomicBool,
}

impl RowProgress {
    /// `initial[r]`: CTBs of row `r` finished before the parallel section
    /// starts (by earlier slices).
    pub fn new(initial: impl IntoIterator<Item = usize>) -> Self {
        Self {
            done: initial.into_iter().map(AtomicUsize::new).collect(),
            abort: AtomicBool::new(false),
        }
    }

    /// Finished CTBs of row `r` (`Acquire`: everything the row's worker
    /// wrote for those CTBs is visible afterwards).
    #[inline]
    pub fn done(&self, r: usize) -> usize {
        self.done[r].load(Ordering::Acquire)
    }

    /// Publish that row `r` has finished its first `n` CTBs (`Release`).
    #[inline]
    pub fn publish(&self, r: usize, n: usize) {
        self.done[r].store(n, Ordering::Release);
    }

    pub fn abort(&self) {
        self.abort.store(true, Ordering::Release);
    }

    pub fn aborted(&self) -> bool {
        self.abort.load(Ordering::Acquire)
    }

    /// Wait until row `r` has finished at least `n` CTBs. Returns false if
    /// decoding was aborted first.
    pub fn wait(&self, r: usize, n: usize) -> bool {
        let mut spins = 0u32;
        while self.done(r) < n {
            if self.aborted() {
                return false;
            }
            if spins < 64 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
        true
    }
}

// ---- Access checking (test builds) ---------------------------------------

/// How a table's elements map onto CTBs: element `i` is at table row
/// `i / row_len`, column `i % row_len`, and a CTB covers `ctb_rows` table
/// rows and `ctb_cols` table columns.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub row_len: usize,
    pub ctb_rows: usize,
    pub ctb_cols: usize,
}

/// Validates a worker's table accesses against the wavefront protocol (in
/// test builds): writes only into the CTB being decoded, reads only from
/// the worker's own CTB row or from finished CTBs of rows above.
pub struct AccessCheck {
    progress: Arc<RowProgress>,
    /// The CTB being decoded, `(row, column)`, or `None` between CTBs.
    current: std::sync::Mutex<Option<(usize, usize)>>,
}

impl AccessCheck {
    pub fn new(progress: Arc<RowProgress>) -> Self {
        Self {
            progress,
            current: std::sync::Mutex::new(None),
        }
    }

    pub fn set_current(&self, ctb: Option<(usize, usize)>) {
        *self.current.lock().unwrap() = ctb;
    }

    fn check(&self, g: &Geometry, range: Range<usize>, write: bool) {
        if range.is_empty() {
            return;
        }
        let Some((row, col)) = *self.current.lock().unwrap() else {
            return;
        };
        for i in [range.start, range.end - 1] {
            let (r, c) = ((i / g.row_len) / g.ctb_rows, (i % g.row_len) / g.ctb_cols);
            let ok = if write {
                r == row && c == col
            } else {
                r == row || (r < row && self.progress.done(r) > c)
            };
            assert!(
                ok,
                "wavefront violation: CTB ({row}, {col}) {} element {i} in CTB ({r}, {c})",
                if write { "writes" } else { "reads" },
            );
        }
        // A range spanning several table rows covers whole rows in between:
        // checking its first and last element is enough for one CTB row,
        // but not when it crosses CTB rows.
        let (r0, r1) = (
            (range.start / g.row_len) / g.ctb_rows,
            ((range.end - 1) / g.row_len) / g.ctb_rows,
        );
        assert!(
            r0 == r1,
            "wavefront violation: CTB ({row}, {col}) accesses a range spanning CTB rows {r0}..={r1}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_behaves_like_a_vec() {
        let mut t: SharedTable<u32> = vec![1, 2, 3, 4].into();
        assert_eq!(t.len(), 4);
        t[1] = 7;
        t[2..4].copy_from_slice(&[8, 9]);
        assert_eq!(&t[..], &[1, 7, 8, 9]);
        assert_eq!(t.get(4), None);
        let c = t.clone();
        assert_eq!(t.into_vec(), vec![1, 7, 8, 9]);
        assert_eq!(&c[..], &[1, 7, 8, 9]);
    }

    #[test]
    fn views_write_disjoint_elements_from_threads() {
        let t: SharedTable<u32> = vec![0; 64].into();
        // SAFETY: the views are dropped inside the scope, before `t` is
        // used again, and each thread writes only its own 16 elements.
        let views: Vec<SharedTable<u32>> = (0..4).map(|_| unsafe { t.view() }).collect();
        std::thread::scope(|s| {
            for (k, mut v) in views.into_iter().enumerate() {
                s.spawn(move || {
                    for i in k * 16..(k + 1) * 16 {
                        v[i] = i as u32;
                    }
                    v[k * 16..k * 16 + 2].copy_from_slice(&[100, 101]);
                });
            }
        });
        for k in 0..4 {
            assert_eq!(t[k * 16], 100);
            assert_eq!(t[k * 16 + 1], 101);
            assert_eq!(t[k * 16 + 5], (k * 16 + 5) as u32);
        }
    }

    #[test]
    #[should_panic(expected = "whole-table access through a worker view")]
    fn view_refuses_whole_table_access() {
        let t: SharedTable<u8> = vec![0; 4].into();
        // SAFETY: single-threaded; the view is dropped before `t`.
        let v = unsafe { t.view() };
        let _ = v.iter().count();
    }

    #[test]
    fn access_check_enforces_the_wavefront_rule() {
        // 3 CTB rows x 4 CTB columns, one element per CTB.
        let g = Geometry {
            row_len: 4,
            ctb_rows: 1,
            ctb_cols: 1,
        };
        let progress = Arc::new(RowProgress::new([4, 3, 0]));
        let check = AccessCheck::new(progress);
        check.set_current(Some((1, 1)));
        check.check(&g, 0..1, false); // row above, finished
        check.check(&g, 4 + 3..4 + 4, false); // own row, any column
        check.check(&g, 4 + 1..4 + 2, true); // own CTB
        let r = std::panic::catch_unwind(|| check.check(&g, 4 + 2..4 + 3, true));
        assert!(r.is_err(), "write outside the current CTB");
        let r = std::panic::catch_unwind(|| check.check(&g, 8..9, false));
        assert!(r.is_err(), "read from a row below");
        let check2 = AccessCheck::new(Arc::new(RowProgress::new([2, 0, 0])));
        check2.set_current(Some((1, 1)));
        let r = std::panic::catch_unwind(|| check2.check(&g, 2..3, false));
        assert!(r.is_err(), "read of an unfinished CTB above");
    }

    #[test]
    fn progress_wait_and_abort() {
        let p = Arc::new(RowProgress::new([0, 0]));
        let p2 = Arc::clone(&p);
        let h = std::thread::spawn(move || p2.wait(0, 3));
        p.publish(0, 3);
        assert!(h.join().unwrap());
        let p3 = Arc::clone(&p);
        let h = std::thread::spawn(move || p3.wait(1, 1));
        p.abort();
        assert!(!h.join().unwrap());
    }
}
