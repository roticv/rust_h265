//! Running picture-level passes (the loop filters) over row bands in
//! parallel.
//!
//! A pass that can be split into bands of rows whose writes do not overlap
//! hands each band disjoint `&mut` slices of the three sample planes and
//! runs the bands on scoped threads. With one band, or one thread, the work
//! runs directly on the calling thread, so single-threaded decoding takes
//! exactly the sequential path.

use std::ops::Range;

use crate::shared::SharedTable;

/// One band of the three planes: for each plane, the band's samples and
/// the plane row they start at (indices into the band are
/// `(row - row0) * stride + x`).
pub(crate) type Band<'a, P> = [(&'a mut [P], usize); 3];

/// Split `units` (rows of some fixed height) into at most `threads`
/// contiguous, near-equal ranges.
pub(crate) fn split_units(units: usize, threads: usize) -> Vec<Range<usize>> {
    let n = threads.clamp(1, units.max(1));
    (0..n)
        .map(|i| units * i / n..units * (i + 1) / n)
        .filter(|r| !r.is_empty())
        .collect()
}

/// Run `work(i, band)` for every band `i`. `starts[i]` gives the first row
/// of band `i` in each plane; band `i` runs up to the next band's start
/// (the last to the end of the plane), so `starts[0]` must be all zeros
/// and each plane's starts must be non-decreasing.
pub(crate) fn for_each_band<P, F>(
    planes: &mut [SharedTable<P>; 3],
    strides: [usize; 3],
    starts: &[[usize; 3]],
    work: F,
) where
    P: Send,
    F: Fn(usize, Band<'_, P>) + Sync,
{
    // Each plane cut into its bands, as ranges of exactly their samples
    // (the planes may be views of a picture other threads read).
    let mut cut: Vec<std::vec::IntoIter<&mut [P]>> = planes
        .iter_mut()
        .enumerate()
        .map(|(p, plane)| {
            let len = plane.len();
            let mut cuts: Vec<usize> = starts
                .iter()
                .map(|s| (s[p] * strides[p]).min(len))
                .collect();
            cuts.push(len);
            plane.split_mut(&cuts).into_iter()
        })
        .collect();
    let mut bands: Vec<Band<'_, P>> = starts
        .iter()
        .map(|start| {
            std::array::from_fn(|p| (cut[p].next().expect("one slice per band"), start[p]))
        })
        .collect();
    if bands.len() == 1 {
        let band = bands.pop().unwrap();
        work(0, band);
        return;
    }
    std::thread::scope(|s| {
        let work = &work;
        let mut iter = bands.into_iter().enumerate();
        let first = iter.next();
        for (i, band) in iter {
            s.spawn(move || work(i, band));
        }
        if let Some((i, band)) = first {
            work(i, band);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_units_covers_everything_once() {
        for units in 0..40 {
            for threads in 1..12 {
                let r = split_units(units, threads);
                assert!(r.len() <= threads.max(1));
                let flat: Vec<usize> = r.into_iter().flatten().collect();
                assert_eq!(flat, (0..units).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn bands_are_disjoint_and_complete() {
        // 3 planes with different strides; every band writes its index into
        // its own rows, and every row ends up written exactly once.
        let strides = [4, 2, 2];
        let rows = [12, 6, 6];
        let mut planes: [SharedTable<u16>; 3] =
            std::array::from_fn(|p| vec![0; strides[p] * rows[p]].into());
        let starts = [[0, 0, 0], [4, 2, 2], [9, 4, 5]];
        for_each_band(&mut planes, strides, &starts, |i, band| {
            for (p, (slice, row0)) in band.into_iter().enumerate() {
                assert_eq!(row0, starts[i][p]);
                slice.iter_mut().for_each(|s| *s = i as u16 + 1);
            }
        });
        for p in 0..3 {
            for row in 0..rows[p] {
                let band = starts.iter().rposition(|s| s[p] <= row).unwrap() as u16 + 1;
                assert!(
                    planes[p][row * strides[p]..][..strides[p]]
                        .iter()
                        .all(|&s| s == band)
                );
            }
        }
    }
}
