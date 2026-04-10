//! Phase 3d-1: Decoded Picture Buffer (DPB) infrastructure.
//!
//! HEVC references pictures by POC (picture order count) rather than by
//! index. The DPB is a pool of recently decoded pictures, each marked as
//! "short-term reference", "long-term reference", or "unused for reference".
//! Pictures are removed from the DPB when no future slice in decode order
//! could reference them any more.
//!
//! **Phase 3d-1 scope**: only the data structures + insertion / lookup /
//! marking plumbing. No actual inter decoding uses the DPB yet — see Phase
//! 3d-2 / 3d-3 for that. The purpose of landing the DPB now is to give
//! non-IDR slice header parsing somewhere to look up references and mark
//! them.
//!
//! The lifetime shape intentionally uses `Rc<RefCell<...>>` so that the
//! same decoded picture can be simultaneously held by the DPB (for
//! marking / lookup) and later by a `RefPicList` entry during inter
//! decode, without resorting to raw pointers or indices that would be
//! invalidated when the DPB is compacted.

use std::cell::RefCell;
use std::rc::Rc;

use crate::sps::Sps;

/// How this picture is currently marked in the DPB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureReferenceStatus {
    /// No future slice can reference this picture.
    UnusedForReference,
    /// Short-term reference (POC delta relative to the current picture).
    ShortTerm,
    /// Long-term reference (identified by POC LSB).
    LongTerm,
}

/// One decoded picture's planes + metadata.
#[derive(Debug)]
pub struct DecodedPicture {
    /// Luma plane in raster order, `width * height` bytes.
    pub y: Vec<u8>,
    /// Cb plane.
    pub u: Vec<u8>,
    /// Cr plane.
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Signed picture order count (may be negative if a non-IDR with
    /// earlier LSB appears after an IDR — in practice this is only
    /// non-positive for RASL / RADL pictures, which Phase 3d-1 doesn't
    /// decode anyway).
    pub poc: i32,
    /// Reference marking state. Mutable since a picture's state changes
    /// over its lifetime in the DPB.
    pub reference_status: RefCell<PictureReferenceStatus>,
    /// Whether the picture has been output yet (emitted as a `Frame`).
    pub output: RefCell<bool>,
}

impl DecodedPicture {
    pub fn new(y: Vec<u8>, u: Vec<u8>, v: Vec<u8>, width: u32, height: u32, poc: i32) -> Self {
        Self {
            y,
            u,
            v,
            width,
            height,
            poc,
            reference_status: RefCell::new(PictureReferenceStatus::ShortTerm),
            output: RefCell::new(false),
        }
    }

    pub fn mark(&self, status: PictureReferenceStatus) {
        *self.reference_status.borrow_mut() = status;
    }

    pub fn reference_status(&self) -> PictureReferenceStatus {
        *self.reference_status.borrow()
    }

    pub fn is_reference(&self) -> bool {
        !matches!(
            self.reference_status(),
            PictureReferenceStatus::UnusedForReference
        )
    }
}

/// The decoded picture buffer.
///
/// Holds a pool of recently decoded pictures. Phase 3d-1 does not
/// implement proper bumping / reorder-aware output — it only provides the
/// scaffolding so later phases can wire in actual inter decoding and
/// display ordering.
#[derive(Debug, Default)]
pub struct DecodedPictureBuffer {
    pictures: Vec<Rc<DecodedPicture>>,
    /// Maximum number of pictures that may sit in the DPB at once
    /// (`sps_max_dec_pic_buffering_minus1 + 1`). Zero means "unbounded";
    /// the caller will fill it in from the active SPS.
    max_dec_pic_buffering: usize,
    /// `sps_max_num_reorder_pics` — the maximum number of pictures that
    /// may sit in the DPB before the current picture in decode order
    /// precedes the current picture in output order. Used by `bump()`.
    max_num_reorder_pics: usize,
    /// `sps_max_latency_increase_plus1` — when non-zero, a latency bound
    /// on how long a picture may sit in the DPB awaiting output.
    #[allow(dead_code)]
    max_latency_increase_plus1: u32,
}

impl DecodedPictureBuffer {
    /// Create an empty DPB. Call `configure_from_sps` on the first slice
    /// of each new sequence to populate the sizing fields from the SPS.
    pub fn new() -> Self {
        Self::default()
    }

    /// Populate sizing fields from the active SPS. Should be called once
    /// per activated SPS (idempotent — calling twice is harmless).
    pub fn configure_from_sps(&mut self, sps: &Sps) {
        self.max_dec_pic_buffering = (sps.sps_max_dec_pic_buffering_minus1 + 1) as usize;
        self.max_num_reorder_pics = sps.sps_max_num_reorder_pics as usize;
        self.max_latency_increase_plus1 = sps.sps_max_latency_increase_plus1;
    }

    /// Insert a freshly decoded picture into the DPB. The new picture
    /// starts out marked as `ShortTerm` per HEVC conventions; callers
    /// that need a different initial state can `mark()` it afterwards.
    pub fn insert(&mut self, pic: Rc<DecodedPicture>) {
        self.pictures.push(pic);
    }

    /// Total number of pictures currently in the DPB.
    pub fn len(&self) -> usize {
        self.pictures.len()
    }

    /// Returns true when the DPB contains no pictures.
    pub fn is_empty(&self) -> bool {
        self.pictures.is_empty()
    }

    /// Find a picture by POC. Returns `Some(...)` for the first match,
    /// `None` if no picture in the DPB has the given POC.
    pub fn find_by_poc(&self, poc: i32) -> Option<Rc<DecodedPicture>> {
        self.pictures.iter().find(|p| p.poc == poc).cloned()
    }

    /// Find a short-term reference picture by POC. Like `find_by_poc`
    /// but skips pictures that aren't currently marked as ST references.
    pub fn find_st_ref(&self, poc: i32) -> Option<Rc<DecodedPicture>> {
        self.pictures
            .iter()
            .find(|p| p.poc == poc && p.reference_status() == PictureReferenceStatus::ShortTerm)
            .cloned()
    }

    /// Find a long-term reference picture by POC LSB. Checks the picture's
    /// full POC's low bits against `poc_lsb`, which mirrors how HEVC LT
    /// refs are identified (spec 8.3.2).
    pub fn find_lt_ref_by_lsb(&self, poc_lsb: u32, max_poc_lsb: i32) -> Option<Rc<DecodedPicture>> {
        self.pictures
            .iter()
            .find(|p| {
                p.reference_status() == PictureReferenceStatus::LongTerm
                    && (p.poc.rem_euclid(max_poc_lsb)) == poc_lsb as i32
            })
            .cloned()
    }

    /// Mark every reference picture in the DPB as `UnusedForReference`.
    /// Used when the current RPS has been fully resolved and we need to
    /// flip everything that wasn't referenced back to unused state.
    pub fn unmark_all_references(&self) {
        for p in &self.pictures {
            p.mark(PictureReferenceStatus::UnusedForReference);
        }
    }

    /// Remove every picture that is both "unused for reference" and has
    /// already been output. Mirrors FFmpeg's DPB cleanup step. Safe to
    /// call after every picture insertion.
    pub fn cleanup_unused(&mut self) {
        self.pictures
            .retain(|p| p.is_reference() || !*p.output.borrow());
    }

    /// Phase 3d-1 stub: return the next picture in output order if the
    /// DPB is "full enough" to bump one out, or `None` otherwise. Because
    /// we don't yet handle reorder-aware output, this always returns None
    /// — the `Decoder` emits each picture immediately after it is decoded.
    /// Phase 3d-3+ will replace this with a real bumping process.
    pub fn bump(&mut self) -> Option<Rc<DecodedPicture>> {
        None
    }

    /// Direct access to the pictures list, for iteration by the caller
    /// when deriving the reference picture set.
    pub fn pictures(&self) -> &[Rc<DecodedPicture>] {
        &self.pictures
    }
}

/// Phase 3d-1: derived reference picture set groupings per spec 8.3.2.
///
/// The five sets are the input to `RefPicList0` / `RefPicList1`
/// construction (which Phase 3d-1 doesn't yet do). We compute them here
/// so that inter-decode phases can slot straight in.
#[derive(Debug, Default, Clone)]
pub struct ReferencePictureSets {
    /// Short-term "before current picture" references that the current
    /// picture itself uses (used_by_curr_pic_s0_flag = 1).
    pub st_curr_before: Vec<i32>,
    /// Short-term "after current picture" references that the current
    /// picture itself uses.
    pub st_curr_after: Vec<i32>,
    /// Short-term references that the current picture does NOT use but
    /// which must still be kept around (they're referenced by later
    /// pictures according to the RPS).
    pub st_foll: Vec<i32>,
    /// Long-term references that the current picture uses.
    pub lt_curr: Vec<i32>,
    /// Long-term references that the current picture does not use but
    /// which later pictures will.
    pub lt_foll: Vec<i32>,
}

impl ReferencePictureSets {
    /// Total number of "current" references (= `NumPocTotalCurr` in the
    /// spec). Used to decide whether `ref_pic_lists_modification` can
    /// appear in the slice header, etc.
    pub fn num_poc_total_curr(&self) -> usize {
        self.st_curr_before.len() + self.st_curr_after.len() + self.lt_curr.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup_by_poc() {
        let mut dpb = DecodedPictureBuffer::new();
        let pic = Rc::new(DecodedPicture::new(
            vec![0; 16 * 16],
            vec![0; 8 * 8],
            vec![0; 8 * 8],
            16,
            16,
            5,
        ));
        dpb.insert(pic);
        assert_eq!(dpb.len(), 1);
        assert!(dpb.find_by_poc(5).is_some());
        assert!(dpb.find_by_poc(4).is_none());
    }

    #[test]
    fn mark_short_term_reference() {
        let pic = Rc::new(DecodedPicture::new(vec![], vec![], vec![], 0, 0, 10));
        assert_eq!(pic.reference_status(), PictureReferenceStatus::ShortTerm);
        pic.mark(PictureReferenceStatus::LongTerm);
        assert_eq!(pic.reference_status(), PictureReferenceStatus::LongTerm);
        pic.mark(PictureReferenceStatus::UnusedForReference);
        assert!(!pic.is_reference());
    }

    #[test]
    fn cleanup_drops_unreferenced_and_output() {
        let mut dpb = DecodedPictureBuffer::new();
        let a = Rc::new(DecodedPicture::new(vec![], vec![], vec![], 0, 0, 0));
        a.mark(PictureReferenceStatus::UnusedForReference);
        *a.output.borrow_mut() = true;
        let b = Rc::new(DecodedPicture::new(vec![], vec![], vec![], 0, 0, 1));
        // b is still a reference.
        dpb.insert(a);
        dpb.insert(b);
        dpb.cleanup_unused();
        assert_eq!(dpb.len(), 1);
        assert_eq!(dpb.pictures()[0].poc, 1);
    }

    #[test]
    fn num_poc_total_curr_sums_three_sets() {
        let rps = ReferencePictureSets {
            st_curr_before: vec![-1, -2],
            st_curr_after: vec![1],
            st_foll: vec![-4],
            lt_curr: vec![100],
            lt_foll: vec![],
        };
        assert_eq!(rps.num_poc_total_curr(), 4);
    }
}
