//! Decode an H.265 bitstream and display frames in a window.
//!
//! Usage: cargo run --example play -- <input.h265> [--fps N] [--loop] [--threads N]
//!
//! Streams decode: feeds NALs incrementally, lets the decoder's
//! display-order mode release frames in display order, then displays each
//! frame at the specified rate.
//! Press Escape to quit. With --loop, playback repeats from the beginning.
//! Decodes on all cores by default (`Decoder::with_threads`); `--threads 1`
//! decodes on the calling thread.

use minifb::{Key, Window, WindowOptions};
use rust_h265::decoder::{Decoder, Frame};
use rust_h265::nal::{NalUnit, parse_annex_b};
use std::time::{Duration, Instant};

/// Convert a YUV frame (4:2:0 or 4:4:4) to an ARGB pixel buffer for display.
fn yuv_to_argb(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    chroma_format_idc: u32,
) -> Vec<u32> {
    // Chroma subsampling shift: 1 for 4:2:0, 0 for 4:4:4.
    let cs = if chroma_format_idc == 3 { 0 } else { 1 };
    let cw = width >> cs;
    let mut argb = vec![0u32; width * height];
    for row in 0..height {
        for col in 0..width {
            let y_val = y[row * width + col] as i32;
            let u_val = u[(row >> cs) * cw + (col >> cs)] as i32 - 128;
            let v_val = v[(row >> cs) * cw + (col >> cs)] as i32 - 128;

            let r = (y_val + ((v_val * 359 + 128) >> 8)).clamp(0, 255) as u32;
            let g = (y_val - ((u_val * 88 + v_val * 183 + 128) >> 8)).clamp(0, 255) as u32;
            let b = (y_val + ((u_val * 454 + 128) >> 8)).clamp(0, 255) as u32;

            argb[row * width + col] = (r << 16) | (g << 8) | b;
        }
    }
    argb
}

/// Frames in display order, decoded on demand: the decoder's display-order
/// mode releases each frame once the stream's reorder limit allows it.
struct FrameSource<'a> {
    nals: Vec<NalUnit<'a>>,
    next_nal: usize,
    decoder: Decoder,
}

impl<'a> FrameSource<'a> {
    fn new(data: &'a [u8], threads: usize) -> Self {
        Self {
            nals: parse_annex_b(data),
            next_nal: 0,
            decoder: Decoder::new().with_display_order().with_threads(threads),
        }
    }

    fn next_frame(&mut self) -> Option<Frame> {
        loop {
            if let Some(f) = self.decoder.next_output() {
                return Some(f);
            }
            let Some(nal) = self.nals.get(self.next_nal) else {
                return self.decoder.flush();
            };
            if let Err(e) = self.decoder.decode_nal(nal) {
                eprintln!("Error decoding NAL {}: {:?}", self.next_nal, e);
            }
            self.next_nal += 1;
        }
    }
}

fn to_argb(f: &Frame) -> Vec<u32> {
    yuv_to_argb(
        f.y.as_u8().expect("8-bit"),
        f.u.as_u8().expect("8-bit"),
        f.v.as_u8().expect("8-bit"),
        f.width as usize,
        f.height as usize,
        f.chroma_format_idc,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: play <input.h265> [--fps N] [--loop] [--threads N]");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let mut fps_override: Option<f64> = None;
    let mut do_loop = false;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fps" => {
                i += 1;
                fps_override = args.get(i).and_then(|s| s.parse().ok());
            }
            "--loop" => do_loop = true,
            "--threads" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse().ok()) {
                    Some(n) => threads = n,
                    None => eprintln!("--threads needs a number"),
                }
            }
            _ => eprintln!("Unknown option: {}", args[i]),
        }
        i += 1;
    }

    let h265_data = match std::fs::read(input_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error: cannot read '{}': {}", input_path, e);
            std::process::exit(1);
        }
    };

    let mut source = FrameSource::new(&h265_data, threads);
    let Some(first) = source.next_frame() else {
        eprintln!("No frames decoded.");
        std::process::exit(1);
    };
    let (width, height) = (first.width as usize, first.height as usize);
    let first_frame_argb = to_argb(&first);

    let fps = fps_override.unwrap_or(30.0);
    let fps_source = if fps_override.is_some() {
        "user override"
    } else {
        "default"
    };

    eprintln!(
        "Playing {} ({}x{}) at {} fps ({}) -- press Escape to quit",
        input_path, width, height, fps, fps_source
    );

    // Auto-scale small videos.
    let scale = if width <= 128 && height <= 128 {
        4
    } else if width <= 320 && height <= 240 {
        2
    } else {
        1
    };

    let mut window = Window::new(
        &format!("rust_h265 -- {} ({}x{})", input_path, width, height),
        width * scale,
        height * scale,
        WindowOptions {
            resize: true,
            scale_mode: minifb::ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");

    let frame_duration = Duration::from_secs_f64(1.0 / fps);
    let mut last_frame_time = Instant::now();
    let mut frame_count = 0u64;

    // Display first frame.
    window
        .update_with_buffer(&first_frame_argb, width, height)
        .expect("failed to update window");
    frame_count += 1;
    let mut last_argb = first_frame_argb;

    'outer: loop {
        let Some(f) = source.next_frame() else {
            // End of stream.
            if do_loop {
                source = FrameSource::new(&h265_data, threads);
                eprintln!("Looping... ({} frames played)", frame_count);
                continue;
            }
            eprintln!("Finished ({} frames). Press Escape to quit.", frame_count);
            while window.is_open() && !window.is_key_down(Key::Escape) {
                window
                    .update_with_buffer(&last_argb, width, height)
                    .expect("failed to update window");
                std::thread::sleep(Duration::from_millis(10));
            }
            break;
        };
        last_argb = to_argb(&f);
        frame_count += 1;

        // Wait for frame timing.
        loop {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                break 'outer;
            }
            let now = Instant::now();
            if now.duration_since(last_frame_time) >= frame_duration {
                last_frame_time = now;
                break;
            }
            window.update();
            std::thread::sleep(Duration::from_millis(1));
        }

        window
            .update_with_buffer(&last_argb, width, height)
            .expect("failed to update window");
    }

    eprintln!("Played {} frames", frame_count);
}
