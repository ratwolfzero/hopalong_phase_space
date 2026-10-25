//! Hopalong orbit density histogram, single pass (Rust port of hopalong_orbit_onepass.py).
//!
//! The histogram lives on a dyadic grid. Whenever a point leaves the grid, the
//! occupied block of counts is translated if it still fits at the current cell
//! size, and otherwise merged 2x2 into half the resolution, so counts stay exact.
//! Memory use is independent of the number of iterations.

use clap::Parser;
use std::path::Path;

const ESCAPE: f64 = 1e6;
const WARMUP: usize = 4096; // points buffered to choose the starting cell size
const H_MIN: f64 = 1.0 / 4294967296.0; // 2^-32; all cell sizes are H_MIN * 2^k

#[derive(Parser, Debug)]
#[command(author, version, about = "Hopalong orbit density histogram (one pass) in Rust")]
struct Args {
    #[arg(long, default_value_t = 0.7, allow_negative_numbers = true)]
    a: f64,
    #[arg(long, default_value_t = 0.4, allow_negative_numbers = true)]
    b: f64,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    c: f64,
    /// number of plotted points
    #[arg(long, default_value_t = 10_000_000_000)]
    iters: u64,
    /// discarded transient steps
    #[arg(long, default_value_t = 500)]
    skip: u64,
    /// side of the square count grid (default: scales with --iters, at most 2048)
    #[arg(long)]
    bins: Option<usize>,
    /// Gaussian-smooth the histogram (sigma: 1.5 pixels)
    #[arg(long)]
    smooth: bool,
    /// accepted for compatibility with the Python script; the PNG is always written
    #[arg(long)]
    save: bool,
    /// directory for the output PNG
    #[arg(long, default_value = ".")]
    outdir: String,
}

#[inline(always)]
fn step(a: f64, b: f64, c: f64, x: f64, y: f64) -> (f64, f64) {
    let t = (b * x - c).abs().sqrt();
    // y - sign(x) * t, written as a select: exact, and without the multiply on the
    // serial dependency chain.
    let nx = if x > 0.0 {
        y - t
    } else if x < 0.0 {
        y + t
    } else {
        y
    };
    (nx, a - x)
}

struct Grid {
    size: usize,
    hist: Vec<i64>,
    spare: Vec<i64>, // invariant: all zero between calls
    x0: f64,
    y0: f64,
    h: f64,
    grows: u32,
}

impl Grid {
    fn new(size: usize, x0: f64, y0: f64, h: f64) -> Self {
        Self {
            size,
            hist: vec![0; size * size],
            spare: vec![0; size * size],
            x0,
            y0,
            h,
            grows: 0,
        }
    }

    /// Index bounds (ix0, ix1, iy0, iy1) of the non-zero cells.
    fn occupied_box(&self) -> (isize, isize, isize, isize) {
        let size = self.size;
        let mut col = vec![0i64; size];
        let (mut iy0, mut iy1) = (size, 0usize);
        for iy in 0..size {
            let mut row = 0i64;
            for (c, &v) in col.iter_mut().zip(&self.hist[iy * size..(iy + 1) * size]) {
                row |= v;
                *c |= v;
            }
            if row != 0 {
                if iy0 == size {
                    iy0 = iy;
                }
                iy1 = iy;
            }
        }
        let (mut ix0, mut ix1) = (size, 0usize);
        for (ix, &c) in col.iter().enumerate() {
            if c != 0 {
                if ix0 == size {
                    ix0 = ix;
                }
                ix1 = ix;
            }
        }
        (ix0 as isize, ix1 as isize, iy0 as isize, iy1 as isize)
    }

    /// Translate the occupied block into `spare` (shifted by dx, dy), clear it in `hist`.
    fn move_block(&mut self, ix0: isize, ix1: isize, iy0: isize, iy1: isize, dx: isize, dy: isize) {
        let size = self.size;
        let len = (ix1 - ix0 + 1) as usize;
        for iy in iy0..=iy1 {
            let src = iy as usize * size + ix0 as usize;
            let dst = (iy + dy) as usize * size + (ix0 + dx) as usize;
            self.spare[dst..dst + len].copy_from_slice(&self.hist[src..src + len]);
            self.hist[src..src + len].fill(0);
        }
    }

    /// Merge 2x2 blocks of the occupied block into `spare` (shifted by dx, dy), clear it in `hist`.
    fn merge_block(&mut self, ix0: isize, ix1: isize, iy0: isize, iy1: isize, dx: isize, dy: isize) {
        let size = self.size;
        for oy in iy0 / 2..=iy1 / 2 {
            let r0 = 2 * oy as usize * size;
            let r1 = r0 + size;
            let drow = (oy + dy) as usize * size;
            for ox in ix0 / 2..=ix1 / 2 {
                let sx = 2 * ox as usize;
                self.spare[drow + (ox + dx) as usize] = self.hist[r0 + sx]
                    + self.hist[r0 + sx + 1]
                    + self.hist[r1 + sx]
                    + self.hist[r1 + sx + 1];
            }
        }
        for iy in iy0..=iy1 {
            let src = iy as usize * size + ix0 as usize;
            self.hist[src..src + (ix1 - ix0 + 1) as usize].fill(0);
        }
    }

    /// Reframe so that all counted points and the point (x, y) fit.
    ///
    /// If the occupied block plus the new point fit at the current cell size the
    /// counts are only translated (no resolution loss); otherwise the cell size is
    /// doubled (2x2 merge) until they fit. The block is always centred.
    fn make_room(&mut self, x: f64, y: f64) {
        let size = self.size as isize;
        let (mut ix0, mut ix1, mut iy0, mut iy1) = self.occupied_box();
        loop {
            let px = ((x - self.x0) / self.h).floor() as isize;
            let py = ((y - self.y0) / self.h).floor() as isize;
            let lox = ix0.min(px);
            let hix = ix1.max(px);
            let loy = iy0.min(py);
            let hiy = iy1.max(py);

            if hix - lox < size && hiy - loy < size {
                // NOTE: dx, dy may be negative (block moves left/down), so they are signed.
                let dx = (size - (hix - lox + 1)) / 2 - lox;
                let dy = (size - (hiy - loy + 1)) / 2 - loy;
                self.move_block(ix0, ix1, iy0, iy1, dx, dy);
                std::mem::swap(&mut self.hist, &mut self.spare);
                self.x0 -= dx as f64 * self.h;
                self.y0 -= dy as f64 * self.h;
                return;
            }

            // Does not fit: halve the resolution, with the old block centred.
            let (mx0, mx1, my0, my1) = (ix0 / 2, ix1 / 2, iy0 / 2, iy1 / 2);
            let dx = (size - (mx1 - mx0 + 1)) / 2 - mx0;
            let dy = (size - (my1 - my0 + 1)) / 2 - my0;
            self.merge_block(ix0, ix1, iy0, iy1, dx, dy);
            std::mem::swap(&mut self.hist, &mut self.spare);
            ix0 = mx0 + dx;
            ix1 = mx1 + dx;
            iy0 = my0 + dy;
            iy1 = my1 + dy;
            self.x0 -= dx as f64 * 2.0 * self.h;
            self.y0 -= dy as f64 * 2.0 * self.h;
            self.h *= 2.0;
            self.grows += 1;
        }
    }

    /// Bin a point, clamping to the grid (only used right after a reframe / warm-up).
    fn bin_clamped(&mut self, x: f64, y: f64) {
        let max = (self.size - 1) as f64;
        let fx = ((x - self.x0) / self.h).clamp(0.0, max) as usize;
        let fy = ((y - self.y0) / self.h).clamp(0.0, max) as usize;
        self.hist[fy * self.size + fx] += 1;
    }
}

#[derive(PartialEq, Eq)]
enum Status {
    /// all requested points were binned
    Done,
    /// the point of the current step lies outside the grid (not binned yet)
    Outside,
    /// the orbit escaped (the point of the current step is not binned)
    Escaped,
}

/// Number of histogram updates kept in flight (power of two). Each point's cell is
/// prefetched when the point is computed and incremented PF_DIST points later, so the
/// cache misses of the scattered updates overlap with the sqrt-bound orbit computation.
const PF_DIST: usize = 16;

#[inline(always)]
fn prefetch(p: *const i64) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetching is only a hint; it cannot fault, whatever the address.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Hot loop: run the orbit for up to `left` more steps and bin each point until one
/// leaves the grid or the orbit escapes. Returns (status, steps still left, x, y).
/// All points counted in `left` are in `hist` when it returns.
///
/// This is deliberately a separate, non-inlined function that works on locals only:
/// the orbit is a serial dependency chain (mul -> sqrt -> add), so the speed is set by
/// its latency and anything that lengthens the chain, spills registers or adds
/// divisions to it shows up directly. `C_ZERO` drops the `- c` term from the chain when
/// c == 0, which is exact (b*x - 0.0 == b*x).
#[inline(never)]
fn fill_until_outside<const C_ZERO: bool>(
    hist: &mut [i64],
    size: usize,
    a: f64,
    b: f64,
    c: f64,
    mut left: u64,
    mut x: f64,
    mut y: f64,
    x0: f64,
    y0: f64,
    inv_h: f64,
) -> (Status, u64, f64, f64) {
    // These make the unchecked accesses below sound.
    assert!(size >= 2 && size <= 1 << 30 && hist.len() == size * size);
    let sz = size as f64;
    let mut ring = [0usize; PF_DIST]; // cell indices waiting to be incremented
    let mut queued = 0usize; // points pushed through the ring so far
    let status = loop {
        if left == 0 {
            break Status::Done;
        }
        let old_x = x;
        let d = if C_ZERO { b * old_x } else { b * old_x - c };
        let t = d.abs().sqrt();
        x = if old_x > 0.0 {
            y - t
        } else if old_x < 0.0 {
            y + t
        } else {
            y
        };
        y = a - old_x;
        if (x.abs() > ESCAPE) | (y.abs() > ESCAPE) {
            break Status::Escaped;
        }
        // h is a power of two, so multiplying by 1/h is exactly dividing by h.
        let fx = (x - x0) * inv_h;
        let fy = (y - y0) * inv_h;
        if fx >= 0.0 && fx < sz && fy >= 0.0 && fy < sz {
            // SAFETY: 0 <= fx, fy < size <= 2^30, so the casts are in range and
            // idx < size * size == hist.len(). Ring entries are always such indices.
            unsafe {
                let ix = fx.to_int_unchecked::<i32>() as usize;
                let iy = fy.to_int_unchecked::<i32>() as usize;
                let idx = iy * size + ix;
                prefetch(hist.as_ptr().add(idx));
                let slot = queued & (PF_DIST - 1);
                if queued >= PF_DIST {
                    *hist.get_unchecked_mut(*ring.get_unchecked(slot)) += 1;
                }
                *ring.get_unchecked_mut(slot) = idx;
            }
            queued += 1;
        } else {
            break Status::Outside;
        }
        left -= 1;
    };
    // Flush the updates still waiting in the ring (order does not matter).
    for j in queued - queued.min(PF_DIST)..queued {
        hist[ring[j & (PF_DIST - 1)]] += 1;
    }
    (status, left, x, y)
}

struct Orbit {
    grid: Grid,
    escaped: bool,
}

/// Single-pass visit-count histogram on a size x size dyadic grid.
/// Returns None if the orbit escaped before producing any plotted point.
fn orbit_histogram(a: f64, b: f64, c: f64, n: u64, skip: u64, size: usize) -> Option<Orbit> {
    let total = n + skip;
    let mut buf_x = vec![0.0f64; WARMUP];
    let mut buf_y = vec![0.0f64; WARMUP];
    let mut nbuf = 0usize;
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let mut escaped = false;
    let mut i: u64 = 0;

    while i < total && nbuf < WARMUP {
        (x, y) = step(a, b, c, x, y);
        if x.abs() > ESCAPE || y.abs() > ESCAPE {
            escaped = true;
            break;
        }
        if i >= skip {
            buf_x[nbuf] = x;
            buf_y[nbuf] = y;
            nbuf += 1;
        }
        i += 1;
    }
    if nbuf == 0 {
        return None;
    }

    // Starting grid: smallest cell size that fits the buffered points, centred.
    let fold = |v: &[f64], init: f64, f: fn(f64, f64) -> f64| v.iter().copied().fold(init, f);
    let xmin = fold(&buf_x[..nbuf], f64::INFINITY, f64::min);
    let xmax = fold(&buf_x[..nbuf], f64::NEG_INFINITY, f64::max);
    let ymin = fold(&buf_y[..nbuf], f64::INFINITY, f64::min);
    let ymax = fold(&buf_y[..nbuf], f64::NEG_INFINITY, f64::max);
    let extent = (xmax - xmin).max(ymax - ymin);
    let mut h = H_MIN;
    while extent / h >= (size - 1) as f64 {
        h *= 2.0;
    }
    let s = size as isize;
    let lo_x = (xmin / h).floor() as isize;
    let lo_y = (ymin / h).floor() as isize;
    let wid_x = (xmax / h).floor() as isize - lo_x + 1;
    let wid_y = (ymax / h).floor() as isize - lo_y + 1;
    let x0 = (lo_x - (s - wid_x) / 2) as f64 * h;
    let y0 = (lo_y - (s - wid_y) / 2) as f64 * h;

    let mut grid = Grid::new(size, x0, y0, h);
    for k in 0..nbuf {
        grid.bin_clamped(buf_x[k], buf_y[k]);
    }

    // Main loop: the hot loop bins points directly; reframe only when one falls outside.
    // (After warm-up, i >= skip always holds, so every remaining step is plotted.)
    let mut left = total - i;
    while !escaped && left > 0 {
        let inv_h = 1.0 / grid.h;
        let (status, rem, nx, ny) = if c == 0.0 {
            fill_until_outside::<true>(
                &mut grid.hist, size, a, b, c, left, x, y, grid.x0, grid.y0, inv_h,
            )
        } else {
            fill_until_outside::<false>(
                &mut grid.hist, size, a, b, c, left, x, y, grid.x0, grid.y0, inv_h,
            )
        };
        left = rem;
        x = nx;
        y = ny;
        match status {
            Status::Done => break,
            Status::Escaped => escaped = true,
            Status::Outside => {
                grid.make_room(x, y);
                grid.bin_clamped(x, y);
                left -= 1;
            }
        }
    }
    Some(Orbit { grid, escaped })
}

/// Drop the empty margin. Returns (counts, width, height, extent = [xmin, xmax, ymin, ymax]).
fn crop_to_occupied(g: &Grid) -> (Vec<i64>, usize, usize, [f64; 4]) {
    let size = g.size;
    let (mut r0, mut r1, mut c0, mut c1) = (size, 0usize, size, 0usize);
    for iy in 0..size {
        for ix in 0..size {
            if g.hist[iy * size + ix] != 0 {
                r0 = r0.min(iy);
                r1 = r1.max(iy);
                c0 = c0.min(ix);
                c1 = c1.max(ix);
            }
        }
    }
    let (w, hgt) = (c1 - c0 + 1, r1 - r0 + 1);
    let mut sub = Vec::with_capacity(w * hgt);
    for iy in r0..=r1 {
        sub.extend_from_slice(&g.hist[iy * size + c0..iy * size + c1 + 1]);
    }
    let extent = [
        g.x0 + c0 as f64 * g.h,
        g.x0 + (c1 + 1) as f64 * g.h,
        g.y0 + r0 as f64 * g.h,
        g.y0 + (r1 + 1) as f64 * g.h,
    ];
    (sub, w, hgt, extent)
}

/// Separable Gaussian filter (zero padding), sigma in pixels, on a w x h array.
fn smooth_histogram(data: &[i64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma).ceil() as isize;
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= sum);

    let src: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let mut tmp = vec![0.0f64; w * h];
    let mut out = vec![0.0f64; w * h];

    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &wt) in kernel.iter().enumerate() {
                let sx = x as isize + k as isize - radius;
                if sx >= 0 && (sx as usize) < w {
                    acc += wt * src[y * w + sx as usize];
                }
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &wt) in kernel.iter().enumerate() {
                let sy = y as isize + k as isize - radius;
                if sy >= 0 && (sy as usize) < h {
                    acc += wt * tmp[sy as usize * w + x];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Polynomial approximation of matplotlib's "magma" colormap, t in [0, 1].
fn magma(t: f64) -> [u8; 3] {
    const C: [[f64; 3]; 7] = [
        [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
        [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
        [8.353717279216625, -3.577719514958484, 0.3144679030132573],
        [-27.66873308576866, 14.26473078096533, -13.64921318813922],
        [52.17613981234068, -27.94360607168351, 12.94416944238394],
        [-50.76852536473588, 29.04658282127291, 4.23415299384598],
        [18.65570506591883, -11.48977351997711, -5.601961508734096],
    ];
    let t = t.clamp(0.0, 1.0);
    let mut rgb = [0u8; 3];
    for ch in 0..3 {
        let mut v = 0.0;
        for coef in C.iter().rev() {
            v = v * t + coef[ch];
        }
        rgb[ch] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    rgb
}

fn main() {
    let args = Args::parse();
    if !(args.a.is_finite() && args.b.is_finite() && args.c.is_finite()) {
        eprintln!("error: --a, --b and --c must be finite numbers");
        std::process::exit(2);
    }
    if args.iters == 0 {
        eprintln!("error: --iters must be a positive integer");
        std::process::exit(2);
    }
    if args.bins == Some(0) {
        eprintln!("error: --bins must be a positive integer");
        std::process::exit(2);
    }

    let mut size = args.bins.unwrap_or_else(|| {
        let side = ((args.iters as f64 / 12.0).sqrt() as usize).clamp(200, 1500);
        (2 * side).min(2048)
    });
    size += size % 2; // the dyadic merge needs an even grid side

    let started = std::time::Instant::now();
    let Some(orbit) = orbit_histogram(args.a, args.b, args.c, args.iters, args.skip, size) else {
        eprintln!("The orbit escaped before producing any plotted points.");
        std::process::exit(1);
    };
    let seconds = started.elapsed().as_secs_f64();
    let grid = &orbit.grid;
    let points: i64 = grid.hist.iter().sum();

    let (hist, w, h, extent) = crop_to_occupied(grid);
    println!(
        "single pass: {} points, grid {}x{}, cell {:.4e}, {} grid doublings{} ({:.2}s)",
        points,
        size,
        size,
        grid.h,
        grid.grows,
        if orbit.escaped { ", orbit escaped" } else { "" },
        seconds
    );
    println!(
        "histogram {}x{}, x {:.6}..{:.6}, y {:.6}..{:.6}",
        w, h, extent[0], extent[1], extent[2], extent[3]
    );

    // log1p density, optionally Gaussian-smoothed first (as in the Python script).
    let density: Vec<f64> = if args.smooth {
        smooth_histogram(&hist, w, h, 1.5)
    } else {
        hist.iter().map(|&v| v as f64).collect()
    };
    let logd: Vec<f64> = density.iter().map(|d| d.ln_1p()).collect();
    let vmin = logd.iter().copied().fold(f64::INFINITY, f64::min);
    let vmax = logd.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = if vmax > vmin { vmax - vmin } else { 1.0 };

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (px, py, pixel) in img.enumerate_pixels_mut() {
        // Row 0 of the histogram is the lowest y, so flip for a standard orientation.
        let v = logd[(h - 1 - py as usize) * w + px as usize];
        *pixel = image::Rgb(magma((v - vmin) / span));
    }

    if let Err(e) = std::fs::create_dir_all(&args.outdir) {
        eprintln!("cannot create {}: {e}", args.outdir);
        std::process::exit(1);
    }
    let path = Path::new(&args.outdir).join("hopalong_orbit.png");
    match img.save(&path) {
        Ok(()) => println!("saved {}", path.display()),
        Err(e) => {
            eprintln!("cannot save {}: {e}", path.display());
            std::process::exit(1);
        }
    }
}
