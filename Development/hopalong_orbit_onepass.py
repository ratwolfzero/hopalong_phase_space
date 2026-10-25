"""Render the classic Hopalong orbit as an on-the-fly density histogram (one pass).

No survey pass is needed. The histogram lives on a dyadic grid: whenever a
point leaves the grid, the block of occupied counts is shifted if it still fits
at the current cell size, and otherwise merged 2x2 into half the resolution
(the count pyramid's marginalisation property), so the counts stay exact.
Memory use is independent of the number of iterations. The first few thousand
points are buffered (not recomputed) only to pick the starting cell size;
correctness does not depend on it. Use --smooth to display a Gaussian-smoothed
histogram.

Usage:
    python hopalong_orbit_onepass.py
    python hopalong_orbit_onepass.py --smooth
    python hopalong_orbit_onepass.py --save
    python hopalong_orbit_onepass.py --a 1.1 --b 0.5 --c 0.3 --iters 2000000 --save
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

ESCAPE = 1e6
WARMUP = 4096  # points buffered to choose the starting cell size
H_MIN = 2.0 ** -32  # finest cell size; all cell sizes are H_MIN * 2**k


@njit(cache=True)
def fill_until_outside(hist, a, b, c, i, total, skip, x, y, x0, y0, inv_h):
    """Run the orbit from step i and bin points until one leaves the grid.

    Returns (status, i, x, y). status: 0 = finished, 1 = the point of step i
    lies outside the grid (not binned yet), 2 = the orbit escaped.
    """
    size = hist.shape[0]
    while i < total:
        old_x = x
        delta = b * old_x - c
        if old_x > 0.0:
            sign_x = 1.0
        elif old_x < 0.0:
            sign_x = -1.0
        else:
            sign_x = 0.0
        x = y - sign_x * np.sqrt(abs(delta))
        y = a - old_x
        if abs(x) > ESCAPE or abs(y) > ESCAPE:
            return 2, i, x, y
        if i >= skip:
            fx = (x - x0) * inv_h
            fy = (y - y0) * inv_h
            if 0.0 <= fx < size and 0.0 <= fy < size:
                hist[int(fy), int(fx)] += 1
            else:
                return 1, i, x, y
        i += 1
    return 0, i, x, y


@njit(cache=True)
def occupied_box(hist):
    """Index bounds (ix0, ix1, iy0, iy1) of the non-zero cells; one cheap pass."""
    size = hist.shape[0]
    col = np.zeros(size, dtype=np.int64)
    iy0 = size
    iy1 = -1
    for iy in range(size):
        row = np.int64(0)
        for ix in range(size):
            v = hist[iy, ix]
            row |= v
            col[ix] |= v
        if row != 0:
            if iy0 == size:
                iy0 = iy
            iy1 = iy
    ix0 = size
    ix1 = -1
    for ix in range(size):
        if col[ix] != 0:
            if ix0 == size:
                ix0 = ix
            ix1 = ix
    return ix0, ix1, iy0, iy1


@njit(cache=True)
def move_block(src, dst, ix0, ix1, iy0, iy1, dx, dy):
    """Translate the occupied block of src into the (all-zero) dst; clear it in src."""
    for iy in range(iy0, iy1 + 1):
        dst[iy + dy, ix0 + dx:ix1 + 1 + dx] = src[iy, ix0:ix1 + 1]
    src[iy0:iy1 + 1, ix0:ix1 + 1] = 0


@njit(cache=True)
def merge_block(src, dst, ix0, ix1, iy0, iy1, dx, dy):
    """Merge 2x2 blocks of the occupied block into dst, translated; clear it in src."""
    for oy in range(iy0 // 2, iy1 // 2 + 1):
        for ox in range(ix0 // 2, ix1 // 2 + 1):
            dst[oy + dy, ox + dx] = (src[2 * oy, 2 * ox] + src[2 * oy, 2 * ox + 1]
                                     + src[2 * oy + 1, 2 * ox]
                                     + src[2 * oy + 1, 2 * ox + 1])
    src[iy0:iy1 + 1, ix0:ix1 + 1] = 0


@njit(cache=True)
def make_room(hist, spare, x0, y0, h, x, y):
    """Reframe the grid so that all counted points and the point (x, y) fit.

    If the occupied block plus the new point fit at the current cell size, the
    counts are only translated, which loses no resolution. Otherwise the cell
    size is doubled (2x2 merge) until they fit. The block is always centred, so
    there is room on both sides. Every cell size is a power of two times H_MIN,
    so old and new cell boundaries stay aligned and the counts stay exact.
    `spare` is an all-zero array of the same shape; the two arrays are swapped.
    Returns (hist, spare, x0, y0, h, doublings).
    """
    size = hist.shape[0]
    ix0, ix1, iy0, iy1 = occupied_box(hist)
    grows = 0
    while True:
        px = int(np.floor((x - x0) / h))
        py = int(np.floor((y - y0) / h))
        lox = min(ix0, px)
        hix = max(ix1, px)
        loy = min(iy0, py)
        hiy = max(iy1, py)
        if hix - lox < size and hiy - loy < size:
            dx = (size - (hix - lox + 1)) // 2 - lox
            dy = (size - (hiy - loy + 1)) // 2 - loy
            move_block(hist, spare, ix0, ix1, iy0, iy1, dx, dy)
            return spare, hist, x0 - dx * h, y0 - dy * h, h, grows
        # Does not fit: halve the resolution, with the old block centred.
        dx = (size - (ix1 // 2 - ix0 // 2 + 1)) // 2 - ix0 // 2
        dy = (size - (iy1 // 2 - iy0 // 2 + 1)) // 2 - iy0 // 2
        merge_block(hist, spare, ix0, ix1, iy0, iy1, dx, dy)
        tmp = hist
        hist = spare
        spare = tmp
        ix0 = ix0 // 2 + dx
        ix1 = ix1 // 2 + dx
        iy0 = iy0 // 2 + dy
        iy1 = iy1 // 2 + dy
        x0 -= dx * 2.0 * h
        y0 -= dy * 2.0 * h
        h *= 2.0
        grows += 1


@njit(cache=True)
def orbit_histogram(a, b, c, n, skip, size):
    """Single-pass visit-count histogram on a size x size dyadic grid.

    The first WARMUP plotted points are buffered to choose the starting cell
    size; from then on the grid adapts whenever a point falls outside. size
    must be even. Returns (hist, x0, y0, h, grows, escaped) with (x0, y0) the
    lower-left corner of the grid and h the final cell size.
    """
    total = n + skip
    buf_x = np.empty(WARMUP)
    buf_y = np.empty(WARMUP)
    nbuf = 0
    i = 0
    x = 0.0
    y = 0.0
    escaped = False
    while i < total and nbuf < WARMUP:
        old_x = x
        delta = b * old_x - c
        if old_x > 0.0:
            sign_x = 1.0
        elif old_x < 0.0:
            sign_x = -1.0
        else:
            sign_x = 0.0
        x = y - sign_x * np.sqrt(abs(delta))
        y = a - old_x
        if abs(x) > ESCAPE or abs(y) > ESCAPE:
            escaped = True
            break
        if i >= skip:
            buf_x[nbuf] = x
            buf_y[nbuf] = y
            nbuf += 1
        i += 1

    hist = np.zeros((size, size), dtype=np.int64)
    if nbuf == 0:
        return hist, 0.0, 0.0, H_MIN, 0, escaped

    # Starting grid: smallest cell size that fits the buffered points, centred.
    xmin = buf_x[:nbuf].min()
    xmax = buf_x[:nbuf].max()
    ymin = buf_y[:nbuf].min()
    ymax = buf_y[:nbuf].max()
    extent = max(xmax - xmin, ymax - ymin)
    h = H_MIN
    while extent / h >= size - 1:
        h *= 2.0
    lo_x = int(np.floor(xmin / h))
    lo_y = int(np.floor(ymin / h))
    wid_x = int(np.floor(xmax / h)) - lo_x + 1
    wid_y = int(np.floor(ymax / h)) - lo_y + 1
    x0 = (lo_x - (size - wid_x) // 2) * h
    y0 = (lo_y - (size - wid_y) // 2) * h
    for k in range(nbuf):
        fx = min(max(int((buf_x[k] - x0) / h), 0), size - 1)
        fy = min(max(int((buf_y[k] - y0) / h), 0), size - 1)
        hist[fy, fx] += 1

    spare = np.zeros((size, size), dtype=np.int64)
    grows = 0
    while not escaped:
        status, i, x, y = fill_until_outside(
            hist, a, b, c, i, total, skip, x, y, x0, y0, 1.0 / h)
        if status == 2:
            escaped = True
        if status != 1:
            break
        hist, spare, x0, y0, h, g = make_room(hist, spare, x0, y0, h, x, y)
        grows += g
        fx = min(max(int((x - x0) / h), 0), size - 1)
        fy = min(max(int((y - y0) / h), 0), size - 1)
        hist[fy, fx] += 1
        i += 1
    return hist, x0, y0, h, grows, escaped


def crop_to_occupied(hist, x0, y0, h):
    """Drop the empty margin; return the sub-array and its imshow extent."""
    rows = np.flatnonzero(hist.any(axis=1))
    cols = np.flatnonzero(hist.any(axis=0))
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    sub = hist[r0:r1 + 1, c0:c1 + 1]
    extent = (x0 + c0 * h, x0 + (c1 + 1) * h,
              y0 + r0 * h, y0 + (r1 + 1) * h)
    return sub, extent


def smooth_histogram(histogram, sigma=1.5):
    """Apply a separable Gaussian filter, with sigma measured in pixels."""
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = histogram.astype(np.float64)
    for axis in (0, 1):
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        padded = np.pad(smoothed, pad_width, mode="constant")
        filtered = np.zeros_like(smoothed)
        for index, weight in enumerate(kernel):
            source = [slice(None), slice(None)]
            source[axis] = slice(index, index + smoothed.shape[axis])
            filtered += weight * padded[tuple(source)]
        smoothed = filtered
    return smoothed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=float, default=0.7)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--c", type=float, default=0)
    parser.add_argument("--iters", type=int, default=10_000_000_000,
                        help="number of plotted points (default: 10_000_000_000)")
    parser.add_argument("--skip", type=int, default=500,
                        help="discarded transient steps (default: 500)")
    parser.add_argument("--bins", type=int,
                        help="side of the square count grid; larger values usually "
                             "give finer cells but use memory proportional to bins^2 "
                             "(default: scales with --iters, at most 2048)")
    parser.add_argument("--smooth", action="store_true",
                        help="Gaussian-smooth the histogram (sigma: 1.5 pixels)")
    parser.add_argument("--save", action="store_true",
                        help="save hopalong_orbit.png instead of showing it")
    parser.add_argument("--outdir", default=".",
                        help="directory for the output PNG (default: current directory)")
    args = parser.parse_args(argv)
    if not all(np.isfinite(value) for value in (args.a, args.b, args.c)):
        parser.error("--a, --b and --c must be finite numbers")
    if args.iters <= 0:
        parser.error("--iters must be a positive integer")
    if args.skip < 0:
        parser.error("--skip must be non-negative")
    if args.bins is not None and args.bins <= 0:
        parser.error("--bins must be a positive integer")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.bins is not None:
        size = args.bins
    else:
        size = min(2 * int(np.clip(np.sqrt(args.iters / 12), 200, 1500)), 2048)
    size += size % 2  # the dyadic merge needs an even grid side

    started = time.perf_counter()
    hist, x0, y0, h, grows, escaped = orbit_histogram(
        args.a, args.b, args.c, args.iters, args.skip, size)
    seconds = time.perf_counter() - started
    points = int(hist.sum())
    if points == 0:
        raise SystemExit(
            "The orbit escaped before producing any plotted points.")

    histogram, extent = crop_to_occupied(hist, x0, y0, h)
    nby, nbx = histogram.shape
    print(f"single pass: {points:,} points, grid {size}x{size}, "
          f"cell {h:.4g}, {grows} grid doublings"
          f"{', orbit escaped' if escaped else ''} ({seconds:.2f}s)")
    print(f"histogram {nbx}x{nby}, "
          f"x {extent[0]:.6g}..{extent[1]:.6g}, y {extent[2]:.6g}..{extent[3]:.6g}")

    density = smooth_histogram(histogram) if args.smooth else histogram
    density_label = ("log Gaussian-smoothed visit density" if args.smooth
                     else "log visit density")
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(np.log1p(density), origin="lower", cmap="magma",
              extent=extent, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(f"Hopalong orbit: a={args.a}, b={args.b}, c={args.c}\n"
                 f"{points:,} points, {density_label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if args.save:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, "hopalong_orbit.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        print("saved", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
