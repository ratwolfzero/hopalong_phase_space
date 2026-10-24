"""Render the classic Hopalong orbit as an on-the-fly density histogram (one pass).

No survey pass is needed. The histogram lives on a dyadic grid: whenever a
point leaves the grid, the cell size is doubled and 2x2 blocks of counts are
merged (the count pyramid's marginalisation property), so the counts stay
exact. Memory use is independent of the number of iterations. A tiny pilot
orbit (a few thousand steps) only picks the starting size; correctness does
not depend on it. Use --smooth to display a Gaussian-smoothed histogram.

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
PILOT_STEPS = 2000


@njit(cache=True)
def pilot_bounds(a, b, c, steps):
    """Bounds of the first `steps` orbit points; only used for the start size."""
    x = 0.0
    y = 0.0
    xmin = 1e300
    xmax = -1e300
    ymin = 1e300
    ymax = -1e300
    for _ in range(steps):
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
            break
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    return xmin, xmax, ymin, ymax


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
def coarsen(hist, left, down):
    """Merge 2x2 blocks into half the resolution on a grid of twice the span.

    The old grid ends up in one half of the new one, on the side opposite to
    the direction in which the grid is extended.
    """
    size = hist.shape[0]
    half = size // 2
    ox = half if left else 0
    oy = half if down else 0
    out = np.zeros_like(hist)
    for iy in range(size):
        for ix in range(size):
            v = hist[iy, ix]
            if v:
                out[oy + (iy >> 1), ox + (ix >> 1)] += v
    return out


@njit(cache=True)
def orbit_histogram(a, b, c, n, skip, size, cx, cy, span0):
    """Single-pass visit-count histogram on a size x size dyadic grid.

    The grid starts with span `span0` centred on (cx, cy) and doubles its span
    (halving the resolution) whenever a point falls outside. size must be even.
    Returns (hist, x0, y0, h, grows, escaped) with (x0, y0) the lower-left
    corner of the grid and h the final cell size.
    """
    hist = np.zeros((size, size), dtype=np.int64)
    h = span0 / size
    x0 = cx - 0.5 * span0
    y0 = cy - 0.5 * span0
    total = n + skip
    i = 0
    x = 0.0
    y = 0.0
    grows = 0
    escaped = False
    while True:
        status, i, x, y = fill_until_outside(
            hist, a, b, c, i, total, skip, x, y, x0, y0, 1.0 / h)
        if status == 2:
            escaped = True
        if status != 1:
            break
        fx = (x - x0) / h
        fy = (y - y0) / h
        while not (0.0 <= fx < size and 0.0 <= fy < size):
            left = fx < 0.0
            down = fy < 0.0
            hist = coarsen(hist, left, down)
            if left:
                x0 -= size * h
            if down:
                y0 -= size * h
            h *= 2.0
            fx = (x - x0) / h
            fy = (y - y0) / h
            grows += 1
        hist[int(fy), int(fx)] += 1
        i += 1
    return hist, x0, y0, h, grows, escaped


def start_grid(a, b, c):
    """Centre and span of the starting grid, from a tiny pilot orbit."""
    xmin, xmax, ymin, ymax = pilot_bounds(a, b, c, PILOT_STEPS)
    if xmax < xmin:  # escaped immediately
        return 0.0, 0.0, 4.0
    span = max(xmax - xmin, ymax - ymin) * 1.05
    if not np.isfinite(span) or span <= 0.0:
        span = 4.0
    return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), span


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
                        help="side of the square count grid (default: twice the "
                             "old heuristic, at most 2048). Because the grid cell "
                             "size only doubles, the attractor ends up covering "
                             "roughly 40-100%% of the grid side.")
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
    cx, cy, span0 = start_grid(args.a, args.b, args.c)
    hist, x0, y0, h, grows, escaped = orbit_histogram(
        args.a, args.b, args.c, args.iters, args.skip, size, cx, cy, span0)
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
