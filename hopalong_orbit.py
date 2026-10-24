"""Render the classic Hopalong orbit as an on-the-fly density histogram.

The first pass finds the orbit bounds without storing points. The second pass
repeats the orbit and fills the histogram directly. Memory use is therefore
independent of the number of iterations. Use --smooth to display a Gaussian-
smoothed version of the histogram.

Usage:
    python hopalong_orbit.py
    python hopalong_orbit.py --smooth
    python hopalong_orbit.py --save
    python hopalong_orbit.py --a 1.1 --b 0.5 --c 0.3 --iters 2000000 --save
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

ESCAPE = 1e6


@njit(cache=True)
def orbit_extent(a, b, c, n, skip):
    """Return the exact bounds and count for the orbit from (0, 0)."""
    x = 0.0
    y = 0.0
    xmin = 1e300
    xmax = -1e300
    ymin = 1e300
    ymax = -1e300
    count = 0

    for i in range(n + skip):
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
        if i >= skip:
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
            count += 1

    return xmin, xmax, ymin, ymax, count


@njit(cache=True)
def orbit_histogram(a, b, c, n, skip, xlo, xhi, ylo, yhi, nbx, nby):
    """Fill a 2D visit-count histogram while recomputing the orbit."""
    histogram = np.zeros((nby, nbx), dtype=np.int64)
    scale_x = nbx / (xhi - xlo)
    scale_y = nby / (yhi - ylo)
    x = 0.0
    y = 0.0
    count = 0
    outside = 0

    for i in range(n + skip):
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
        if i >= skip:
            ix = int((x - xlo) * scale_x)
            iy = int((y - ylo) * scale_y)
            if 0 <= ix < nbx and 0 <= iy < nby:
                histogram[iy, ix] += 1
            else:
                outside += 1
            count += 1

    return histogram, count, outside


def histogram_layout(bounds, bins, n):
    xmin, xmax, ymin, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    padding_x = max(width * 1e-6, max(abs(xmin), abs(xmax), 1.0) * 1e-12)
    padding_y = max(height * 1e-6, max(abs(ymin), abs(ymax), 1.0) * 1e-12)
    extent = (xmin - padding_x, xmax + padding_x,
              ymin - padding_y, ymax + padding_y)
    longest = bins if bins is not None else int(
        np.clip(np.sqrt(n / 12), 200, 1500))
    span_x = extent[1] - extent[0]
    span_y = extent[3] - extent[2]
    longest_span = max(span_x, span_y)
    shape = (max(int(longest * span_x / longest_span), 1),
             max(int(longest * span_y / longest_span), 1))
    return extent, shape


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
    parser.add_argument("--a", type=float, default=1.7)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=10_000_000,
                        help="number of plotted points (default: 10,000,000)")
    parser.add_argument("--skip", type=int, default=500,
                        help="discarded transient steps (default: 500)")
    parser.add_argument("--bins", type=int,
                        help="pixels on the longer side (default: scales with --iters)")
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
    started = time.perf_counter()
    bounds_and_count = orbit_extent(
        args.a, args.b, args.c, args.iters, args.skip)
    survey_seconds = time.perf_counter() - started
    bounds = bounds_and_count[:4]
    surveyed_points = bounds_and_count[4]
    if surveyed_points == 0:
        raise SystemExit(
            "The orbit escaped before producing any plotted points.")

    extent, (nbx, nby) = histogram_layout(bounds, args.bins, args.iters)
    started = time.perf_counter()
    histogram, points, outside = orbit_histogram(
        args.a, args.b, args.c, args.iters, args.skip, *extent, nbx, nby)
    render_seconds = time.perf_counter() - started
    if points != surveyed_points:
        raise RuntimeError(
            "The orbit count changed between the survey and histogram passes.")

    print(f"pass 1: {surveyed_points:,} points, bounds "
          f"x {bounds[0]:.6g}..{bounds[1]:.6g}, y {bounds[2]:.6g}..{bounds[3]:.6g} "
          f"({survey_seconds:.2f}s)")
    print(f"pass 2: histogram {nbx}x{nby}, {points:,} points, "
          f"{outside:,} outside bounds ({render_seconds:.2f}s)")

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
