"""Hopalong (Barry Martin) map: orbit, Lyapunov-exponent map and orbit portrait.

The map is
    x' = y - sign(x) * sqrt(|b*x - c|)
    y' = a - x

It preserves area, so it has no attractor: what is usually drawn is the
footprint of ONE orbit, and the picture depends on the starting point. This
script produces three views of the same map:

    orbit     one orbit from (0,0), drawn as a density image (the classic picture)
    map       a Lyapunov exponent for every starting point on a grid:
              ~0 = regular motion, > 0 = chaotic motion
    portrait  many orbits from a line of starting points, coloured by that test
    header    a wide (16:9) version of the portrait, e.g. for a blog header

The run has two passes:

    pass 1 (survey, fast)  finds the extents: the natural length scale
        L of the parameters, the centre of the structure (the map's fixed
        point), the box of the orbit from (0,0), the plot window of the map,
        a line of starting points for the portrait and the range and resolution
        of the density image
    pass 2 (render)  draws the views with those extents, so the framing follows
        a, b, c and the number of points

The map keeps its form under (x, y, a, b, c) -> (s*x, s*y, s*a, s*b, s*s*c), so
the framing is built from the length scale L = max(|a|, |b|, sqrt|c|) and
scales exactly with it. The extent of the orbit from (0,0) is measured over the
full run, because it can grow when the run is long enough.

The numerical loops run in numba; plotting uses matplotlib.

Usage:
    python hopalong_phase_space.py                      # all three, shown on screen
    python hopalong_phase_space.py --save               # all three, written as PNG
    python hopalong_phase_space.py --which map --grid 800 --lyap-iters 20000 --save
    python hopalong_phase_space.py --a 1.1 --b 0.5 --c 0.3 --iters 2000000
    python hopalong_phase_space.py --which header --save
    python hopalong_phase_space.py --a 2 --b 1 --survey-only    # pass 1 only
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numba import njit, prange

ESCAPE = 1e6   # an orbit beyond this radius counts as escaped


# --------------------------------------------------------------------------
# numerical core (numba)
# --------------------------------------------------------------------------
@njit(cache=True)
def orbit(a, b, c, n, x0, y0, skip):
    """n points of the orbit from (x0, y0) after discarding `skip` steps.

    Returns fewer than n points if the orbit escapes to infinity."""
    xs = np.empty(n)
    ys = np.empty(n)
    x, y = x0, y0
    count = 0
    for i in range(n + skip):
        x, y = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        if abs(x) > ESCAPE or abs(y) > ESCAPE:
            break
        if i >= skip:
            xs[count] = x
            ys[count] = y
            count += 1
    return xs[:count], ys[:count]


@njit(cache=True)
def lyapunov_orbit(a, b, c, x, y, n, skip):
    """Largest finite-time Lyapunov exponent of the orbit starting at (x, y).

    A tangent vector is pushed forward with the Jacobian of the map and
    renormalised every step; the mean log of the stretching is the exponent.
    The first `skip` steps are discarded so the vector can align first.
    Returns NaN if the orbit escapes to infinity."""
    vx, vy = 1.0, 0.0
    total = 0.0
    for i in range(n + skip):
        d = b * x - c
        r = max(np.sqrt(abs(d)), 1e-12)               # guard the sqrt singularity
        dfdx = -np.sign(x) * b * np.sign(d) / (2.0 * r)
        x, y = y - np.sign(x) * np.sqrt(abs(d)), a - x
        vx, vy = dfdx * vx + vy, -vx                  # tangent vector * Jacobian
        nrm = np.sqrt(vx * vx + vy * vy)
        vx, vy = vx / nrm, vy / nrm
        if i >= skip:
            total += np.log(nrm)
        if abs(x) > ESCAPE or abs(y) > ESCAPE:
            return np.nan
    return total / n


@njit(parallel=True, cache=True)
def lyapunov_grid(a, b, c, xs, ys, n, skip):
    """Exponent for every starting point (xs[i], ys[j]); rows follow ys."""
    out = np.empty((len(ys), len(xs)))
    for j in prange(len(ys)):
        for i in range(len(xs)):
            out[j, i] = lyapunov_orbit(a, b, c, xs[i], ys[j], n, skip)
    return out


@njit(cache=True)
def orbit_density(a, b, c, n, skip, xlo, xhi, ylo, yhi, nbx, nby):
    """Density histogram and exponent of the orbit from (0,0), in one pass.

    The histogram is filled while the orbit runs, so no point is stored and
    the memory use does not depend on n. Returns (counts[nby, nbx], exponent,
    points counted, points outside the histogram range)."""
    H = np.zeros((nby, nbx), dtype=np.int64)
    sx = nbx / (xhi - xlo)
    sy = nby / (yhi - ylo)
    x, y = 0.0, 0.0
    vx, vy = 1.0, 0.0
    total = 0.0
    count = 0
    outside = 0
    for i in range(n + skip):
        d = b * x - c
        sg = np.sign(x)
        r = max(np.sqrt(abs(d)), 1e-12)
        dfdx = -sg * b * np.sign(d) / (2.0 * r)
        x, y = y - sg * np.sqrt(abs(d)), a - x
        vx, vy = dfdx * vx + vy, -vx
        nrm = np.sqrt(vx * vx + vy * vy)
        vx, vy = vx / nrm, vy / nrm
        if abs(x) > ESCAPE or abs(y) > ESCAPE:
            break
        if i >= skip:
            total += np.log(nrm)
            count += 1
            ix = int((x - xlo) * sx)
            iy = int((y - ylo) * sy)
            if 0 <= ix < nbx and 0 <= iy < nby:
                H[iy, ix] += 1
            else:
                outside += 1
    lam = total / count if count > 0 else np.nan
    return H, lam, count, outside


@njit(cache=True)
def orbit_extent(a, b, c, n, skip):
    """Exact bounding box of the orbit from (0,0) over n points, without storing it.

    Returns (xmin, xmax, ymin, ymax, points counted)."""
    x = y = 0.0
    xmin = ymin = 1e300
    xmax = ymax = -1e300
    count = 0
    for i in range(n + skip):
        x, y = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        if abs(x) > ESCAPE or abs(y) > ESCAPE:
            break
        if i >= skip:
            count += 1
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
    return xmin, xmax, ymin, ymax, count


# --------------------------------------------------------------------------
# pass 1: survey
# --------------------------------------------------------------------------
@dataclass
class Layout:
    """Everything pass 1 determines."""
    scale: float            # natural length scale L = max(|a|, |b|, sqrt|c|)
    centre: tuple           # centre of the structure (x, y)
    centre_kind: str        # how the centre was found
    orbit_box: tuple        # (xmin, xmax, ymin, ymax) of the orbit from (0,0)
    hist_extent: tuple      # (xlo, xhi, ylo, yhi) of the density image
    bins: tuple             # (nbx, nby) of the density image
    window: tuple           # ((x_lo, x_hi), (y_lo, y_hi)) of the Lyapunov map
    line: tuple             # ((x, y), (x, y)) starting points of the portrait
    survey_points: int      # points counted in the orbit box
    survey_seconds: float = 0.0


def natural_scale(a, b, c):
    """Length scale of the map.

    The map keeps its form under (x, y, a, b, c) -> (s*x, s*y, s*a, s*b, s*s*c),
    so a and b behave like lengths and c like a length squared. Orbits scale
    with L, and the Lyapunov exponent does not change at all."""
    scale = max(abs(a), abs(b), np.sqrt(abs(c)))
    return scale if scale > 0 else 1.0


def fixed_points(a, b, c, n=40001):
    """Real fixed points of the map as (x, y, trace) triples.

    They solve 2x - a + sign(x)*sqrt(|b*x - c|) = 0 with y = a - x. The Jacobian
    has determinant 1, so a point is elliptic (linearly stable, typically the centre
    of an island) when |trace| < 2 and hyperbolic otherwise."""
    reach = 10.0 * natural_scale(a, b, c)

    def f(x):
        return 2.0 * x - a + np.sign(x) * np.sqrt(np.abs(b * x - c))

    roots = []
    for lo, hi in ((-reach, 0.0), (0.0, reach)):     # f may jump at x = 0, so split there
        xs = np.linspace(lo, hi, n)[1:-1] if lo == 0.0 else np.linspace(lo, hi, n)[:-1]
        if hi == 0.0:
            xs = xs[:-1]
        fx = f(xs)
        for k in np.nonzero(np.sign(fx[:-1]) * np.sign(fx[1:]) < 0)[0]:
            left, right = xs[k], xs[k + 1]
            for _ in range(80):                       # bisection
                mid = 0.5 * (left + right)
                if np.sign(f(mid)) == np.sign(f(left)):
                    left = mid
                else:
                    right = mid
            roots.append(0.5 * (left + right))
    out = []
    for x in roots:
        d = b * x - c
        if d != 0.0:
            out.append((x, a - x, -np.sign(x) * b * np.sign(d) / (2.0 * np.sqrt(abs(d)))))
    return out


def survey(params, args):
    """Pass 1: find the extents of all views. Returns a Layout, or None if the
    orbit from (0,0) escapes."""
    a, b, c = params
    t0 = time.time()
    scale = natural_scale(a, b, c)

    # exact box of the orbit from (0,0) over the full run (nothing is stored)
    xmin, xmax, ymin, ymax, n_box = orbit_extent(a, b, c, args.iters, args.skip)
    if n_box < 1000:
        return None
    box = (xmin, xmax, ymin, ymax)

    # density image range: the box itself, so every point falls inside
    pad = 1e-6
    w, h = box[1] - box[0], box[3] - box[2]
    hist_extent = (box[0] - pad * w - 1e-9, box[1] + pad * w + 1e-9,
                   box[2] - pad * h - 1e-9, box[3] + pad * h + 1e-9)
    longest = int(np.clip(np.sqrt(args.iters / 12), 200, 1500)) if args.bins is None else args.bins
    ww, hh = hist_extent[1] - hist_extent[0], hist_extent[3] - hist_extent[2]
    bins = (max(int(longest * ww / max(ww, hh)), 50), max(int(longest * hh / max(ww, hh)), 50))

    # centre of the structure: the fixed point nearest the origin, elliptic if there is one
    fps = fixed_points(a, b, c)
    elliptic = [p for p in fps if abs(p[2]) < 2.0]
    pool, kind = (elliptic, "elliptic fixed point") if elliptic else (fps, "hyperbolic fixed point")
    if pool:
        x_fp, y_fp, _ = min(pool, key=lambda p: p[0] ** 2 + p[1] ** 2)
        centre = (x_fp, y_fp)
    else:
        centre, kind = ((box[0] + box[1]) / 2, (box[2] + box[3]) / 2), "centre of the orbit box (no fixed point)"

    # map window and portrait line, both in units of the length scale
    half = args.window_scale * scale
    window = ((centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half))
    start = centre
    end = (centre[0] + 0.9 * half, centre[1] + 0.9 * half)

    return Layout(scale, centre, kind, box, hist_extent, bins, window, (start, end),
                  n_box, time.time() - t0)


def report(layout):
    """Print what pass 1 found."""
    bx = layout.orbit_box
    (x0, x1), (y0, y1) = layout.window
    p0, p1 = layout.line
    print(f"pass 1 (survey, {layout.survey_seconds:.1f}s, includes JIT compile on first run)")
    print(f"  length scale L = {layout.scale:.4g}  (max of |a|, |b|, sqrt|c|)")
    print(f"  centre:           ({layout.centre[0]:.4g}, {layout.centre[1]:.4g})  [{layout.centre_kind}]")
    print(f"  orbit from (0,0): x {bx[0]:.4g}..{bx[1]:.4g}, y {bx[2]:.4g}..{bx[3]:.4g} "
          f"({layout.survey_points:,} points)")
    print(f"  map window:       x {x0:.4g}..{x1:.4g}, y {y0:.4g}..{y1:.4g}")
    print(f"  portrait line:    ({p0[0]:.4g}, {p0[1]:.4g}) -> ({p1[0]:.4g}, {p1[1]:.4g})")
    print(f"  density image:    {layout.bins[0]} x {layout.bins[1]} bins")


# --------------------------------------------------------------------------
# pass 2: figures
# --------------------------------------------------------------------------
def plot_orbit(H, extent, params, lam, n_points, outside):
    """Density image of one orbit: the classic Hopalong picture."""
    a, b, c = params
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(np.log1p(H), origin="lower", cmap="magma",
              extent=[extent[0], extent[1], extent[2], extent[3]])
    ax.set_aspect("equal")
    note = f", {outside:,} outside range" if outside else ""
    ax.set_title(f"One orbit from (0,0): a={a}, b={b}, c={c}\n"
                 f"{n_points:,} points{note}, Lyapunov exponent ≈ {lam:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def plot_lyapunov_map(L, window, thresh, params, n, lam0):
    """Left: exponent per starting point. Right: regular / chaotic / escaped."""
    a, b, c = params
    (x_lo, x_hi), (y_lo, y_hi) = window
    extent = [x_lo, x_hi, y_lo, y_hi]
    escaped = np.isnan(L)
    chaotic = L > thresh
    regular = ~escaped & ~chaotic

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)

    vmax = np.nanpercentile(L, 99.5) if (~escaped).any() else 1.0
    im = axes[0].imshow(np.clip(L, 0, None), origin="lower", extent=extent,
                        cmap="inferno", vmin=0, vmax=max(vmax, 1e-6))
    fig.colorbar(im, ax=axes[0], shrink=0.8,
                 label=f"finite-time Lyapunov exponent (N={n:,})")
    axes[0].set_title("Lyapunov exponent per starting point\n"
                      "dark = regular, bright = chaotic")

    cls = np.where(escaped, 2, np.where(chaotic, 1, 0))
    axes[1].imshow(cls, origin="lower", extent=extent, vmin=0, vmax=2,
                   cmap=ListedColormap(["#3b6ea5", "#f2a03d", "#cccccc"]),
                   interpolation="nearest")
    axes[1].set_title(f"Classification (threshold = {thresh})\n"
                      f"regular {regular.mean():.1%}   chaotic {chaotic.mean():.1%}"
                      f"   escaped {escaped.mean():.1%}")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x0")
        ax.set_ylabel("y0")
        ax.plot(0, 0, marker="x", color="white", ms=9, mew=2)
        ax.plot(0, 0, marker="x", color="black", ms=9, mew=0.8)
    if x_lo <= 0 <= x_hi and y_lo <= 0 <= y_hi:
        text_xy = (x_lo + 0.62 * (x_hi - x_lo), y_lo + 0.06 * (y_hi - y_lo))
        axes[1].annotate(f"start (0,0)\nλ ≈ {lam0:.3f}", (0, 0), text_xy,
                         arrowprops=dict(arrowstyle="->"), fontsize=9,
                         bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.suptitle(f"Hopalong map, a={a}, b={b}, c={c}", fontsize=13)
    return fig


def draw_portrait(ax, params, p0, p1, n_orbits, n_points, lyap_iters, thresh, skip):
    """Draw orbits from a line of starting points onto `ax`.

    Regular orbits: cool colours by position along the line (decorative).
    Chaotic orbits: warm colours by the size of the exponent.
    Returns (number of chaotic orbits, bounding box of everything drawn)."""
    a, b, c = params
    cool, warm = plt.get_cmap("cool"), plt.get_cmap("autumn")
    n_chaotic = 0
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for s in np.linspace(0.0, 1.0, n_orbits):
        x0 = p0[0] + s * (p1[0] - p0[0])
        y0 = p0[1] + s * (p1[1] - p0[1])
        lam = lyapunov_orbit(a, b, c, x0, y0, lyap_iters, skip)
        xs, ys = orbit(a, b, c, n_points, x0, y0, skip)
        if len(xs) == 0:
            continue
        if lam > thresh:
            n_chaotic += 1
            color = warm(0.15 + 0.7 * min(lam / 0.03, 1.0))
            size, alpha = 0.15, 0.35
        else:
            color = cool(s)
            size, alpha = 0.25, 0.8
        ax.scatter(xs, ys, s=size, color=color, alpha=alpha, linewidths=0,
                   rasterized=True)
        lo = np.minimum(lo, [xs.min(), ys.min()])
        hi = np.maximum(hi, [xs.max(), ys.max()])
    return n_chaotic, (lo[0], hi[0], lo[1], hi[1])


def plot_portrait(params, line, n_orbits, n_points, lyap_iters, thresh, skip):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    n_ch, (x0, x1, y0, y1) = draw_portrait(ax, params, *line, n_orbits, n_points,
                                           lyap_iters, thresh, skip)
    m = 0.03 * max(x1 - x0, y1 - y0)                 # exact limits of what was drawn
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")
    ax.axis("off")
    print(f"portrait: {n_ch} of {n_orbits} orbits chaotic (threshold {thresh})")
    return fig


def plot_header(params, line, n_orbits, n_points, lyap_iters, thresh, skip,
                size, zoom):
    """Wide image of the portrait with a black background and no margins."""
    width, height = size
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.axis("off")
    n_ch, (x0, x1, y0, y1) = draw_portrait(ax, params, *line, n_orbits, n_points,
                                           lyap_iters, thresh, skip)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_h = (y1 - y0) / 2 / zoom                     # zoom > 1 crops the sides
    half_w = half_h * width / height
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    print(f"header: {n_ch} of {n_orbits} orbits chaotic, {width}x{height} px")
    return fig


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("map parameters")
    g.add_argument("--a", type=float, default=0.7)
    g.add_argument("--b", type=float, default=0.4)
    g.add_argument("--c", type=float, default=0.0)

    g = p.add_argument_group("what to draw")
    g.add_argument("--which", default="all",
                   choices=["all", "orbit", "map", "portrait", "header"],
                   help="'all' = orbit, map and portrait (default); header is separate")
    g.add_argument("--save", action="store_true", help="write PNG files instead of showing")
    g.add_argument("--outdir", default=".", help="folder for the PNG files")
    g.add_argument("--survey-only", action="store_true",
                   help="run pass 1 only: print the extents and exit")

    g = p.add_argument_group("iterations and resolution")
    g.add_argument("--iters", type=int, default=10_000_000,
                   help="points of the single orbit from (0,0); memory use does not depend on it")
    g.add_argument("--bins", type=int,
                   help="density image pixels on the longer side (default: from --iters)")
    g.add_argument("--lyap-iters", type=int, default=8000,
                   help="iterations per orbit for the Lyapunov map")
    g.add_argument("--grid", type=int, default=400, help="grid points per axis")
    g.add_argument("--skip", type=int, default=500, help="discarded transient steps")
    g.add_argument("--thresh", type=float, default=0.004,
                   help="exponent above which an orbit counts as chaotic")

    g = p.add_argument_group("map window")
    g.add_argument("--window-scale", type=float, default=3.8,
                   help="map window half-size in units of the length scale L, "
                        "centred on the fixed point")

    g = p.add_argument_group("overrides (automatic by default)")
    g.add_argument("--xlim", type=float, nargs=2, help="x window of the map")
    g.add_argument("--ylim", type=float, nargs=2, help="y window of the map")
    g.add_argument("--start", type=float, nargs=2, help="portrait line: first start point")
    g.add_argument("--end", type=float, nargs=2, help="portrait line: last start point")

    g = p.add_argument_group("portrait and header")
    g.add_argument("--portrait-orbits", type=int, default=150)
    g.add_argument("--portrait-iters", type=int, default=30000,
                   help="points drawn per portrait orbit")
    g.add_argument("--header-size", type=int, nargs=2, default=(1600, 900),
                   metavar=("W", "H"), help="pixels (default 1600 900)")
    g.add_argument("--header-zoom", type=float, default=1.4,
                   help="above 1 crops the picture more tightly")
    return p.parse_args(argv)


def finish(fig, name, args):
    """Save the figure or keep it for plt.show()."""
    if args.save:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, name)
        if name == "hopalong_header.png":      # exact pixel size, no cropping
            fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
        else:
            fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                        bbox_inches="tight", pad_inches=0.1)
        print("saved", path)


def main(argv=None):
    args = parse_args(argv)
    params = (args.a, args.b, args.c)
    wanted = {"all": ["orbit", "map", "portrait"]}.get(args.which, [args.which])

    # ---- pass 1: extents ----
    layout = survey(params, args)
    if layout is None:
        sys.exit(f"The orbit from (0,0) escaped to infinity within 1000 points "
                 f"for a,b,c = {params}. Try other parameters.")
    window = layout.window
    if args.xlim:
        window = (tuple(args.xlim), window[1])
    if args.ylim:
        window = (window[0], tuple(args.ylim))
    line = (tuple(args.start) if args.start else layout.line[0],
            tuple(args.end) if args.end else layout.line[1])
    layout.window, layout.line = window, line
    report(layout)
    if args.survey_only:
        return

    # ---- pass 2: render ----
    figs = {}
    lam0 = None
    if "orbit" in wanted:
        t0 = time.time()
        H, lam0, n_pts, outside = orbit_density(*params, args.iters, args.skip,
                                                *layout.hist_extent, *layout.bins)
        print(f"orbit from (0,0): {n_pts:,} points, exponent {lam0:.4f}, "
              f"{outside:,} outside the pass-1 range ({time.time() - t0:.1f}s)")
        figs["hopalong_orbit.png"] = plot_orbit(H, layout.hist_extent, params, lam0,
                                                n_pts, outside)
    elif "map" in wanted:
        lam0 = lyapunov_orbit(*params, 0.0, 0.0, args.iters, args.skip)

    if "map" in wanted:
        t0 = time.time()
        gx = np.linspace(*window[0], args.grid)
        gy = np.linspace(*window[1], args.grid)
        L = lyapunov_grid(*params, gx, gy, args.lyap_iters, args.skip)
        ok = np.isfinite(L)
        print(f"map: {args.grid}x{args.grid} orbits x {args.lyap_iters:,} iterations "
              f"in {time.time() - t0:.1f}s")
        print(f"map: regular {(ok & (L <= args.thresh)).mean():.1%}  "
              f"chaotic {(L > args.thresh).mean():.1%}  escaped {(~ok).mean():.1%}")
        figs["hopalong_lyapunov_map.png"] = plot_lyapunov_map(
            L, window, args.thresh, params, args.lyap_iters, lam0)

    if "portrait" in wanted:
        figs["hopalong_orbit_portrait.png"] = plot_portrait(
            params, line, args.portrait_orbits, args.portrait_iters,
            args.lyap_iters, args.thresh, args.skip)

    if "header" in wanted:
        figs["hopalong_header.png"] = plot_header(
            params, line, args.portrait_orbits, args.portrait_iters,
            args.lyap_iters, args.thresh, args.skip,
            args.header_size, args.header_zoom)

    for name, fig in figs.items():
        finish(fig, name, args)
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
