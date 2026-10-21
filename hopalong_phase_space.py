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

The numerical loops run in numba; plotting uses matplotlib.

Usage:
    python hopalong_phase_space.py                      # all three, shown on screen
    python hopalong_phase_space.py --save               # all three, written as PNG
    python hopalong_phase_space.py --which map --grid 800 --lyap-iters 20000 --save
    python hopalong_phase_space.py --a 1.1 --b 0.5 --c 0.3 --iters 2000000
    python hopalong_phase_space.py --which header --save
"""
import argparse
import os
import sys
import time

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


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def orbit_box(xs, ys, min_half=0.5):
    """Centre and half-widths of the box around the orbit."""
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    hx = max((xs.max() - xs.min()) / 2, min_half)
    hy = max((ys.max() - ys.min()) / 2, min_half)
    return cx, cy, hx, hy


def default_window(box, scale):
    """Plot window: the orbit's box, enlarged by `scale` about its centre."""
    cx, cy, hx, hy = box
    return (cx - scale * hx, cx + scale * hx), (cy - scale * hy, cy + scale * hy)


LINE_START = (0.45, -0.45)   # in units of the orbit box half-widths, from its centre
LINE_END = (2.8, 1.95)


def default_line(box):
    """Line of starting points for the portrait, placed relative to the orbit box.

    The offsets were chosen by eye so the line crosses several islands for the
    default parameters. For other parameters it is only a reasonable guess:
    use --start and --end to place it yourself."""
    cx, cy, hx, hy = box
    return ((cx + LINE_START[0] * hx, cy + LINE_START[1] * hy),
            (cx + LINE_END[0] * hx, cy + LINE_END[1] * hy))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def plot_orbit(xs, ys, params, lam):
    """Density image of one orbit: the classic Hopalong picture."""
    a, b, c = params
    H, xe, ye = np.histogram2d(xs, ys, bins=900)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(np.log1p(H.T), origin="lower", cmap="magma",
              extent=[xe[0], xe[-1], ye[0], ye[-1]])
    ax.set_aspect("equal")
    ax.set_title(f"One orbit from (0,0): a={a}, b={b}, c={c}\n"
                 f"{len(xs):,} points, Lyapunov exponent ≈ {lam:.4f}")
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
    n_ch, _ = draw_portrait(ax, params, *line, n_orbits, n_points,
                            lyap_iters, thresh, skip)
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

    g = p.add_argument_group("iterations and resolution")
    g.add_argument("--iters", type=int, default=10_000_000,
                   help="points of the single orbit from (0,0)")
    g.add_argument("--lyap-iters", type=int, default=8000,
                   help="iterations per orbit for the Lyapunov map")
    g.add_argument("--grid", type=int, default=400, help="grid points per axis")
    g.add_argument("--skip", type=int, default=500, help="discarded transient steps")
    g.add_argument("--thresh", type=float, default=0.004,
                   help="exponent above which an orbit counts as chaotic")

    g = p.add_argument_group("window and portrait line (automatic by default)")
    g.add_argument("--window-scale", type=float, default=3.0,
                   help="plot window = orbit box from (0,0) enlarged by this factor")
    g.add_argument("--xlim", type=float, nargs=2, help="override x window of the map")
    g.add_argument("--ylim", type=float, nargs=2, help="override y window of the map")
    g.add_argument("--start", type=float, nargs=2, help="portrait line: first start point")
    g.add_argument("--end", type=float, nargs=2, help="portrait line: last start point")
    g.add_argument("--portrait-orbits", type=int, default=150)
    g.add_argument("--portrait-iters", type=int, default=30000,
                   help="points drawn per portrait orbit")

    g = p.add_argument_group("header image")
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

    # the orbit from (0,0) frames everything else, so compute it first
    t0 = time.time()
    xs, ys = orbit(*params, args.iters, 0.0, 0.0, args.skip)
    if len(xs) < 1000:
        sys.exit(f"The orbit from (0,0) escaped to infinity after {len(xs)} points "
                 f"for a,b,c = {params}. Try other parameters.")
    lam0 = lyapunov_orbit(*params, 0.0, 0.0, args.iters, args.skip)
    print(f"orbit from (0,0): {len(xs):,} points, exponent {lam0:.4f} "
          f"({time.time() - t0:.1f}s, includes JIT compile on first run)")

    box = orbit_box(xs, ys)
    window = default_window(box, args.window_scale)
    if args.xlim:
        window = (tuple(args.xlim), window[1])
    if args.ylim:
        window = (window[0], tuple(args.ylim))
    p0, p1 = default_line(box)
    line = (tuple(args.start) if args.start else p0, tuple(args.end) if args.end else p1)

    figs = {}
    if "orbit" in wanted:
        figs["hopalong_orbit.png"] = plot_orbit(xs, ys, params, lam0)

    if "map" in wanted:
        t0 = time.time()
        gx = np.linspace(*window[0], args.grid)
        gy = np.linspace(*window[1], args.grid)
        L = lyapunov_grid(*params, gx, gy, args.lyap_iters, args.skip)
        ok = np.isfinite(L)
        print(f"map: {args.grid}x{args.grid} orbits x {args.lyap_iters:,} iterations "
              f"in {time.time() - t0:.1f}s | window x {window[0][0]:.2f}..{window[0][1]:.2f}, "
              f"y {window[1][0]:.2f}..{window[1][1]:.2f}")
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
