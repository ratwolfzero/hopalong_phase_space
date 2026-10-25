# One-pass Hopalong orbit density plot

This script renders the classic Hopalong map as a visit-density image. It
streams the orbit into a fixed-size histogram and adapts the represented
coordinate range when necessary, rather than storing every orbit point or
making a full preliminary pass to discover the bounds.

## Target

The goal is to plot a long orbit with:

- Memory bounded by the histogram dimensions, rather than the number of
  iterations.
- Exact integer visit counts at the histogram's **final** cell resolution.
- Automatic coverage when the orbit moves beyond the current grid, at the
  finest cell size that still fits the whole orbit.
- A useful density visualization, optionally Gaussian-smoothed.

The implementation uses NumPy arrays, Matplotlib, and Numba. Save the script
as `hopalong_orbit_onepass.py` or substitute its actual filename in the
commands below.

## Implementation details

### Orbit recurrence

Starting at `(x, y) = (0, 0)`, each step uses the previous `x`:

```text
old_x = x
x = y - sign(old_x) * sqrt(abs(b * old_x - c))
y = a - old_x
```

The default parameters are `a=0.7`, `b=0.4`, and `c=0`. The first `--skip`
steps are treated as a transient and are not added to the histogram.

### Starting grid

The first 4,096 plotted points are held in a small buffer (64 KB). Their
bounding box selects the starting cell size: the smallest power of two times
`H_MIN = 2**-32` at which the box fits the grid. The grid is centred on the
box, the buffered points are binned, and the orbit simply continues from where
it was. No recurrence step is computed twice, and there is no separate pilot
orbit.

This is a sizing heuristic only. Later points are still checked, and the grid
is reframed whenever a point falls outside it. The attractor's extent usually
keeps growing with the number of points, so the buffer cannot predict the final
size; it only avoids a burst of cheap-but-pointless regridding at the start.

### Streaming histogram and dyadic growth

The histogram is a square `size x size` array of signed 64-bit integer
counts. A point inside the current bounds is mapped to a cell by scaling its
coordinates by the inverse cell width and converting to integer indices. The
hot loop does nothing else: it carries no bookkeeping of its own.

When a plotted point falls outside the grid, the grid is reframed:

1. The occupied block of cells is located with one pass over the array.
2. If the occupied block **and** the new point still fit at the current cell
   size, the counts are only **translated**, so the block is centred. Nothing
   is merged and no resolution is lost.
3. Otherwise the cell width is doubled: each 2x2 block of counts is merged into
   one cell by integer addition, and the merged block is centred in the grid.
   This repeats until everything fits.
4. The triggering point is binned and orbit processing resumes.

Centring leaves room on both sides, so a reframe is rarely needed again soon.
Only the occupied block is read, written and cleared, using two preallocated
arrays that swap roles, so a reframe costs a few milliseconds at 2048 x 2048.

All cell widths are `H_MIN * 2**k`, so old and new cell boundaries stay
aligned: translating and 2x2 merging are exact aggregations onto the new
lattice. Earlier points are not lost or approximated in count. However, their
original fine-cell locations cannot be recovered after merging: all points are
ultimately represented at the final, coarser resolution.

Because the grid doubles only when the occupied region no longer fits, the
final occupied region covers between about 50% and 100% of the grid side.

### Output image

Empty outer rows and columns are cropped before display. By default the
script displays `log1p(count)` with a `magma` color map. With `--smooth`, it
applies a separable Gaussian filter with sigma 1.5 cells before taking the
logarithm. `--save` writes `hopalong_orbit.png` to the selected output
directory instead of opening a window.

## Result

The result is a two-dimensional image of how often the orbit visits each
cell. Dense regions appear brighter, making the orbit's spatial structure
visible without keeping the full point sequence in memory.

On normal completion, the script reports the number of binned points, grid
size, final cell width, number of grid doublings, elapsed time, and occupied
image dimensions and coordinate extent. If the orbit reaches the escape
threshold, processing stops and the report indicates that it escaped; fewer
than the requested number of points may then be plotted. If it escapes before
any point is plotted, the script exits with a message.

## Caveats

- **Resolution decreases on growth.** Each doubling of the coordinate span
  also doubles the cell width. This preserves a fixed-size grid, but an orbit
  whose extent keeps growing with the number of points ends up with a coarser
  final image the longer it runs. The cell width is the smallest power of two
  at which the whole occupied region fits.
- **Counts are exact only for the final bins.** Integer merges preserve visit
  totals, but do not preserve sub-cell positions or allow a later
  higher-resolution reconstruction.
- **Floating-point bin edges.** Orbit coordinates and grid-to-cell mapping
  use floating-point arithmetic. Points extremely close to a cell boundary
  can be assigned to either neighboring cell due to rounding.
- **Reframing has a small fixed cost.** Each reframe scans the grid once and
  moves only the occupied block. Typical runs need zero to a few dozen
  reframes, and heavy-tailed orbits somewhat more; the total is on the order of
  0.1 s at 2048 x 2048 and does not scale with the number of iterations.
- **Finer images touch more memory.** A sharper final grid has more occupied
  cells, so the scattered histogram updates miss the cache a little more often.
  Measured on one machine, this was a few percent, and it is the price of the
  better resolution rather than of the reframing code.
- **Memory is bounded by grid size, not necessarily small.** Two count arrays
  are kept (about `16 * bins * bins` bytes in total), plus temporary memory for
  plotting/smoothing.
- **Finest cell size.** Cells never get finer than `2**-32`. This only matters
  for orbits whose entire extent is below about `5e-7`.
- **Escape guard.** The orbit stops if either coordinate's absolute value
  exceeds `1e6`. This is a safety cutoff, not a general mathematical test
  that every parameter choice produces a bounded orbit.
- **Smoothing changes the displayed density.** Gaussian smoothing is a
  visualization step, not part of the visit counts. Its sigma is measured in
  cells, so the apparent blur depends on the final cell size. Zero padding can
  reduce smoothed values near the image boundary.

## Use cases

- Exploring the spatial density and structure of Hopalong orbits.
- Quickly comparing plots produced with different `a`, `b`, and `c`
  parameters.
- Producing a bounded-memory visualization for very long runs when a
  fixed-resolution image is acceptable.
- Creating a smoothed, presentation-friendly density image with `--smooth`.

This method is intended for visualization, not for retaining a high
precision point cloud or supporting later zooming and re-binning. For those
needs, store orbit points or use a sparse/finer-resolution representation,
accepting the associated memory or processing cost.

## Requirements

Install the dependencies in the Python environment used to run the script:

```bash
python -m pip install numpy matplotlib numba
```

## Usage

```bash
python hopalong_orbit_onepass.py
python hopalong_orbit_onepass.py --smooth
python hopalong_orbit_onepass.py --save
python hopalong_orbit_onepass.py --a 1.1 --b 0.5 --c 0.3 --iters 2000000 --save
```

`--bins` sets the square grid side length. If omitted, the script derives it
from `--iters` and caps it at 2,048; the resulting side length is made even
for the 2x2 merge operation. `--outdir` selects where the PNG is saved.
