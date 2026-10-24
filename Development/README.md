# One-pass Hopalong orbit density plot

This script renders the classic Hopalong map as a visit-density image. It
streams the orbit into a fixed-size histogram and expands the represented
coordinate range when necessary, rather than storing every orbit point or
making a full preliminary pass to discover the bounds.

## Target

The goal is to plot a long orbit with:

- Memory bounded by the histogram dimensions, rather than the number of
  iterations.
- Exact integer visit counts at the histogram's **final** cell resolution.
- Automatic coverage when the orbit moves beyond the current grid.
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

### Starting grid and pilot

`pilot_bounds` runs a short orbit of 2,000 steps to estimate a starting
center and square span. This is a sizing heuristic only: later points are
still checked, and the grid grows whenever a point falls outside it.

The pilot is not retained as histogram data. The main pass starts the orbit
again from `(0, 0)`, so the initial pilot steps are computed twice. The
approach avoids a full bounds survey, but is not literally a single
evaluation of every recurrence step.

### Streaming histogram and dyadic growth

The histogram is a square `size x size` array of signed 64-bit integer
counts. A point inside the current bounds is mapped to a cell by scaling its
coordinates by the inverse cell width and converting to integer indices.

When a plotted point falls outside the grid:

1. Existing counts are merged from each 2x2 block into one cell, using
   integer addition.
2. The grid's cell width doubles, so its coordinate span doubles while its
   number of cells per side remains constant.
3. The old grid is positioned in the half opposite the direction of growth.
   The point that triggered growth is then binned, and orbit processing
   resumes.

The merge preserves the total count and is an exact aggregation onto the
coarser grid. Earlier points are not lost or approximated in count. However,
their original fine-cell locations cannot be recovered after merging: all
points are ultimately represented at the final, coarser resolution.

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
than the requested number of points may then be plotted.

## Caveats

- **Resolution decreases on growth.** Each doubling of the coordinate span
  also doubles the cell width. This preserves a fixed-size grid, but a
  trajectory with a much larger range than the pilot predicted will have a
  coarser final image.
- **Counts are exact only for the final bins.** Integer merges preserve visit
  totals, but do not preserve sub-cell positions or allow a later
  higher-resolution reconstruction.
- **Floating-point bin edges.** Orbit coordinates and grid-to-cell mapping
  use floating-point arithmetic. Points extremely close to a cell boundary
  can be assigned to either neighboring cell due to rounding.
- **Growth has a cost.** Each growth allocates and scans another full grid.
  Usually this is infrequent, but a trajectory that repeatedly expands the
  range can incur additional time and temporary memory.
- **Memory is bounded by grid size, not necessarily small.** The count array
  uses approximately `8 * bins * bins` bytes, with additional temporary
  memory during regridding and for plotting/smoothing.
- **Escape guard.** The orbit stops if either coordinate's absolute value
  exceeds `1e6`. This is a safety cutoff, not a general mathematical test
  that every parameter choice produces a bounded orbit.
- **Smoothing changes the displayed density.** Gaussian smoothing is a
  visualization step, not part of the visit counts. Zero padding can reduce
  smoothed values near the image boundary.
- **Pilot overhead.** The 2,000-step pilot repeats those recurrence
  calculations in the main pass. For very short runs, this overhead may be
  noticeable.
- **Default/help-text mismatch in the script.** The code sets `--iters` to
  `10,000,000,000`, while its help text says `10,000,000`. Set `--iters`
  explicitly when the exact run length matters.

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
