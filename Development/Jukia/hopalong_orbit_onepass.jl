#!/usr/bin/env julia
# Render the classic Hopalong orbit as an on-the-fly density histogram (one pass).
#
# Julia port of hopalong_orbit_onepass.py. Same algorithm: the histogram lives
# on a dyadic grid; whenever a point leaves the grid, the occupied block is
# translated if it still fits at the current cell size, otherwise merged 2x2
# into half the resolution, so the counts stay exact. Memory use is independent
# of the number of iterations.
#
# Setup (once):
#     julia -e 'using Pkg; Pkg.add(["CairoMakie", "GLMakie"])'
#   (CairoMakie is used with --save, GLMakie for the interactive window; install
#    only the one you need. Neither is loaded with --noplot.)
#
# Usage:
#     julia -O3 hopalong_orbit_onepass.jl
#     julia -O3 hopalong_orbit_onepass.jl --smooth
#     julia -O3 hopalong_orbit_onepass.jl --save
#     julia -O3 hopalong_orbit_onepass.jl --a 1.1 --b 0.5 --c 0.3 --iters 2000000 --save
#     julia -O3 hopalong_orbit_onepass.jl --iters 1_000_000_000 --noplot   # timing only
#
# Options: --a --b --c --iters --skip --bins --smooth --save --outdir --noplot --help

using Printf

const ESCAPE = 1.0e6
const WARMUP = 4096          # points buffered to choose the starting cell size
const H_MIN = 2.0^-32        # finest cell size; all cell sizes are H_MIN * 2^k

# ---------------------------------------------------------------------------
# Plotting backend: loaded only when this file is run as a script and plotting
# is requested, so `include`-ing it (tests, REPL) or --noplot stays fast.
# ---------------------------------------------------------------------------
const IS_MAIN = abspath(PROGRAM_FILE) == (@__FILE__)
const PLOTTING = IS_MAIN && !any(in(ARGS), ("--noplot", "--help", "-h"))
if PLOTTING
    if "--save" in ARGS
        using CairoMakie
    else
        using GLMakie
    end
end

# ---------------------------------------------------------------------------
# Core: orbit + dyadic histogram.
#
# Layout note: hist[ix, iy] (x is the fast, contiguous index). This is the same
# memory layout as NumPy's hist[iy, ix], so the hot loop has identical cache
# behaviour, and the matrix can be handed to Makie without transposing.
# Indices below are 0-based in the logic (as in the Python original) and shifted
# by +1 only where an array is accessed.
# ---------------------------------------------------------------------------

@inline function hop_step(a::Float64, b::Float64, c::Float64, x::Float64, y::Float64)
    sign_x = ifelse(x > 0.0, 1.0, ifelse(x < 0.0, -1.0, 0.0))
    nx = y - sign_x * sqrt(abs(b * x - c))
    return nx, a - x
end

"""
Run the orbit from step `i` and bin points until one leaves the grid.

Returns `(status, i, x, y)`. status: 0 = finished, 1 = the point of step `i`
lies outside the grid (not binned yet), 2 = the orbit escaped.
"""
function fill_until_outside!(hist::Matrix{Int64}, a::Float64, b::Float64, c::Float64,
                             i::Int, total::Int, skip::Int, x::Float64, y::Float64,
                             x0::Float64, y0::Float64, inv_h::Float64)
    side = Float64(size(hist, 1))
    @inbounds while i < total
        x, y = hop_step(a, b, c, x, y)
        if abs(x) > ESCAPE || abs(y) > ESCAPE
            return 2, i, x, y
        end
        if i >= skip
            fx = (x - x0) * inv_h
            fy = (y - y0) * inv_h
            if 0.0 <= fx < side && 0.0 <= fy < side
                hist[unsafe_trunc(Int, fx) + 1, unsafe_trunc(Int, fy) + 1] += 1
            else
                return 1, i, x, y
            end
        end
        i += 1
    end
    return 0, i, x, y
end

"Index bounds (ix0, ix1, iy0, iy1), 0-based inclusive, of the non-zero cells."
function occupied_box(hist::Matrix{Int64})
    side = size(hist, 1)
    colmask = zeros(Int64, side)
    iy0 = side
    iy1 = -1
    @inbounds for iy in 0:side-1
        row = Int64(0)
        for ix in 0:side-1
            v = hist[ix+1, iy+1]
            row |= v
            colmask[ix+1] |= v
        end
        if row != 0
            iy0 == side && (iy0 = iy)
            iy1 = iy
        end
    end
    ix0 = side
    ix1 = -1
    @inbounds for ix in 0:side-1
        if colmask[ix+1] != 0
            ix0 == side && (ix0 = ix)
            ix1 = ix
        end
    end
    return ix0, ix1, iy0, iy1
end

"Translate the occupied block of `src` into the (all-zero) `dst`; clear it in `src`."
function move_block!(src::Matrix{Int64}, dst::Matrix{Int64},
                     ix0::Int, ix1::Int, iy0::Int, iy1::Int, dx::Int, dy::Int)
    @inbounds for iy in iy0:iy1
        for ix in ix0:ix1
            dst[ix+dx+1, iy+dy+1] = src[ix+1, iy+1]
            src[ix+1, iy+1] = 0
        end
    end
    return nothing
end

"Merge 2x2 blocks of the occupied block into `dst`, translated; clear it in `src`."
function merge_block!(src::Matrix{Int64}, dst::Matrix{Int64},
                      ix0::Int, ix1::Int, iy0::Int, iy1::Int, dx::Int, dy::Int)
    @inbounds for oy in (iy0 ÷ 2):(iy1 ÷ 2)
        for ox in (ix0 ÷ 2):(ix1 ÷ 2)
            dst[ox+dx+1, oy+dy+1] = src[2ox+1, 2oy+1] + src[2ox+2, 2oy+1] +
                                    src[2ox+1, 2oy+2] + src[2ox+2, 2oy+2]
        end
    end
    @inbounds for iy in iy0:iy1, ix in ix0:ix1
        src[ix+1, iy+1] = 0
    end
    return nothing
end

"""
Reframe the grid so that all counted points and the point `(x, y)` fit.

If the occupied block plus the new point fit at the current cell size, the
counts are only translated (no resolution lost). Otherwise the cell size is
doubled (2x2 merge) until they fit. The block is always centred. Every cell
size is a power of two times `H_MIN`, so old and new cell boundaries stay
aligned and the counts stay exact. `spare` is an all-zero array of the same
shape; the two arrays are swapped. Returns `(hist, spare, x0, y0, h, doublings)`.
"""
function make_room!(hist::Matrix{Int64}, spare::Matrix{Int64},
                    x0::Float64, y0::Float64, h::Float64, x::Float64, y::Float64)
    side = size(hist, 1)
    ix0, ix1, iy0, iy1 = occupied_box(hist)
    grows = 0
    while true
        px = floor(Int, (x - x0) / h)
        py = floor(Int, (y - y0) / h)
        lox = min(ix0, px)
        hix = max(ix1, px)
        loy = min(iy0, py)
        hiy = max(iy1, py)
        if hix - lox < side && hiy - loy < side
            dx = (side - (hix - lox + 1)) ÷ 2 - lox
            dy = (side - (hiy - loy + 1)) ÷ 2 - loy
            move_block!(hist, spare, ix0, ix1, iy0, iy1, dx, dy)
            return spare, hist, x0 - dx * h, y0 - dy * h, h, grows
        end
        # Does not fit: halve the resolution, with the old block centred.
        dx = (side - (ix1 ÷ 2 - ix0 ÷ 2 + 1)) ÷ 2 - ix0 ÷ 2
        dy = (side - (iy1 ÷ 2 - iy0 ÷ 2 + 1)) ÷ 2 - iy0 ÷ 2
        merge_block!(hist, spare, ix0, ix1, iy0, iy1, dx, dy)
        hist, spare = spare, hist
        ix0 = ix0 ÷ 2 + dx
        ix1 = ix1 ÷ 2 + dx
        iy0 = iy0 ÷ 2 + dy
        iy1 = iy1 ÷ 2 + dy
        x0 -= dx * 2.0 * h
        y0 -= dy * 2.0 * h
        h *= 2.0
        grows += 1
    end
end

"""
Single-pass visit-count histogram on a `side` x `side` dyadic grid.

The first `WARMUP` plotted points are buffered to choose the starting cell
size; from then on the grid adapts whenever a point falls outside. `side` must
be even. Returns `(hist, x0, y0, h, grows, escaped)` with `(x0, y0)` the
lower-left corner of the grid and `h` the final cell size.
"""
function orbit_histogram(a::Float64, b::Float64, c::Float64, n::Int, skip::Int, side::Int)
    total = n + skip
    buf_x = Vector{Float64}(undef, WARMUP)
    buf_y = Vector{Float64}(undef, WARMUP)
    nbuf = 0
    i = 0
    x = 0.0
    y = 0.0
    escaped = false
    while i < total && nbuf < WARMUP
        x, y = hop_step(a, b, c, x, y)
        if abs(x) > ESCAPE || abs(y) > ESCAPE
            escaped = true
            break
        end
        if i >= skip
            nbuf += 1
            buf_x[nbuf] = x
            buf_y[nbuf] = y
        end
        i += 1
    end

    hist = zeros(Int64, side, side)
    if nbuf == 0
        return hist, 0.0, 0.0, H_MIN, 0, escaped
    end

    # Starting grid: smallest cell size that fits the buffered points, centred.
    xmin, xmax = extrema(view(buf_x, 1:nbuf))
    ymin, ymax = extrema(view(buf_y, 1:nbuf))
    extent = max(xmax - xmin, ymax - ymin)
    h = H_MIN
    while extent / h >= side - 1
        h *= 2.0
    end
    lo_x = floor(Int, xmin / h)
    lo_y = floor(Int, ymin / h)
    wid_x = floor(Int, xmax / h) - lo_x + 1
    wid_y = floor(Int, ymax / h) - lo_y + 1
    x0 = (lo_x - (side - wid_x) ÷ 2) * h
    y0 = (lo_y - (side - wid_y) ÷ 2) * h
    for k in 1:nbuf
        fx = clamp(trunc(Int, (buf_x[k] - x0) / h), 0, side - 1)
        fy = clamp(trunc(Int, (buf_y[k] - y0) / h), 0, side - 1)
        hist[fx+1, fy+1] += 1
    end

    spare = zeros(Int64, side, side)
    grows = 0
    while !escaped
        status, i, x, y = fill_until_outside!(hist, a, b, c, i, total, skip,
                                              x, y, x0, y0, 1.0 / h)
        status == 2 && (escaped = true)
        status != 1 && break
        hist, spare, x0, y0, h, g = make_room!(hist, spare, x0, y0, h, x, y)
        grows += g
        fx = clamp(trunc(Int, (x - x0) / h), 0, side - 1)
        fy = clamp(trunc(Int, (y - y0) / h), 0, side - 1)
        hist[fx+1, fy+1] += 1
        i += 1
    end
    return hist, x0, y0, h, grows, escaped
end

# ---------------------------------------------------------------------------
# Post-processing (runs once, not performance critical)
# ---------------------------------------------------------------------------

"Drop the empty margin; return the sub-matrix and its (xmin, xmax, ymin, ymax) extent."
function crop_to_occupied(hist::Matrix{Int64}, x0::Float64, y0::Float64, h::Float64)
    cols = findall(vec(any(!iszero, hist; dims = 2)))   # ix with any count
    rows = findall(vec(any(!iszero, hist; dims = 1)))   # iy with any count
    c0, c1 = first(cols), last(cols)
    r0, r1 = first(rows), last(rows)
    sub = hist[c0:c1, r0:r1]
    extent = (x0 + (c0 - 1) * h, x0 + c1 * h, y0 + (r0 - 1) * h, y0 + r1 * h)
    return sub, extent
end

"Separable Gaussian filter (zero padding), `sigma` measured in pixels."
function smooth_histogram(histogram::AbstractMatrix; sigma::Float64 = 1.5)
    radius = ceil(Int, 4sigma)
    kernel = [exp(-0.5 * (k / sigma)^2) for k in -radius:radius]
    kernel ./= sum(kernel)

    src = Float64.(histogram)
    n1, n2 = size(src)
    for dim in 1:2
        dst = zeros(Float64, n1, n2)
        for (idx, k) in enumerate(-radius:radius)
            w = kernel[idx]
            if dim == 1
                lo, hi = max(1, 1 - k), min(n1, n1 - k)
                lo <= hi && @views dst[lo:hi, :] .+= w .* src[lo+k:hi+k, :]
            else
                lo, hi = max(1, 1 - k), min(n2, n2 - k)
                lo <= hi && @views dst[:, lo:hi] .+= w .* src[:, lo+k:hi+k]
            end
        end
        src = dst
    end
    return src
end

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

Base.@kwdef struct Args
    a::Float64 = 0.7
    b::Float64 = 0.4
    c::Float64 = 0.0
    iters::Int = 10_000_000_000
    skip::Int = 500
    bins::Union{Int,Nothing} = nothing
    smooth::Bool = false
    save::Bool = false
    noplot::Bool = false
    outdir::String = "."
end

const HELP = """
usage: julia hopalong_orbit_onepass.jl [options]

  --a A            (default 0.7)
  --b B            (default 0.4)
  --c C            (default 0)
  --iters N        number of plotted points (default 10_000_000_000)
  --skip N         discarded transient steps (default 500)
  --bins N         side of the square count grid (default: scales with --iters,
                   at most 2048); memory ~ 16*bins^2 bytes
  --smooth         Gaussian-smooth the histogram (sigma: 1.5 pixels)
  --save           save hopalong_orbit.png instead of showing it
  --outdir DIR     directory for the output PNG (default: current directory)
  --noplot         only compute and print statistics (no plotting packages loaded)
  -h, --help       show this help
"""

function die(msg)
    println(stderr, "error: ", msg)
    exit(2)
end

function parse_count(name, s)
    t = replace(s, "_" => "")
    v = tryparse(Int, t)
    if v === nothing
        f = tryparse(Float64, t)
        (f !== nothing && isfinite(f) && isinteger(f)) || die("$name must be an integer, got '$s'")
        v = Int(f)
    end
    return v
end

function parse_float(name, s)
    v = tryparse(Float64, s)
    (v !== nothing && isfinite(v)) || die("--$name must be a finite number, got '$s'")
    return v
end

function partition_eq(s)
    j = findfirst('=', s)
    j === nothing ? (s, "", "") : (s[1:j-1], "=", s[j+1:end])
end

function parse_args(argv)
    opts = Dict{Symbol,Any}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        i += 1
        if arg in ("-h", "--help")
            print(HELP)
            exit(0)
        end
        startswith(arg, "--") || die("unrecognized argument '$arg'")
        key, eq, val = partition_eq(arg[3:end])
        if key in ("smooth", "save", "noplot")
            opts[Symbol(key)] = true
            continue
        end
        key in ("a", "b", "c", "iters", "skip", "bins", "outdir") || die("unrecognized argument '$arg'")
        if isempty(eq)
            i <= length(argv) || die("--$key expects a value")
            val = argv[i]
            i += 1
        end
        opts[Symbol(key)] = if key in ("a", "b", "c")
            parse_float(key, val)
        elseif key == "outdir"
            String(val)
        else
            parse_count("--$key", val)
        end
    end
    args = Args(; opts...)
    args.iters > 0 || die("--iters must be a positive integer")
    args.skip >= 0 || die("--skip must be non-negative")
    (args.bins === nothing || args.bins > 0) || die("--bins must be a positive integer")
    return args
end

function commas(n::Integer)
    s = string(abs(n))
    groups = [s[max(1, i - 2):i] for i in length(s):-3:1]   # 3-digit groups, from the right
    return (n < 0 ? "-" : "") * join(reverse(groups), ",")
end

# ---------------------------------------------------------------------------
# Plot (Makie; Cairo for --save, GL for the interactive window)
# ---------------------------------------------------------------------------

function plot_density(density, extent, args::Args, points::Int, density_label::String)
    fig = Figure(size = (800, 800))
    ax = Axis(fig[1, 1];
              title = "Hopalong orbit: a=$(args.a), b=$(args.b), c=$(args.c)\n" *
                      "$(commas(points)) points, $density_label",
              xlabel = "x", ylabel = "y", aspect = DataAspect())
    image!(ax, extent[1]..extent[2], extent[3]..extent[4], log1p.(density);
           colormap = :magma, interpolate = false)
    if args.save
        mkpath(args.outdir)
        path = joinpath(args.outdir, "hopalong_orbit.png")
        save(path, fig; px_per_unit = 1.5)
        println("saved ", path)
    else
        wait(display(fig))      # block until the window is closed, like plt.show()
    end
    return nothing
end

function main(argv = ARGS)
    args = parse_args(argv)
    side = args.bins === nothing ?
           min(2 * trunc(Int, clamp(sqrt(args.iters / 12), 200, 1500)), 2048) :
           args.bins
    side += side % 2   # the dyadic merge needs an even grid side

    # Tiny run first so JIT compilation is not included in the reported time.
    orbit_histogram(args.a, args.b, args.c, 30_000, 0, 64)

    started = time_ns()
    hist, x0, y0, h, grows, escaped = orbit_histogram(
        args.a, args.b, args.c, args.iters, args.skip, side)
    seconds = (time_ns() - started) / 1e9
    points = Int(sum(hist))
    if points == 0
        println(stderr, "The orbit escaped before producing any plotted points.")
        exit(1)
    end

    histogram, extent = crop_to_occupied(hist, x0, y0, h)
    nbx, nby = size(histogram)
    @printf("single pass: %s points, grid %dx%d, cell %.4g, %d grid doublings%s (%.2fs, compile excluded)\n",
            commas(points), side, side, h, grows, escaped ? ", orbit escaped" : "", seconds)
    @printf("histogram %dx%d, x %.6g..%.6g, y %.6g..%.6g\n",
            nbx, nby, extent[1], extent[2], extent[3], extent[4])

    args.noplot && return nothing
    density = args.smooth ? smooth_histogram(histogram) : histogram
    label = args.smooth ? "log Gaussian-smoothed visit density" : "log visit density"
    plot_density(density, extent, args, points, label)
    return nothing
end

if IS_MAIN
    main()
end
