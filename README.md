# Hopalong map: regular and chaotic regions

`hopalong_phase_space.py` explores the phase space of the Hopalong map (Barry Martin) with three views of the same map:

- **orbit:** one orbit from (0,0) as a density image, the classic Hopalong picture
- **map:** a Lyapunov exponent for every starting point on a grid, classifying each as regular or chaotic
- **portrait:** many orbits on a dark background, coloured by that classification (meant for looking at, not for measuring)

A fourth option, **header**, renders a wide version of the portrait for use as a blog header.

![One orbit from (0,0)](hopalong_orbit.png)

![Lyapunov-exponent map](hopalong_lyapunov_map.png)

![Orbit portrait](hopalong_orbit_portrait.png)

## Scope

The Hopalong map is often called an "attractor", but it preserves area, and mixed regular and chaotic motion is the standard behaviour of area-preserving maps. Nothing here is a new result. The script checks that behaviour numerically for one parameter set (`a=0.7`, `b=0.4`, `c=0`) and gives a way to see it. No comparison with the literature on this specific map has been made.

## The map

For parameters `a`, `b`, `c`:

$$
\begin{aligned}
x_{n+1}&=y_n-\mathrm{sign}(x_n)\sqrt{\lvert bx_n-c\rvert}\\
y_{n+1}&=a-x_n
\end{aligned}
$$

Where it is differentiable, its Jacobian is

$$
J(x)=\begin{pmatrix}
-\dfrac{b\mathrm{sign}(x)\mathrm{sign}(bx-c)}{2\sqrt{\lvert bx-c\rvert}}&1\\
-1&0
\end{pmatrix}
$$

with determinant

$$
\det J=1
$$

The map is also invertible, because `x_n=a-y_{n+1}` gives `x_n` and then `y_n` follows from the first equation. Together this means it preserves area, so nothing contracts onto an attracting set and the picture depends on where an orbit starts.

## The Lyapunov exponent

A tangent vector is pushed forward along the orbit and renormalised at every step:

$$
v_n=J(x_{n-1})\hat v_{n-1}
$$

$$
\hat v_n=\frac{v_n}{\lVert v_n\rVert}
$$

The largest finite-time exponent over N steps is the mean logarithmic stretching:

$$
\lambda_N=\frac{1}{N}\sum_{n=1}^{N}\ln\lVert v_n\rVert
$$

For a regular (quasi-periodic) orbit the stretching is typically only linear in n, so λ falls toward 0, roughly like

$$
\lambda_N\sim\frac{\ln N}{N}
$$

and finite-time noise can push it slightly below zero. For a chaotic orbit the stretching is exponential and λ stays clearly positive. The first 500 steps are discarded so the tangent vector can align.

## Usage

```bash
pip install numpy matplotlib numba
python hopalong_phase_space.py --save                  # orbit, map and portrait as PNG
python hopalong_phase_space.py --which header --save   # 1600 x 900 header image
python hopalong_phase_space.py --a 1.1 --b 0.5 --c 0.3 --iters 2000000
```

Without `--save` the figures open in windows. The default run of all three views took about 40 s on a single core here, including compilation. The map is the slow part and runs in parallel across cores. The main options:

- `--a`, `--b`, `--c`: map parameters (default 0.7, 0.4, 0)
- `--which`: `all` (orbit, map, portrait), `orbit`, `map`, `portrait` or `header`
- `--iters`: points of the single orbit from (0,0) (default 10,000,000)
- `--lyap-iters`, `--grid`: iterations per orbit and grid points per axis for the map (default 8000 and 400)
- `--thresh`: exponent above which an orbit counts as chaotic (default 0.004)
- `--window-scale`, `--xlim`, `--ylim`: plot window of the map
- `--portrait-orbits`, `--portrait-iters`, `--start`, `--end`: number of orbits, points per orbit and the line of starting points
- `--header-size`, `--header-zoom`: header pixels (default 1600 900) and crop (default 1.4)
- `--outdir`: folder for the PNG files

The plot window of the map is the box around the orbit from (0,0), enlarged by `--window-scale` (default 3). For the default parameters that is x from -2.53 to 2.79 and y from -2.09 to 3.23. The portrait line is placed relative to the same box. Both are only guesses for other parameters.

## How to read the figures

**Orbit:** brightness is the log of how often the orbit visited each pixel. Black regions were never visited by this orbit.

## Map

- Left panel: colour is the finite-time exponent of the orbit that starts at that point. Dark means about 0, so regular. Bright means chaotic.
- Right panel: the same data split at the threshold. Blue is regular, orange is chaotic, grey would mean the orbit escaped.
- Large dark or blue blobs are islands of regular motion. The cross marks (0,0).

## Portrait

- Each colour is one orbit. Cool colours (cyan to magenta) are regular orbits and only encode the position of the starting point along the line, so they carry no measured value. Warm colours (yellow to red) are chaotic orbits, brighter for a larger exponent.
- Nested rings are regular orbits circling an island. Dense warm bands are chaotic orbits filling the space between islands.
- The starting line is an arbitrary choice. Islands it does not cross may be missing.

## What the run shows

Results for the default settings (`a=0.7`, `b=0.4`, `c=0`, 400 x 400 grid, N=8000, threshold 0.004):

1. 69.8% of starting points are regular and 30.2% are chaotic in the default window. None escaped.
2. The orbit from (0,0) has λ ≈ 0.018 (10 million iterations; 30 million gave 0.0182, while a 2 million run gave 0.0199, so single-orbit estimates vary by about 10%). The classic picture is therefore the footprint of one chaotic orbit. The regions it leaves empty are regular islands it does not enter.
3. Starting points far away, such as (50,50), (-200,30) and (1000,-1000), gave λ below 3e-4, which is regular. The chaotic share is 31.3% for a window 6 times the orbit box and 19.9% for 12 times, so the percentage depends on the window.
4. Exponents inside the chaotic region vary, from about 0.01 to about 0.03 in the figure. The script does not explain why.
5. The chaotic share is 29-31% for N from 4000 to 20000 and thresholds from 0.002 to 0.004. At a threshold of 0.008 it falls to about 25-26%.

Visible in the portrait but not tested: a ring of seven large islands, each filled with nested rings, and an inner ring of about ten smaller ones. This is consistent with chains of islands that a single orbit visits one after another.

## Caveats

- Only one parameter set was examined. Five other parameter sets ran without errors, but their pictures were not studied, and nothing here says how the picture changes with `a`, `b`, `c`.
- The percentages depend on the window and describe the plotted region, not the whole plane.
- The exponent is finite-time. Chaotic orbits that stick near islands can look regular over N steps, so the chaotic share can be under-counted near island boundaries.
- Islands smaller than the grid spacing (about 0.013 at the defaults) are not resolved.
- The square-root term has an unbounded derivative at `bx=c`. A floor of 1e-12 on the square root keeps the code finite, and smooth-map theory such as KAM is only an analogy here.
- Orbits are computed in floating point. Exponents and fractions were checked only by varying N, threshold and window.

## References

[1]
J. Lansdown and R. A. Earnshaw (eds.), Computers in Art, Design and Animation.
New York: Springer-Verlag, 1989.
e-ISBN-13: 978-1-4612-4538-4.

[2]
Barry Martin, "Graphic Potential of Recursive Functions," in Computers in Art, Design and Animation [1],
pp. 109–129.

[3]
A.K. Dewdney, Psychotapeten, algorithm HÜPFER, in Spektrum der Wissenschaft: Computer Kurzweil.
Spektrum der Wissenschaft Verlagsgesellschaft mbH & Co., Heidelberg, 1988.
(German version of Scientific American).
ISBN-10: 3922508502, ISBN-13: 978-3922508502.
