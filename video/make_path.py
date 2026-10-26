"""Turn The Starry Night into one closed line, then into Fourier circles, round by round.

painting -> brush direction field -> long, evenly spaced streamlines that follow Van Gogh's strokes
         -> chained (nearest end first, starting at the big swirl) into one closed line
         -> arc-length resample -> FFT -> circles sorted largest first
         -> for every round in config.json: the curve drawn by its first N circles

Writes into data/:
  meta.json   sample count, rounds, oversampling of the last round, palette
  terms.bin   int32 freq[K], float32 amp[K], float32 phase[K]   (K = most circles any round uses, +1)
  trails.bin  float32 (x, y) per point; every round has SAMPLES points, the last round SAMPLES*FINAL_OVERSAMPLE
  attr.bin    per sample: uint8 pen_up, uint8 palette index

Usage: python3 make_path.py
"""
import colorsys
import json
import os

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
SRC = os.path.join(HERE, "assets/starry_night.jpg")
OUT = os.path.join(HERE, "data")

W = 1080                 # painting width in frame pixels
FOCUS = (585.0, 300.0)   # centre of the big swirl; the line starts here
DSEP = 12.0              # spacing between neighbouring streamlines
LMAX = 500               # longest streamline
MIN_POINTS = 25          # drop shorter scraps
TENSOR_SIGMA = 7         # smoothing of the brush direction field
SAMPLES = 2 ** 15
FINAL_OVERSAMPLE = 4     # the last round is seen in close-up, so it gets a finer trail
PALETTE = 64

rng = np.random.default_rng(1)


def load():
    im = Image.open(SRC).convert("RGB")
    return im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)


def direction_field(rgb):
    g = ndi.gaussian_filter(rgb.mean(2), 1.2)
    gx, gy = ndi.sobel(g, 1), ndi.sobel(g, 0)
    jxx = ndi.gaussian_filter(gx * gx, TENSOR_SIGMA)
    jxy = ndi.gaussian_filter(gx * gy, TENSOR_SIGMA)
    jyy = ndi.gaussian_filter(gy * gy, TENSOR_SIGMA)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # along the brush, across the gradient
    return np.cos(theta), np.sin(theta)


def streamlines(dx, dy):
    """Jobard–Lefer style: trace from random seeds, stop near lines already laid down."""
    h, w = dx.shape
    mask = np.zeros((h, w), bool)
    r_test = DSEP * 0.55
    r = int(np.ceil(r_test))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    stamp = xx ** 2 + yy ** 2 <= r_test ** 2
    gy, gx = np.mgrid[0:h:DSEP, 0:w:DSEP]
    seeds = np.column_stack([gx.ravel(), gy.ravel()]) + rng.random((gx.size, 2)) * DSEP
    rng.shuffle(seeds)

    def free(x, y):
        xi, yi = int(x), int(y)
        return 0 <= xi < w and 0 <= yi < h and not mask[yi, xi]

    def trace(x, y, sgn):
        pts = []
        px, py = dx[int(y), int(x)] * sgn, dy[int(y), int(x)] * sgn
        for _ in range(LMAX // 2):
            vx, vy = dx[int(y), int(x)], dy[int(y), int(x)]
            if vx * px + vy * py < 0:
                vx, vy = -vx, -vy
            x, y = x + vx, y + vy
            if not free(x, y):
                break
            pts.append((x, y))
            px, py = vx, vy
        return pts

    lines = []
    for sx, sy in seeds:
        if not free(sx, sy):
            continue
        pts = trace(sx, sy, -1)[::-1] + [(sx, sy)] + trace(sx, sy, 1)
        if len(pts) < MIN_POINTS:
            continue
        lines.append(np.array(pts))
        for x, y in pts:
            xi, yi = int(x), int(y)
            y0, y1, x0, x1 = max(0, yi - r), min(h, yi + r + 1), max(0, xi - r), min(w, xi + r + 1)
            mask[y0:y1, x0:x1] |= stamp[y0 - yi + r:y1 - yi + r, x0 - xi + r:x1 - xi + r]
    return lines


def chain(lines):
    """One closed line: from the swirl, always continue with the nearest unused line end."""
    starts = np.array([l[0] for l in lines])
    ends = np.array([l[-1] for l in lines])
    left = np.ones(len(lines), bool)
    path, pen_up = [], []
    cur = np.array(FOCUS)
    for _ in range(len(lines)):
        ds = np.where(left, np.hypot(*(starts - cur).T), np.inf)
        de = np.where(left, np.hypot(*(ends - cur).T), np.inf)
        k = int(np.argmin(np.minimum(ds, de)))
        left[k] = False
        l = lines[k] if ds[k] <= de[k] else lines[k][::-1]
        path.append(l)
        up = np.zeros(len(l), bool)
        up[-1] = True  # the segment leaving a line's last point hops to the next line
        pen_up.append(up)
        cur = l[-1]
    return np.vstack(path), np.concatenate(pen_up)


def resample(poly, n):
    """n points at equal arc length around the closed polyline, plus the vertex each one follows."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    pts = np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])
    vertex = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(poly) - 1)
    return pts, vertex, s[-1]


def neon_palette(im, pts):
    """Painting colour under each sample, brighter and more saturated, quantised to PALETTE colours."""
    col = np.asarray(im.filter(ImageFilter.GaussianBlur(2)), np.float64) / 255
    xi = np.clip(pts[:, 0].astype(int), 0, col.shape[1] - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, col.shape[0] - 1)
    out = np.empty((len(pts), 3))
    for i, rgb in enumerate(col[yi, xi]):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        out[i] = colorsys.hls_to_rgb(h, 0.45 + 0.45 * l ** 0.8, min(1, 0.6 + s * 1.4))
    q = Image.fromarray((out[None] * 255).astype(np.uint8)).quantize(PALETTE, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[:PALETTE * 3]).reshape(-1, 3)
    return np.asarray(q)[0].astype(np.uint8), pal.tolist()


def curve(c, freq, keep, m):
    """What the circles in `keep` draw, at m points around the loop."""
    big = np.zeros(m, complex)
    big[freq[keep] % m] = c[keep]
    return np.fft.ifft(big) * m


def main():
    os.makedirs(OUT, exist_ok=True)
    im = load()
    lines = streamlines(*direction_field(np.asarray(im, np.float64) / 255))
    poly, up = chain(lines)
    samples, vertex, length = resample(poly, SAMPLES)
    pen_up = up[vertex]
    pal_idx, palette = neon_palette(im, samples)

    z = samples[:, 0] + 1j * samples[:, 1]
    c = np.fft.fft(z) / len(z)
    freq = np.fft.fftfreq(len(z), d=1.0 / len(z)).astype(np.int64)
    rest = np.argsort(-np.abs(c[1:])) + 1
    order = np.concatenate([[0], rest])  # the constant term anchors the chain, then biggest circles first

    rounds = [r["circles"] for r in CFG["rounds"]]
    with open(os.path.join(OUT, "trails.bin"), "wb") as f:
        for i, n in enumerate(rounds):
            m = SAMPLES * (FINAL_OVERSAMPLE if i == len(rounds) - 1 else 1)
            tr = curve(c, freq, order[:n + 1], m)
            f.write(np.column_stack([tr.real, tr.imag]).astype(np.float32).tobytes())
    err = np.abs(curve(c, freq, order[:rounds[-1] + 1], SAMPLES) - z)

    k = rounds[-1] + 1
    with open(os.path.join(OUT, "terms.bin"), "wb") as f:
        f.write(freq[order[:k]].astype(np.int32).tobytes())
        f.write(np.abs(c[order[:k]]).astype(np.float32).tobytes())
        f.write(np.angle(c[order[:k]]).astype(np.float32).tobytes())
    with open(os.path.join(OUT, "attr.bin"), "wb") as f:
        f.write(np.column_stack([pen_up, pal_idx]).astype(np.uint8).tobytes())
    meta = {
        "samples": SAMPLES, "finalOversample": FINAL_OVERSAMPLE, "terms": k, "rounds": rounds,
        "paintingWidth": W, "paintingHeight": im.height, "focus": FOCUS, "palette": palette,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"{len(lines)} strokes, line {length:.0f}px ({pen_up.mean():.0%} pen-up), "
          f"{rounds[-1]} circles in the last round, error mean {err.mean():.2f}px max {err.max():.2f}px")


if __name__ == "__main__":
    main()
