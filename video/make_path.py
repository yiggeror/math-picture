"""Turn The Starry Night into one closed line, then into Fourier epicycles.

painting -> brush-stroke flow field -> evenly spaced streamlines (one per "brush stroke")
         -> strokes chained ring by ring outward from the central swirl into one closed line
         -> arc-length resample -> FFT -> circles
         -> oversampled reconstruction of the line the circles actually draw

Writes into data/:
  meta.json   sizes, focus point, colour palette
  terms.bin   circles, largest first: int32 freq[K], float32 amp[K], float32 phase[K]
  trail.bin   float32 (x, y) of the pen, OVERSAMPLE points per sample
  attr.bin    per trail point: uint8 pen_up, uint8 palette index

Usage: python3 make_path.py
"""
import colorsys
import json
import os

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "assets/starry_night.jpg")
OUT = os.path.join(HERE, "data")

W = 1080                 # painting width in final-frame pixels
FOCUS = (585.0, 300.0)   # centre of the big swirl; the drawing spirals out from here
DSEP = 4.0               # spacing between neighbouring strokes
LMAX = 60                # longest stroke
TENSOR_SIGMA = 5         # smoothing of the brush direction field
RING_GROWTH = 1.1        # each ring is 10% farther out than the last
N_SAMPLES = 2 ** 18      # samples along the line
N_TERMS = 2 ** 17        # circles kept, largest first
OVERSAMPLE = 4           # extra trail points per sample, so close-ups stay smooth
PALETTE = 64

rng = np.random.default_rng(1)


def load():
    im = Image.open(SRC).convert("RGB")
    h = round(im.height * W / im.width)
    return im.resize((W, h), Image.LANCZOS)


def direction_field(rgb):
    g = ndi.gaussian_filter(rgb.mean(2), 1.2)
    gx, gy = ndi.sobel(g, 1), ndi.sobel(g, 0)
    jxx = ndi.gaussian_filter(gx * gx, TENSOR_SIGMA)
    jxy = ndi.gaussian_filter(gx * gy, TENSOR_SIGMA)
    jyy = ndi.gaussian_filter(gy * gy, TENSOR_SIGMA)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # along the brush, across the gradient
    return np.cos(theta), np.sin(theta)


def streamlines(dx, dy):
    """Jobard–Lefer style: trace from random seeds, stop near ink already laid down."""
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
        if len(pts) < 6:
            continue
        lines.append(np.array(pts))
        for x, y in pts:
            xi, yi = int(x), int(y)
            y0, y1, x0, x1 = max(0, yi - r), min(h, yi + r + 1), max(0, xi - r), min(w, xi + r + 1)
            mask[y0:y1, x0:x1] |= stamp[y0 - yi + r:y1 - yi + r, x0 - xi + r:x1 - xi + r]
    return lines


def spiral_chain(lines):
    """Grow the drawing outward from FOCUS ring by ring; inside a ring, always hop to the nearest stroke end.

    Rings widen with distance (log-spaced), so the drawing grows at the same pace the camera pulls back."""
    fx, fy = FOCUS
    mids = np.array([l[len(l) // 2] for l in lines])
    ring = np.floor(np.log1p(np.hypot(mids[:, 0] - fx, mids[:, 1] - fy) / 8) / np.log(RING_GROWTH)).astype(int)
    starts = np.array([l[0] for l in lines])
    ends = np.array([l[-1] for l in lines])
    path, pen_up = [], []
    cur = np.array(FOCUS)
    for rg in np.unique(ring):
        idx = np.flatnonzero(ring == rg)
        left = np.ones(len(idx), bool)
        for _ in range(len(idx)):
            ds = np.where(left, np.hypot(*(starts[idx] - cur).T), np.inf)
            de = np.where(left, np.hypot(*(ends[idx] - cur).T), np.inf)
            k = int(np.argmin(np.minimum(ds, de)))
            left[k] = False
            l = lines[idx[k]] if ds[k] <= de[k] else lines[idx[k]][::-1]
            path.append(l)
            up = np.zeros(len(l), bool)
            up[-1] = True  # the segment leaving a stroke's last point hops to the next stroke
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


def neon_palette(im, pts, jitter):
    """Painting colour under each sample, pushed brighter and more saturated, quantised to PALETTE colours."""
    col = np.asarray(im.filter(ImageFilter.GaussianBlur(1.5)), np.float64) / 255
    xi = np.clip(pts[:, 0].astype(int), 0, col.shape[1] - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, col.shape[0] - 1)
    c = col[yi, xi]
    out = np.empty_like(c)
    for i, (rgb, j) in enumerate(zip(c, jitter)):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        out[i] = colorsys.hls_to_rgb((h + j) % 1, 0.42 + 0.45 * l ** 0.8, min(1, 0.55 + s * 1.4))
    img = Image.fromarray((out[None] * 255).astype(np.uint8))
    q = img.quantize(PALETTE, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[:PALETTE * 3]).reshape(-1, 3)
    return np.asarray(q)[0].astype(np.uint8), pal.tolist()


def main():
    os.makedirs(OUT, exist_ok=True)
    im = load()
    rgb = np.asarray(im, np.float64) / 255
    lines = streamlines(*direction_field(rgb))
    poly, up = spiral_chain(lines)
    stroke_id = np.cumsum(np.concatenate([[0], up[:-1]]))  # which stroke each vertex belongs to
    jitter = rng.normal(0, 0.045, len(lines))               # small per-stroke hue shift for variety

    samples, vertex, length = resample(poly, N_SAMPLES)
    s_up = up[vertex]
    pal_idx, palette = neon_palette(im, samples, jitter[stroke_id[vertex]])

    z = samples[:, 0] + 1j * samples[:, 1]
    c = np.fft.fft(z) / len(z)
    freq = np.fft.fftfreq(len(z), d=1.0 / len(z)).astype(np.int64)
    keep = np.argsort(-np.abs(c))[:N_TERMS]
    keep = np.concatenate([[0], keep[keep != 0]])[:N_TERMS]  # constant term first: it anchors the chain

    # what the kept circles really draw, OVERSAMPLE points per sample
    m = len(z) * OVERSAMPLE
    big = np.zeros(m, complex)
    big[freq[keep] % m] = c[keep]
    trail = np.fft.ifft(big) * m
    err = np.abs(trail[::OVERSAMPLE] - z)

    with open(os.path.join(OUT, "terms.bin"), "wb") as f:
        f.write(freq[keep].astype(np.int32).tobytes())
        f.write(np.abs(c[keep]).astype(np.float32).tobytes())
        f.write(np.angle(c[keep]).astype(np.float32).tobytes())
    with open(os.path.join(OUT, "trail.bin"), "wb") as f:
        f.write(np.column_stack([trail.real, trail.imag]).astype(np.float32).tobytes())
    attr = np.column_stack([np.repeat(s_up, OVERSAMPLE), np.repeat(pal_idx, OVERSAMPLE)]).astype(np.uint8)
    with open(os.path.join(OUT, "attr.bin"), "wb") as f:
        f.write(attr.tobytes())
    meta = {
        "samples": len(z), "oversample": OVERSAMPLE, "terms": len(keep),
        "paintingWidth": W, "paintingHeight": im.height, "focus": FOCUS, "palette": palette,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"{len(lines)} strokes, line {length:.0f}px ({s_up.mean():.0%} pen-up), "
          f"{len(keep)} circles, error mean {err.mean():.3f}px max {err.max():.2f}px")


if __name__ == "__main__":
    main()
