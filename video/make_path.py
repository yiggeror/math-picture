"""Turn a picture into one closed line, then into Fourier epicycles.

image -> brightness contour lines (a "topographic" portrait)
      -> chained greedily into one continuous closed line
      -> arc-length resample -> FFT -> terms.json (freq, amp, phase), largest first,
         plus the sample ranges where the line only hops between contours ("pen up")

Usage: python3 make_path.py assets/mona_lisa.jpg terms.json
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage import filters, measure

CROP = (60, 40, 450, 620)        # head, shoulders and hands of the 500x745 source
BLUR = 1.8                       # smooths contours; smaller = more wiggly detail
LEVELS = np.linspace(0.12, 0.88, 11)
MIN_LEN = 30                     # drop contour scraps shorter than this (source px)
N_SAMPLES = 32768                # samples along the line
N_TERMS = 16384                  # epicycles kept, largest first
BOX = (50, 380, 1030, 1860)      # where the drawing sits in the 1080x1920 frame


def contours(path):
    im = Image.open(path).convert("L").crop(CROP)
    im = ImageOps.autocontrast(im, cutoff=1)
    g = filters.gaussian(np.asarray(im, dtype=np.float64) / 255.0, sigma=BLUR)
    out = []
    for lv in LEVELS:
        for c in measure.find_contours(g, lv):
            if np.hypot(*np.diff(c, axis=0).T).sum() >= MIN_LEN:
                out.append(c[:, ::-1].copy())  # (row, col) -> (x, y)
    return out


def chain(lines):
    """Visit every contour once, always hopping to the nearest unvisited one.

    Returns the joined polyline and, per vertex, whether the segment leaving it is a hop."""
    closed = [np.allclose(l[0], l[-1]) for l in lines]
    lengths = [np.hypot(*np.diff(l, axis=0).T).sum() for l in lines]
    todo = set(range(len(lines)))
    first = int(np.argmax(lengths))
    todo.remove(first)
    path = [lines[first]]
    cur = lines[first][-1]
    while todo:
        best = None
        for i in todo:
            l = lines[i]
            if closed[i]:
                d = np.hypot(*(l - cur).T)
                k = int(np.argmin(d))
                cand = (d[k], i, k)
            else:
                d0, d1 = np.hypot(*(l[0] - cur)), np.hypot(*(l[-1] - cur))
                cand = (d0, i, 0) if d0 <= d1 else (d1, i, -1)
            if best is None or cand[0] < best[0]:
                best = cand
        _, i, k = best
        todo.remove(i)
        l = lines[i]
        if closed[i]:
            l = np.vstack([l[k:-1], l[:k + 1]])  # rotate the loop to start where we enter
        elif k == -1:
            l = l[::-1]
        path.append(l)
        cur = l[-1]
    hop = np.zeros(sum(len(l) for l in path), bool)
    ends = np.cumsum([len(l) for l in path]) - 1
    hop[ends] = True  # last vertex of each contour hops to the next (the final one closes the loop)
    return np.vstack(path), hop


def resample(poly, hop, n):
    closed = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    pts = np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])
    on_hop = hop[np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(hop) - 1)]
    return pts, on_hop


def ranges(mask):
    """[[start, end), ...] index ranges where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)]).tolist()


def fit_box(poly):
    x0, y0, x1, y1 = BOX
    mn, mx = poly.min(0), poly.max(0)
    s = min((x1 - x0) / (mx[0] - mn[0]), (y1 - y0) / (mx[1] - mn[1]))
    return (poly - (mn + mx) / 2) * s + [(x0 + x1) / 2, (y0 + y1) / 2]


def main(src, out):
    lines = contours(src)
    line, hop = chain(lines)
    line = fit_box(line)
    samples, on_hop = resample(line, hop, N_SAMPLES)

    z = samples[:, 0] + 1j * samples[:, 1]
    c = np.fft.fft(z) / len(z)
    freq = np.fft.fftfreq(len(z), d=1.0 / len(z)).astype(int)
    rest = np.argsort(-np.abs(c[1:]))[:N_TERMS] + 1
    order = np.concatenate([[0], rest])  # DC anchors the chain, then biggest circles first
    terms = [[int(freq[k]), round(float(abs(c[k])), 5), round(float(np.angle(c[k])), 5)] for k in order]
    with open(out, "w") as f:
        json.dump({"samples": len(z), "hops": ranges(on_hop), "terms": terms}, f, separators=(",", ":"))

    zr = np.zeros(len(z), complex)
    zr[order] = c[order]
    err = np.abs(np.fft.ifft(zr) * len(z) - z)
    length = np.hypot(*np.diff(line, axis=0).T).sum()
    print(f"{len(lines)} contours, line length {length:.0f}px, {len(terms) - 1} circles, {on_hop.mean():.1%} pen-up, "
          f"reconstruction error mean {err.mean():.2f}px max {err.max():.2f}px")

    prev = Image.new("RGB", (1080, 1920), "black")
    draw = ImageDraw.Draw(prev)
    for i in range(len(samples)):
        if not on_hop[i]:
            draw.line([tuple(samples[i]), tuple(samples[(i + 1) % len(samples)])], fill=(255, 240, 210), width=1)
    prev.save(out.rsplit(".", 1)[0] + "_preview.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
