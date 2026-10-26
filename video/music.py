"""Synthesise the soundtrack, timed to config.json.

0 s        deep drone, a few glassy notes in the dark
8 s        arpeggio and pad start; the arpeggio speeds up exponentially, like the pen
14 s       a heartbeat kick joins and quickens
26 s       riser, then a snare roll; Dm Bb F C | Dm Bb F A
climax     a beat of silence, then the hit: sub drop, crash, a wide D major chord
after      D  A  Bm  G at the config bpm; a soft swell with every light ripple
end        fade out

Usage: python3 music.py   -> music.wav
"""
import json
import os

import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
SR = 44100
DUR = CFG["end"] + 1.0
CLIMAX = CFG["climax"]
BEAT = 60 / CFG["bpm"]
GAP = 0.25                      # silence right before the hit
rng = np.random.default_rng(5)

N = int(DUR * SR)
dry = np.zeros((N, 2))
send = np.zeros((N, 2))         # reverb send


def hz(note):
    """'D3' -> frequency."""
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "Ab": 8,
             "A": 9, "A#": 10, "Bb": 10, "B": 11}
    name, octave = note[:-1], int(note[-1])
    return 440.0 * 2 ** ((names[name] + 12 * (octave + 1) - 69) / 12)


def env(n, a, d, s, r, hold):
    """ADSR envelope in samples-free seconds: attack, decay, sustain level, release, hold time before release."""
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = t > hold
    e[rel] *= np.exp(-(t[rel] - hold) / max(r, 1e-4))
    return e


def place(sig, start, gain=1.0, pan=0.0, rev=0.3):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[:N - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    st = np.column_stack([sig * l, sig * r]) * gain
    dry[i:i + len(sig)] += st
    send[i:i + len(sig)] += st * rev


def lowpass(x, cutoff, order=2):
    b, a = signal.butter(order, min(cutoff, SR / 2 - 100) / (SR / 2))
    return signal.lfilter(b, a, x)


def highpass(x, cutoff, order=2):
    b, a = signal.butter(order, cutoff / (SR / 2), "high")
    return signal.lfilter(b, a, x)


def bandpass(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], "band")
    return signal.lfilter(b, a, x)


# ---------- instruments ----------
def bell(f, length=2.5, bright=1.0):
    t = np.arange(int(length * SR)) / SR
    parts = [(1, 1.0, 1.6), (2.76, 0.45 * bright, 0.7), (5.4, 0.22 * bright, 0.35), (8.93, 0.1 * bright, 0.18)]
    x = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t / d) for m, a, d in parts)
    return x * np.minimum(1, t / 0.002)


def pluck(f, length=0.9, cutoff=4000):
    t = np.arange(int(length * SR)) / SR
    x = signal.sawtooth(2 * np.pi * f * t) * 0.6 + np.sin(2 * np.pi * f * t)
    x = lowpass(x, cutoff) * np.exp(-t / 0.22) * np.minimum(1, t / 0.003)
    return x


def supersaw(f, length, cutoff, voices=7, detune=0.012):
    t = np.arange(int(length * SR)) / SR
    x = np.zeros_like(t)
    for v in range(voices):
        d = 1 + detune * (v - (voices - 1) / 2) / ((voices - 1) / 2)
        x += signal.sawtooth(2 * np.pi * f * d * t + rng.random() * 6.28)
    return lowpass(x / voices, cutoff, 2)


def kick(gain=1.0):
    t = np.arange(int(0.6 * SR)) / SR
    f = 45 + 95 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    x += 0.3 * highpass(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.004)
    return np.tanh(1.6 * x) * gain


def snare():
    t = np.arange(int(0.25 * SR)) / SR
    n = bandpass(rng.standard_normal(len(t)), 900, 6000) * np.exp(-t / 0.06)
    return n + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)


# ---------- harmony ----------
MINOR = {"Dm": ["D", "F", "A"], "Bb": ["Bb", "D", "F"], "F": ["F", "A", "C"], "C": ["C", "E", "G"],
         "A": ["A", "C#", "E"], "D": ["D", "F#", "A"], "Bm": ["B", "D", "F#"], "G": ["G", "B", "D"]}
build_end = CLIMAX - GAP
CHORDS = [(8, "Dm"), (12, "Bb"), (16, "F"), (20, "C"), (24, "Dm"), (27, "Bb"), (29.5, "F"), (31.3, "A")]


def chord_at(t):
    cur = CHORDS[0][1]
    for s, c in CHORDS:
        if t >= s:
            cur = c
    return cur


def notes(chord, octave):
    out = []
    for n in MINOR[chord]:
        f = hz(n + str(octave))
        if out and f < out[-1]:
            f *= 2
        out.append(f)
    return out


# ---------- intro: drone + glass ----------
t = np.arange(int((build_end + 0.1) * SR)) / SR
drone = (np.sin(2 * np.pi * hz("D1") * t) * 0.9 + np.sin(2 * np.pi * hz("D2") * t) * 0.35
         + 0.15 * lowpass(signal.sawtooth(2 * np.pi * hz("A1") * t), 300))
drone *= np.minimum(1, t / 4) * (0.55 + 0.45 * np.minimum(1, t / build_end) ** 2)
place(drone, 0, gain=0.32, rev=0.2)
# an airy pad an octave or two up, so phone speakers have something to play before the build
for n in ["D3", "A3", "E4"]:
    x = supersaw(hz(n), 9.5, 900, voices=5, detune=0.008) * env(int(9.5 * SR), 4.0, 1, 1, 1.2, 7.5)
    place(x, 0, gain=0.035, pan=rng.uniform(-0.5, 0.5), rev=0.7)
for tt, n in [(1.2, "A5"), (2.9, "D6"), (4.1, "F5"), (5.6, "C6"), (6.4, "A5"), (7.3, "E6")]:
    place(bell(hz(n), 4, 0.8), tt, gain=0.11, pan=rng.uniform(-0.6, 0.6), rev=0.8)

# ---------- build: accelerating arpeggio ----------
r0, r1 = 3.0, 26.0
a = np.log(r1 / r0) / (build_end - 8)
tt, k = 8.0, 0
pattern = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]
while tt < build_end - 0.02:
    rate = r0 * np.exp(a * (tt - 8))
    chord = chord_at(tt)
    octave = 4 if tt < 20 else 5
    tones = notes(chord, octave) + [f * 2 for f in notes(chord, octave)]
    f = tones[pattern[k % len(pattern)]]
    prog = (tt - 8) / (build_end - 8)
    place(pluck(f, 0.7, 1500 + 6000 * prog), tt, gain=0.06 + 0.07 * prog, pan=0.5 * np.sin(k * 0.7), rev=0.45)
    tt += 1 / rate
    k += 1

# pad, filter opening over the build
for i, (s, c) in enumerate(CHORDS):
    e = CHORDS[i + 1][0] if i + 1 < len(CHORDS) else build_end
    prog = (s - 8) / (build_end - 8)
    for f in notes(c, 3):
        x = supersaw(f, e - s + 1.2, 500 + 4500 * prog ** 1.5, voices=5)
        x *= env(len(x), 0.5, 1.0, 1.0, 0.25, e - s)
        place(x, s, gain=0.03 + 0.045 * prog, pan=rng.uniform(-0.3, 0.3), rev=0.5)

# heartbeat that quickens
tt, interval = 14.0, 1.0
while tt < build_end - 0.05:
    prog = (tt - 14) / (build_end - 14)
    place(kick(0.5 + 0.5 * prog), tt, gain=0.32, rev=0.08)
    interval = 1.0 * (0.22 / 1.0) ** prog
    tt += interval

# riser and snare roll
rs, re = 26.0, build_end
t = np.arange(int((re - rs) * SR)) / SR
p = t / (re - rs)
sweep = np.sin(2 * np.pi * np.cumsum(150 * (3000 / 150) ** p) / SR)
noise = highpass(rng.standard_normal(len(t)), 1500)
place((0.3 * sweep + 0.2 * noise) * p ** 2.5, rs, gain=0.18, rev=0.4)
tt, interval = 29.0, 0.25
while tt < build_end - 0.03:
    prog = (tt - 29) / (build_end - 29)
    place(snare(), tt, gain=0.05 + 0.16 * prog ** 2, pan=rng.uniform(-0.2, 0.2), rev=0.3)
    tt += 0.25 * (0.035 / 0.25) ** prog

# ---------- the hit ----------
t = np.arange(int(4 * SR)) / SR
sub = np.sin(2 * np.pi * np.cumsum(28 + 32 * np.exp(-t / 0.6)) / SR) * np.exp(-t / 1.6)
place(np.tanh(2 * sub), CLIMAX, gain=0.9, rev=0.1)
place(kick(1.0), CLIMAX, gain=0.9, rev=0.2)
tail = np.exp(-t / 0.9) * np.cos(np.pi / 2 * t / t[-1])  # decays all the way to zero
for pan in (-0.35, 0.35):
    place(highpass(rng.standard_normal(len(t)), 4000) * tail, CLIMAX, gain=0.13, pan=pan, rev=0.5)
for i, n in enumerate(["D2", "A2", "D3", "F#3", "A3", "D4", "F#4", "A4", "D5"]):
    x = supersaw(hz(n), 7, 7000, voices=7, detune=0.018) * env(int(7 * SR), 0.01, 2.5, 0.35, 1.5, 5)
    place(x, CLIMAX, gain=0.1, pan=(i / 8 - 0.5) * 1.4, rev=0.6)
for i, n in enumerate(["D6", "A6", "F#6", "D7"]):
    place(bell(hz(n), 5, 1.2), CLIMAX + 0.05 + i * 0.06, gain=0.1, pan=(i - 1.5) / 2, rev=0.8)

# ---------- afterglow ----------
start = CLIMAX + CFG["wave"] * 0.6
after = ["D", "A", "Bm", "G"]
bar = 4 * BEAT
tt, i = start, 0
while tt < CFG["end"] - 1.5:
    c = after[i % 4]
    for f in notes(c, 3):
        x = supersaw(f, bar + 1.5, 2200, voices=5) * env(int((bar + 1.5) * SR), 0.6, 1.0, 1.0, 0.6, bar)
        place(x, tt, gain=0.07, pan=rng.uniform(-0.4, 0.4), rev=0.6)
    place(np.sin(2 * np.pi * hz(MINOR[c][0] + "1") * np.arange(int(bar * SR)) / SR)
          * env(int(bar * SR), 0.3, 1, 1, 0.5, bar - 0.4), tt, gain=0.32, rev=0.1)
    tones = notes(c, 5) + [f * 2 for f in notes(c, 5)]
    for j in range(8):  # eighth notes
        place(bell(tones[[0, 2, 1, 3, 2, 4, 3, 5][j]], 1.8, 0.9), tt + j * BEAT / 2,
              gain=0.09, pan=0.6 * np.sin(j), rev=0.7)
    tt += bar
    i += 1
# a swell with every ripple of light (one every two beats)
tt = start
while tt < CFG["end"] - 2:
    t = np.arange(int(1.2 * SR)) / SR
    swell = np.sin(2 * np.pi * hz("D2") * t) * np.sin(np.pi * t / 1.2) ** 2
    place(swell, tt, gain=0.2, rev=0.3)
    place(kick(0.7), tt, gain=0.45, rev=0.15)
    tt += 2 * BEAT

# ---------- reverb + master ----------
ir_len = int(3.2 * SR)
ti = np.arange(ir_len) / SR
ir = np.column_stack([lowpass(rng.standard_normal(ir_len), 6000) * np.exp(-ti / 0.9) for _ in range(2)])
ir /= np.sqrt((ir ** 2).sum(0))
wet = np.column_stack([signal.fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)])
mix = dry + 0.9 * wet

# the gap before the hit: everything ducks to silence, then comes back with the hit
g = np.ones(N)
gs, ge = int((CLIMAX - GAP) * SR), int(CLIMAX * SR)
g[gs:ge] = 0
g[gs - 400:gs] = np.linspace(1, 0, 400)
mix[:ge] *= g[:ge, None]

fade = np.ones(N)
fs = int((CFG["end"] - CFG["fadeOut"]) * SR)
fade[fs:] = np.linspace(1, 0, N - fs) ** 1.5
mix *= fade[:, None]
mix /= np.abs(mix).max()
mix = np.tanh(mix * 1.8) / np.tanh(1.8)   # gentle limiter; the hit is the loudest moment
mix *= 0.89 / np.abs(mix).max()

out = os.path.join(HERE, "music.wav")
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print("wrote", out, f"{DUR:.1f}s")
