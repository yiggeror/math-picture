"""Synthesise the soundtrack, timed to the rounds in config.json.

every round   a bell on the round's first frame, one scale step higher each time the circles double,
              over a chord that moves with the rounds; the arpeggio gets faster as the circles multiply
round 5 on    a heartbeat kick that quickens
last round    the arpeggio races with the pen, a riser and a snare roll build,
              a beat of silence, then the hit: sub drop, crash, a wide D major chord
after         D  A  Bm  G at the config bpm, with a swell on every ripple of light; fade out

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
BEAT = 60 / CFG["bpm"]
GAP = 0.25                      # silence right before the hit
rng = np.random.default_rng(5)

ROUND_START = []
_t = CFG["intro"]
for r in CFG["rounds"]:
    ROUND_START.append(_t)
    _t += r["seconds"]
CLIMAX = _t
END = CLIMAX + CFG["after"]
N = int((END + 1) * SR)
dry = np.zeros((N, 2))
send = np.zeros((N, 2))         # reverb send

NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10, "B": 11}


def hz(note):
    """'D3' -> frequency."""
    return 440.0 * 2 ** ((NOTE[note[:-1]] + 12 * (int(note[-1]) + 1) - 69) / 12)


def env(n, a, d, s, r, hold):
    """ADSR in seconds: attack, decay time constant, sustain level, release time constant, hold before release."""
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


def pluck(f, length=0.8, cutoff=4000):
    t = np.arange(int(length * SR)) / SR
    x = signal.sawtooth(2 * np.pi * f * t) * 0.6 + np.sin(2 * np.pi * f * t)
    return lowpass(x, cutoff) * np.exp(-t / 0.2) * np.minimum(1, t / 0.003)


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


TRIADS = {"Dm": ["D", "F", "A"], "Bb": ["Bb", "D", "F"], "F": ["F", "A", "C"], "C": ["C", "E", "G"],
          "Gm": ["G", "Bb", "D"], "A": ["A", "C#", "E"], "D": ["D", "F#", "A"], "Bm": ["B", "D", "F#"],
          "G": ["G", "B", "D"]}


def triad(chord, octave):
    out = []
    for n in TRIADS[chord]:
        f = hz(n + str(octave))
        if out and f < out[-1]:
            f *= 2
        out.append(f)
    return out


# ---------- the rounds ----------
NR = len(CFG["rounds"])
ROUND_CHORDS = ["Dm", "Dm", "Bb", "Bb", "F", "F", "C", "C", "Dm", "Bb", "F", "C", "Gm", "A", "A"]
SCALE = ["D4", "E4", "F4", "G4", "A4", "Bb4", "C5", "D5", "E5", "F5", "G5", "A5", "Bb5", "C6", "D6"]
build_end = CLIMAX - GAP

# drone under everything until the hit, plus an airy pad that phone speakers can play
t = np.arange(int((build_end + 0.05) * SR)) / SR
drone = np.sin(2 * np.pi * hz("D1") * t) * 0.8 + np.sin(2 * np.pi * hz("D2") * t) * 0.4
drone *= np.minimum(1, t / 2) * (0.5 + 0.5 * (t / build_end) ** 2)
place(drone, 0, gain=0.3, rev=0.2)

for k in range(NR):
    start = ROUND_START[k]
    end = ROUND_START[k + 1] if k + 1 < NR else build_end
    last = k == NR - 1
    prog = k / (NR - 1)
    chord = ROUND_CHORDS[k]
    # the round's bell: one step up the scale each time the circles double
    place(bell(hz(SCALE[k]), 3.0, 0.8 + 0.4 * prog), start, gain=0.16, pan=0.3 * np.sin(k), rev=0.7)
    place(bell(hz(SCALE[k]) * 2, 2.0, 0.6), start, gain=0.05, pan=-0.3 * np.sin(k), rev=0.8)
    if k > 0:
        place(kick(0.4 + 0.6 * prog), start, gain=0.25 + 0.3 * prog, rev=0.15)
    # pad
    for f in triad(chord, 3):
        x = supersaw(f, end - start + 1.0, 400 + 4500 * prog ** 1.4, voices=5)
        x *= env(len(x), min(0.4, (end - start) / 3), 1.0, 1.0, 0.25, end - start)
        place(x, start, gain=0.03 + 0.04 * prog, pan=rng.uniform(-0.35, 0.35), rev=0.5)
    # arpeggio: faster as the circles multiply; in the last round it races with the pen
    if k >= 1:
        tones = triad(chord, 4 if k < 8 else 5)
        tones = tones + [f * 2 for f in tones]
        pattern = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]
        tt, j = start, 0
        while tt < end - 0.02:
            if last:
                u = (tt - start) / (end - start)
                rate = 8 * np.exp(np.log(32 / 8) * u)
            else:
                rate = 1.5 + 1.5 * k
            place(pluck(tones[pattern[j % len(pattern)]], 0.6, 1200 + 7000 * prog), tt,
                  gain=0.05 + 0.06 * prog, pan=0.5 * np.sin(j * 0.7), rev=0.45)
            tt += 1 / rate
            j += 1

# heartbeat from round 5, quickening into the last round
hb_start = ROUND_START[5]
tt = hb_start
while tt < build_end - 0.05:
    q = (tt - hb_start) / (build_end - hb_start)
    place(kick(0.4 + 0.6 * q), tt, gain=0.3, rev=0.08)
    tt += 0.9 * (0.2 / 0.9) ** q

# riser and snare roll across the last round
rs = ROUND_START[-1]
t = np.arange(int((build_end - rs) * SR)) / SR
p = t / t[-1]
sweep = np.sin(2 * np.pi * np.cumsum(120 * (2500 / 120) ** p) / SR)
noise = highpass(rng.standard_normal(len(t)), 1500)
place((0.3 * sweep + 0.2 * noise) * p ** 2.5, rs, gain=0.2, rev=0.4)
tt = build_end - 3.0
while tt < build_end - 0.03:
    q = (tt - (build_end - 3.0)) / 3.0
    place(snare(), tt, gain=0.05 + 0.17 * q ** 2, pan=rng.uniform(-0.2, 0.2), rev=0.3)
    tt += 0.25 * (0.035 / 0.25) ** q

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
start = CLIMAX + 0.9            # the first ripple of light (scene.html rippleLevel)
bar = 4 * BEAT
tt, i = start, 0
while tt < END - 1.5:
    c = ["D", "A", "Bm", "G"][i % 4]
    for f in triad(c, 3):
        x = supersaw(f, bar + 1.5, 2200, voices=5) * env(int((bar + 1.5) * SR), 0.6, 1.0, 1.0, 0.6, bar)
        place(x, tt, gain=0.07, pan=rng.uniform(-0.4, 0.4), rev=0.6)
    place(np.sin(2 * np.pi * hz(TRIADS[c][0] + "1") * np.arange(int(bar * SR)) / SR)
          * env(int(bar * SR), 0.3, 1, 1, 0.5, bar - 0.4), tt, gain=0.32, rev=0.1)
    tones = triad(c, 5) + [f * 2 for f in triad(c, 5)]
    for j in range(8):  # eighth notes
        place(bell(tones[[0, 2, 1, 3, 2, 4, 3, 5][j]], 1.8, 0.9), tt + j * BEAT / 2,
              gain=0.09, pan=0.6 * np.sin(j), rev=0.7)
    tt += bar
    i += 1
tt = start
while tt < END - 2:             # a swell with every ripple (one every two beats)
    t = np.arange(int(1.2 * SR)) / SR
    place(np.sin(2 * np.pi * hz("D2") * t) * np.sin(np.pi * t / 1.2) ** 2, tt, gain=0.2, rev=0.3)
    place(kick(0.7), tt, gain=0.45, rev=0.15)
    tt += 2 * BEAT

# ---------- reverb + master ----------
ir_len = int(3.2 * SR)
ti = np.arange(ir_len) / SR
ir = np.column_stack([lowpass(rng.standard_normal(ir_len), 6000) * np.exp(-ti / 0.9) for _ in range(2)])
ir /= np.sqrt((ir ** 2).sum(0))
wet = np.column_stack([signal.fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)])
mix = dry + 0.9 * wet

# the gap before the hit: a hard cut to silence, then the hit
gs, ge = int((CLIMAX - GAP) * SR), int(CLIMAX * SR)
mix[gs - 400:gs] *= np.linspace(1, 0, 400)[:, None]
mix[gs:ge] = 0

fs = int((END - CFG["fadeOut"]) * SR)
mix[fs:] *= (np.clip(1 - np.arange(N - fs) / (CFG["fadeOut"] * SR), 0, 1) ** 1.5)[:, None]
mix /= np.abs(mix).max()
mix = np.tanh(mix * 1.8) / np.tanh(1.8)   # gentle limiter; the hit is the loudest moment
mix *= 0.89 / np.abs(mix).max()

out = os.path.join(HERE, "music.wav")
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print("wrote", out, f"{END:.1f}s, climax at {CLIMAX:.2f}s")
