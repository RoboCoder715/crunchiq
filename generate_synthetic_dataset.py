"""
CrunchIQ - Synthetic dataset generator (FOR PIPELINE TESTING ONLY).

*** THIS DATA IS FAKE. IT IS NOT REAL BISCUIT AUDIO. ***
Use it to sanity-check that record -> build_dataset -> train_classifier ->
rule_based -> live_classify all wire together correctly, BEFORE you have a
mic and real samples. Replace with real recordings (record_samples.py) as
early as possible on day 1 - do NOT use results trained on this synthetic
set as evidence in your demo or slides.

Model per class (simple envelope + tone + noise synthesis, not a real
acoustic model):
  - fresh     : fast attack, bright tone (high fundamental + harmonics),
                light noise, short decay
  - stale     : slow attack, dull tone (low fundamental, few harmonics),
                minimal noise, longer/softer decay
  - overbaked : very fast attack, bright tone + heavy crackly noise,
                short choppy decay (simulates brittle shattering)
  - broken    : two mismatched, unevenly-timed cracks (structurally
                uneven snap) with irregular energy between them

Usage:
    python generate_synthetic_dataset.py                 # 12/class, all 4 classes
    python generate_synthetic_dataset.py --count 10
    python generate_synthetic_dataset.py --classes fresh stale overbaked
"""
import argparse
import os
import wave
import numpy as np

SAMPLE_RATE = 22050
CLIP_SECONDS = 2.5
DATA_DIR = "data"

RNG = np.random.default_rng(42)


def _envelope(t, attack_s, decay_tau, start_s=0.0):
    """Fast-attack, exponential-decay amplitude envelope starting at start_s."""
    env = np.zeros_like(t)
    mask = t >= start_s
    tt = t[mask] - start_s
    attack_samples = max(int(attack_s * SAMPLE_RATE), 1)
    rise = np.clip(tt * SAMPLE_RATE / attack_samples, 0, 1)
    decay = np.exp(-tt / decay_tau)
    env[mask] = rise * decay
    return env


def _tone(t, freqs, amps, jitter=0.02):
    """Sum of a few (jittered) sine harmonics."""
    sig = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        f_j = f * (1 + RNG.uniform(-jitter, jitter))
        phase = RNG.uniform(0, 2 * np.pi)
        sig += a * np.sin(2 * np.pi * f_j * t + phase)
    return sig


def _bandlimited_noise(n, low_hz, high_hz, sr=SAMPLE_RATE):
    """White noise band-passed via FFT masking - used for the 'crackle'."""
    white = RNG.normal(0, 1, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[~mask] = 0
    filtered = np.fft.irfft(spectrum, n=n)
    peak = np.max(np.abs(filtered)) + 1e-9
    return filtered / peak


def gen_fresh(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    n = int(duration * sr)
    t = np.arange(n) / sr
    start = RNG.uniform(0.3, 0.5)
    env = _envelope(t, attack_s=0.003, decay_tau=0.06, start_s=start)
    tone = _tone(t, freqs=[2800, 3600, 5200], amps=[0.6, 0.35, 0.15])
    noise = _bandlimited_noise(n, 2000, 6000) * 0.25
    sig = env * (tone + noise)
    return _finalize(sig)


def gen_stale(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    n = int(duration * sr)
    t = np.arange(n) / sr
    start = RNG.uniform(0.3, 0.5)
    env = _envelope(t, attack_s=0.03, decay_tau=0.18, start_s=start)
    tone = _tone(t, freqs=[500, 750, 1050], amps=[0.7, 0.3, 0.1])
    noise = _bandlimited_noise(n, 200, 900) * 0.08
    sig = env * (tone + noise)
    return _finalize(sig)


def gen_overbaked(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    n = int(duration * sr)
    t = np.arange(n) / sr
    start = RNG.uniform(0.3, 0.5)
    env = _envelope(t, attack_s=0.002, decay_tau=0.045, start_s=start)
    tone = _tone(t, freqs=[3200, 4400, 6000], amps=[0.5, 0.4, 0.25])
    noise = _bandlimited_noise(n, 3000, 8500) * 0.55
    # a few extra micro-crack impulses layered on top for "brittle shatter"
    sig = env * (tone + noise)
    for _ in range(RNG.integers(2, 5)):
        click_start = start + RNG.uniform(0.0, 0.08)
        click_env = _envelope(t, attack_s=0.001, decay_tau=0.01, start_s=click_start)
        click_noise = _bandlimited_noise(n, 3500, 9000)
        sig += 0.3 * click_env * click_noise
    return _finalize(sig)


def gen_broken(duration=CLIP_SECONDS, sr=SAMPLE_RATE):
    """Two mismatched, unevenly-timed cracks - structurally uneven snap."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    start1 = RNG.uniform(0.25, 0.4)
    gap = RNG.uniform(0.08, 0.22)
    start2 = start1 + gap

    env1 = _envelope(t, attack_s=0.004, decay_tau=0.05, start_s=start1)
    tone1 = _tone(t, freqs=[2600, 3400], amps=[0.6, 0.3])
    noise1 = _bandlimited_noise(n, 2000, 5500) * 0.3

    env2 = _envelope(t, attack_s=0.006, decay_tau=0.07, start_s=start2)
    tone2 = _tone(t, freqs=[1400, 2000], amps=[0.5, 0.25])
    noise2 = _bandlimited_noise(n, 1200, 4000) * 0.35

    sig = env1 * (tone1 + noise1) * 1.0 + env2 * (tone2 + noise2) * 0.6
    return _finalize(sig)


def _finalize(sig, noise_floor=0.003):
    sig = sig + RNG.normal(0, noise_floor, size=sig.shape)  # ambient mic noise
    peak = np.max(np.abs(sig)) + 1e-9
    sig = sig / peak * RNG.uniform(0.75, 0.95)  # normalize with slight level jitter
    return sig.astype(np.float32)


GENERATORS = {
    "fresh": gen_fresh,
    "stale": gen_stale,
    "overbaked": gen_overbaked,
    "broken": gen_broken,
}


def write_wav(path, audio, sr=SAMPLE_RATE):
    audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12, help="Samples per class")
    parser.add_argument("--classes", nargs="+", default=list(GENERATORS.keys()),
                         choices=list(GENERATORS.keys()))
    args = parser.parse_args()

    print("=== Generating SYNTHETIC dataset (for pipeline testing only) ===\n")
    for cls in args.classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        gen_fn = GENERATORS[cls]
        for i in range(1, args.count + 1):
            audio = gen_fn()
            fname = f"{cls}_{i:02d}.wav"
            fpath = os.path.join(cls_dir, fname)
            write_wav(fpath, audio)
        print(f"  {cls:<12} -> {args.count} synthetic clips in {cls_dir}/")

    print("\nDone. Run build_dataset.py next.")
    print("REMINDER: replace this with real recordings before your actual demo.")


if __name__ == "__main__":
    main()
