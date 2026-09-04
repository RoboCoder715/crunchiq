"""
CrunchIQ - Acoustic feature extraction for biscuit snap/break sounds.

Extracts a compact, judge-explainable feature vector from a WAV clip:
- MFCCs (mean + std of first N coefficients) -> timbral fingerprint
- Spectral centroid (mean/std)               -> brightness of the sound
- Zero-crossing rate (mean/std)               -> "crackliness" / noisiness
- Spectral rolloff (mean)                     -> where most energy sits
- RMS energy stats + attack time              -> loudness envelope / snap sharpness
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 13

FEATURE_NAMES = (
    [f"mfcc{i+1}_mean" for i in range(N_MFCC)]
    + [f"mfcc{i+1}_std" for i in range(N_MFCC)]
    + [
        "centroid_mean",
        "centroid_std",
        "zcr_mean",
        "zcr_std",
        "rolloff_mean",
        "rms_mean",
        "rms_std",
        "rms_peak",
        "attack_time",
    ]
)


def load_audio(path, sr=SAMPLE_RATE):
    """Load a wav file, mono, and trim leading/trailing silence around the snap."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y, sr


def extract_features(path, sr=SAMPLE_RATE):
    """Full feature vector used by the ML classifier. Returns a 1D numpy array."""
    y, sr = load_audio(path, sr=sr)
    if len(y) < sr * 0.05:
        raise ValueError(f"Audio too short/silent after trimming: {path}")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]

    attack_time = _attack_time(y, sr)

    feats = np.concatenate([
        mfcc_mean, mfcc_std,
        [centroid.mean(), centroid.std()],
        [zcr.mean(), zcr.std()],
        [rolloff.mean()],
        [rms.mean(), rms.std(), rms.max()],
        [attack_time],
    ])
    return feats


def _attack_time(y, sr):
    """Time (seconds) from onset to peak amplitude - a proxy for 'snap sharpness'.
    A fresh, crisp snap has a very short, sharp attack; a soggy break is duller/slower.
    """
    if len(y) == 0:
        return 0.0
    peak_idx = int(np.argmax(np.abs(y)))
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
    onset_idx = int(onset_samples[0]) if len(onset_samples) > 0 else 0
    if peak_idx <= onset_idx:
        return 0.0
    return (peak_idx - onset_idx) / sr


def extract_key_features(path, sr=SAMPLE_RATE):
    """Just the two strongest, most explainable features, for the rule-based fallback:
    spectral centroid (brightness) and zero-crossing rate (crackliness).
    """
    y, sr = load_audio(path, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    return float(centroid), float(zcr)
