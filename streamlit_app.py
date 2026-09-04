"""
CrunchIQ — Acoustic QC for Biscuit Texture (Streamlit Web App)

A jury-ready web interface for the CrunchIQ biscuit quality classifier.
Records or uploads a biscuit snap sound, extracts acoustic features,
and classifies texture quality using both a rule-based fallback and a
trained RandomForest model.

Built for Britannia Creatovate 2.0.
"""
import io
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st

# ── Must come before any other st calls ──────────────────────────────
st.set_page_config(
    page_title="CrunchIQ — Acoustic QC for Biscuit Texture",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Lazy-loaded heavy imports (cached) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_librosa():
    import librosa
    import librosa.display
    return librosa

@st.cache_resource(show_spinner=False)
def _load_model():
    """Load the trained RandomForest model bundle, if available."""
    MODEL_FILE = "model.joblib"
    if not os.path.exists(MODEL_FILE):
        return None
    import joblib
    return joblib.load(MODEL_FILE)


# ── Feature extraction (mirrors features.py) ────────────────────────
SAMPLE_RATE = 22050
N_MFCC = 13

def load_audio(audio_bytes):
    """Load audio from bytes, return (y, sr) mono at SAMPLE_RATE."""
    librosa = _load_librosa()
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)
    try:
        data, sr = sf.read(buf)
    except Exception:
        buf.seek(0)
        data, sr = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
        return data, sr

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    data, _ = librosa.effects.trim(data.astype(np.float32), top_db=30)
    return data.astype(np.float32), SAMPLE_RATE


def extract_all_features(y, sr):
    """Extract the full feature vector (matches features.py)."""
    librosa = _load_librosa()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]

    # Attack time
    peak_idx = int(np.argmax(np.abs(y)))
    onset_samples = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
    onset_idx = int(onset_samples[0]) if len(onset_samples) > 0 else 0
    attack_time = max(0.0, (peak_idx - onset_idx) / sr) if peak_idx > onset_idx else 0.0

    feats = np.concatenate([
        mfcc_mean, mfcc_std,
        [centroid.mean(), centroid.std()],
        [zcr.mean(), zcr.std()],
        [rolloff.mean()],
        [rms.mean(), rms.std(), rms.max()],
        [attack_time],
    ])

    key_features = {
        "Spectral Centroid (Hz)": float(centroid.mean()),
        "Zero-Crossing Rate": float(zcr.mean()),
        "Spectral Rolloff (Hz)": float(rolloff.mean()),
        "RMS Energy": float(rms.mean()),
        "Attack Time (s)": float(attack_time),
    }
    return feats, key_features


# ── Classifiers ──────────────────────────────────────────────────────
CENTROID_LOW = 1800
CENTROID_HIGH = 3200
ZCR_HIGH = 0.12

def classify_rule_based(centroid, zcr):
    """Port of rule_based.py — threshold logic."""
    if centroid < CENTROID_LOW:
        return "stale", 0.70
    elif centroid >= CENTROID_HIGH and zcr >= ZCR_HIGH:
        return "overbaked", 0.70
    elif centroid >= CENTROID_LOW:
        return "fresh", 0.70
    return "uncertain", 0.30


def classify_ml(feats, model_bundle):
    """Run the trained RandomForest model."""
    feats_scaled = model_bundle["scaler"].transform(feats.reshape(1, -1))
    proba = model_bundle["model"].predict_proba(feats_scaled)[0]
    classes = model_bundle["model"].classes_
    pred_idx = int(np.argmax(proba))
    return classes[pred_idx], float(proba[pred_idx]), dict(zip(classes, proba.tolist()))


# ── Visualization helpers ────────────────────────────────────────────
CLASS_COLORS = {
    "fresh": "#10b981",
    "stale": "#f59e0b",
    "overbaked": "#ef4444",
    "broken": "#8b5cf6",
    "uncertain": "#6b7280",
}

def plot_waveform(y, sr):
    librosa = _load_librosa()
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#38bdf8", alpha=0.85)
    ax.set_title("Waveform", color="#e2e8f0", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Amplitude", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    plt.tight_layout()
    return fig


def plot_spectrogram(y, sr):
    librosa = _load_librosa()
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma")
    ax.set_title("Mel Spectrogram", color="#e2e8f0", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Hz", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cb.ax.yaxis.set_tick_params(color="#64748b")
    cb.outline.set_edgecolor("#1e293b")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#94a3b8", fontsize=8)
    plt.tight_layout()
    return fig


def plot_probabilities(proba_dict):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")
    classes = list(proba_dict.keys())
    probs = list(proba_dict.values())
    colors = [CLASS_COLORS.get(c, "#6b7280") for c in classes]
    bars = ax.barh(classes, probs, color=colors, height=0.55, edgecolor="#1e293b", linewidth=0.5)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{prob:.0%}", va="center", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_title("Class Probabilities (ML Model)", color="#e2e8f0", fontsize=13,
                 fontweight="bold", pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=11)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


# ── Custom CSS ───────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
}

/* Hero section */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem 1rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #10b981 0%, #38bdf8 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 0.8rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(139, 92, 246, 0.15));
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.8rem;
    color: #10b981;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Verdict cards */
.verdict-card {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease;
}
.verdict-card:hover {
    transform: translateY(-2px);
}
.verdict-label {
    font-size: 2rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
.verdict-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
}
.verdict-source {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* Feature grid */
.feature-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.feature-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #38bdf8;
}
.feature-name {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Info cards */
.info-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(30, 41, 59, 0.3));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.info-card h4 {
    margin: 0 0 0.4rem 0;
    font-weight: 700;
}
.info-card p {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0;
}

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.3), transparent);
    margin: 2rem 0;
    border: none;
}

/* Audio input styling */
.stAudioInput > div {
    border-radius: 12px !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


# ── Main App ─────────────────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🍪 CrunchIQ</div>
        <div class="hero-subtitle">Acoustic Quality Control for Biscuit Texture</div>
        <div class="hero-badge">BRITANNIA CREATOVATE 2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Load model ──
    model_bundle = _load_model()

    # ── Audio Input ──
    st.markdown("### 🎙️ Record or Upload a Biscuit Snap")
    st.caption("Snap a biscuit near your microphone, or upload a pre-recorded WAV file.")

    col_rec, col_upload = st.columns(2)

    audio_bytes = None

    with col_rec:
        recorded = st.audio_input("Record a snap", key="mic_input")
        if recorded is not None:
            audio_bytes = recorded.getvalue()

    with col_upload:
        uploaded = st.file_uploader("Upload WAV file", type=["wav", "mp3", "ogg", "flac"],
                                     key="file_input")
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()

    if audio_bytes is None:
        # ── Show placeholder content when no audio ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔬 How It Works")
        st.markdown("Snap a biscuit near the mic and CrunchIQ instantly analyzes the acoustic signature "
                     "to determine texture quality. Here's what each feature measures:")

        feat_cols = st.columns(4)
        features_info = [
            ("🔊", "Spectral Centroid", "Brightness of the sound. A crisp snap is bright and sharp; a soggy break sounds dull and low-pitched."),
            ("〰️", "Zero-Crossing Rate", "How 'crackly' the waveform is. Overbaked biscuits shatter noisily; soggy ones tear smoothly."),
            ("📊", "Spectral Rolloff", "Where most spectral energy sits. Correlates with centroid but more robust to noise outliers."),
            ("⚡", "Attack Time", "How fast the sound reaches peak loudness. A crisp snap is near-instant; a soggy break builds gradually."),
        ]
        for col, (icon, name, desc) in zip(feat_cols, features_info):
            with col:
                st.markdown(f"""
                <div class="info-card">
                    <h4>{icon} {name}</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Class Comparison Plot ──
        comparison_path = os.path.join("plots", "class_comparison.png")
        if os.path.exists(comparison_path):
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📈 Acoustic Signatures by Class")
            st.caption("Waveform and mel spectrogram comparison across biscuit quality classes (from training data).")
            st.image(comparison_path, use_container_width=True)

        # ── About ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("ℹ️ About CrunchIQ", expanded=False):
            st.markdown("""
            **CrunchIQ** classifies biscuit quality (fresh/crisp vs. stale/soggy vs. overbaked/brittle)
            purely from the **sound** of the snap — no camera needed. It targets a QC dimension
            (internal moisture/texture) that vision-based inspection can't see.

            **Dual classifier system:**
            - **Rule-based**: 2-feature threshold logic (spectral centroid + ZCR). Guaranteed to work, no training needed.
            - **ML model**: RandomForest trained on 35 acoustic features (MFCCs, centroid, ZCR, rolloff, RMS, attack time).

            **Honest scope:** With ~10 samples/class this is a proof-of-concept, not a validated production
            model. The claim is: *the acoustic signal clearly correlates with texture quality*, which is the
            evidence needed to justify a larger data collection effort and a pilot on an actual line.
            """)
        return

    # ══════════════════════════════════════════════════════════════════
    # ── ANALYSIS (audio provided) ────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    with st.spinner("🔬 Analyzing acoustic signature..."):
        y, sr = load_audio(audio_bytes)
        feats, key_features = extract_all_features(y, sr)
        centroid = key_features["Spectral Centroid (Hz)"]
        zcr = key_features["Zero-Crossing Rate"]

        rb_label, rb_conf = classify_rule_based(centroid, zcr)

        ml_label, ml_conf, ml_proba = None, None, None
        if model_bundle is not None:
            ml_label, ml_conf, ml_proba = classify_ml(feats, model_bundle)

    # ── Verdict ──
    st.markdown("### 🎯 Classification Verdict")

    if model_bundle is not None:
        col_rb, col_ml = st.columns(2)
    else:
        col_rb = st.columns(1)[0]

    with col_rb:
        color = CLASS_COLORS.get(rb_label, "#6b7280")
        st.markdown(f"""
        <div class="verdict-card" style="background: linear-gradient(135deg, {color}15, {color}08);">
            <div class="verdict-label" style="color: {color};">{rb_label}</div>
            <div class="verdict-confidence" style="color: {color};">{rb_conf:.0%} confidence</div>
            <div class="verdict-source">Rule-Based Classifier</div>
        </div>
        """, unsafe_allow_html=True)

    if model_bundle is not None and ml_label is not None:
        with col_ml:
            color = CLASS_COLORS.get(ml_label, "#6b7280")
            st.markdown(f"""
            <div class="verdict-card" style="background: linear-gradient(135deg, {color}15, {color}08);">
                <div class="verdict-label" style="color: {color};">{ml_label}</div>
                <div class="verdict-confidence" style="color: {color};">{ml_conf:.0%} confidence</div>
                <div class="verdict-source">ML Model (RandomForest)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── Agreement / Disagreement ──
    if model_bundle is not None and ml_label is not None:
        if rb_label == ml_label:
            st.success(f"✅ **Both classifiers agree: {rb_label.upper()}** — high confidence in this read.")
        else:
            st.warning(f"⚠️ **Classifiers disagree** — Rule-based says *{rb_label}*, ML says *{ml_label}*. "
                       f"Flagged for human review.")

    # ── Extracted Features ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Extracted Acoustic Features")

    feat_cols = st.columns(5)
    for col, (name, value) in zip(feat_cols, key_features.items()):
        with col:
            if "Hz" in name:
                display_val = f"{value:,.0f}"
            elif "Time" in name:
                display_val = f"{value:.4f}"
            else:
                display_val = f"{value:.4f}"
            short_name = name.replace(" (Hz)", "").replace(" (s)", "")
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-value">{display_val}</div>
                <div class="feature-name">{short_name}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Visualizations ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🎵 Audio Analysis")

    col_wave, col_spec = st.columns(2)
    with col_wave:
        fig_wave = plot_waveform(y, sr)
        st.pyplot(fig_wave, use_container_width=True)
        plt.close(fig_wave)

    with col_spec:
        fig_spec = plot_spectrogram(y, sr)
        st.pyplot(fig_spec, use_container_width=True)
        plt.close(fig_spec)

    # ── ML Probabilities ──
    if ml_proba is not None:
        col_proba, col_explain = st.columns([1, 1])
        with col_proba:
            fig_proba = plot_probabilities(ml_proba)
            st.pyplot(fig_proba, use_container_width=True)
            plt.close(fig_proba)
        with col_explain:
            st.markdown("""
            <div class="info-card">
                <h4>🧠 How the ML Model Works</h4>
                <p>A <strong>RandomForest</strong> classifier trained on 35 acoustic features
                (13 MFCC means + 13 MFCC stds + centroid, ZCR, rolloff, RMS, attack time).
                Evaluated honestly given the small sample size (~10/class) — this is
                proof-of-concept evidence, not a production metric.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>📐 Rule-Based Fallback</h4>
                <p>Uses just <strong>2 features</strong> (spectral centroid + ZCR) with hardcoded
                thresholds. Guaranteed to work with no training. If centroid &lt; 1800 Hz → stale.
                If centroid ≥ 3200 Hz and ZCR ≥ 0.12 → overbaked. Otherwise → fresh.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Pitch ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    > *"A crisp biscuit sounds bright and sharp; a soggy one sounds dull and soft;
    > an overbaked one sounds bright but crackly. Those are exactly the acoustic
    > properties spectral centroid, ZCR, and attack time measure."*
    """)

    # ── Class Comparison Plot ──
    comparison_path = os.path.join("plots", "class_comparison.png")
    if os.path.exists(comparison_path):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("📈 Training Data — Class Comparison", expanded=False):
            st.image(comparison_path, use_container_width=True)

    # ── About ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("ℹ️ About CrunchIQ", expanded=False):
        st.markdown("""
        **CrunchIQ** classifies biscuit quality (fresh/crisp vs. stale/soggy vs. overbaked/brittle)
        purely from the **sound** of the snap — no camera needed. It targets a QC dimension
        (internal moisture/texture) that vision-based inspection can't see.

        **Manufacturing value:**
        - 🏭 Continuous, non-destructive QC — no product destroyed
        - ⏱️ Catches moisture/texture drift between lab-sampling intervals
        - 📱 Path to handheld spot-check tool for supervisors
        - 👁️ Complements vision QC — catches what cameras can't see

        **Honest scope:** With ~10 samples/class this is a proof-of-concept, not a validated
        production model. The claim is: *the acoustic signal clearly correlates with texture
        quality*, which justifies a larger data collection effort and a pilot on an actual line.

        Built for **Britannia Creatovate 2.0** 🍪
        """)


if __name__ == "__main__":
    main()
