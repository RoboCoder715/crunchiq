# CrunchIQ — Acoustic QC for Biscuit Texture

Classifies biscuit quality (fresh/crisp vs. stale/soggy vs. overbaked/brittle)
purely from the **sound** of the snap — no camera, targeting a QC dimension
(internal moisture/texture) that vision-based inspection can't see.

Built for Britannia Creatovate 2.0 (biscuit/FMCG manufacturing track).

---

## 1. Setup (do this first)

```bash
pip install -r requirements.txt --break-system-packages
```

If `sounddevice` can't find your mic, run `python -c "import sounddevice as sd; print(sd.query_devices())"`
to check available input devices, and set `sd.default.device` at the top of
`record_samples.py` / `live_classify.py` if needed.

## 2. Full pipeline (day 1)

```bash
# 1. Record 8-10 labeled samples per class
python record_samples.py --class fresh     --count 10
python record_samples.py --class stale     --count 10
python record_samples.py --class overbaked --count 10
# optional 4th class:
python record_samples.py --class broken    --count 10

# 2. Build the feature dataset from data/
python build_dataset.py

# 3. Train + evaluate the RandomForest classifier
python train_classifier.py

# 4. Calibrate the rule-based fallback against your own recordings
python rule_based.py --calibrate
# -> edit CENTROID_LOW / CENTROID_HIGH / ZCR_HIGH in rule_based.py based on the printed stats

# 5. Generate comparison plots for your slides/video
python visualize.py
```

## 3. Demo (day 2 / recording day)

```bash
python live_classify.py
```
Prints both the rule-based prediction (guaranteed to work, no training needed)
and the ML model's prediction side-by-side, so you can show both onscreen.

**Recording protocol (keep it consistent across all samples):** mic ~10-15cm
from the biscuit, same snap motion each time, quiet room, 2-3 second clips.

---

## 4. Why each feature separates the classes (narration cheat-sheet)

Use this to sound confident when a judge asks "why does this work?":

- **Spectral centroid (brightness).** A crisp snap has more high-frequency
  content — think "brighter" timbre, like a sharp crack. A soggy break
  sounds duller and lower-pitched because moisture damps the higher
  frequencies. This is our single strongest, most explainable feature.

- **Zero-crossing rate (ZCR).** How often the waveform crosses zero per
  second — a proxy for noisiness/texture. Brittle, overbaked biscuits
  shatter unevenly, producing a "crackly," high-ZCR signal. Soggy
  biscuits bend and tear more than they crack, giving a smoother,
  lower-ZCR waveform.

- **Spectral rolloff.** The frequency below which most spectral energy
  sits. Correlates with centroid but is more robust to outlier
  high-frequency noise — a second opinion on "brightness."

- **RMS energy / attack time.** How loud the sound is and how fast it
  reaches peak loudness. A fresh snap has a sharp, near-instant attack
  (loud immediately). A soggy break has a slower onset — the biscuit
  bends before it gives way, so energy builds more gradually.

- **MFCCs.** The overall timbral "fingerprint" of the sound — captures
  texture nuances the four features above don't fully summarize on their
  own. This is what gives the ML model its edge over the 2-feature rule
  -based fallback, at the cost of needing more data and being harder to
  explain in one sentence.

**One-line pitch:** *"A crisp biscuit sounds bright and sharp; a soggy one
sounds dull and soft; an overbaked one sounds bright but crackly. Those
are exactly the acoustic properties centroid, ZCR, and attack time measure."*

---

## 5. Onset detection — inline/production framing

The MVP demo uses manual timing (you press Enter, then snap). For a real
production line, `librosa.onset.onset_detect()` — already used internally
in `features.py` for attack-time — would run continuously on a streaming
mic buffer to auto-trigger a classification the moment a snap/break sound
occurs, with no manual triggering needed. That's the natural next step:
a continuously-listening inline sensor rather than a spot-check tool.

Mention this even though your MVP uses manual triggering — it shows you've
thought about the production path, not just the demo.

---

## 6. Manufacturing value — talking points for the close

- **Continuous, non-destructive QC.** Today's checks (manual taste test,
  destructive lab moisture analysis) are slow and sampled — they only
  catch a problem at the moment someone happens to test. An acoustic
  sensor could run inline, on every unit or a much denser sample, without
  destroying product.

- **Catches drift between lab-sampling intervals.** Moisture/texture can
  drift over a shift due to line speed, oven temperature variance, or
  packaging humidity exposure. A continuous acoustic signal could flag
  that drift in near real-time, instead of waiting for the next scheduled
  lab sample.

- **Handheld spot-check tool for supervisors.** Even before a full inline
  deployment, this could ship as a simple handheld/tablet app: a
  supervisor snaps a biscuit off the line and gets an instant
  fresh/stale/overbaked read, cutting reliance on subjective taste
  testing.

- **Complements, not replaces, vision QC.** Vision systems catch surface
  defects (shape, color, cracks visible externally); this catches an
  *internal* property — moisture/texture — that looks the same to a
  camera but sounds different. It's an additional QC dimension, not a
  competitor to existing systems.

- **Honest scope, for credibility with judges.** With ~10 samples/class
  this is a proof-of-concept, not a validated production model — say so.
  The real claim is: *the acoustic signal clearly correlates with texture
  quality, which is the evidence needed to justify a larger data
  collection effort and a pilot on an actual line.*

---

## 7. 2-minute demo video structure (suggested)

| Time | Segment |
|---|---|
| 0:00–0:20 | Problem: moisture/texture QC today is manual, slow, destructive, sampled |
| 0:20–0:35 | Show `plots/class_comparison.png` — waveform + spectrogram side by side, 3 classes. Let it speak before you explain. |
| 0:35–0:45 | One-line explanation: bright/sharp = fresh, dull/soft = soggy, bright/crackly = overbaked |
| 0:45–1:20 | Live demo: run `live_classify.py`, snap 2-3 biscuits on camera, show rule-based (+ ML) prediction printing live |
| 1:20–1:45 | Manufacturing value: continuous non-destructive QC, drift detection, handheld spot-check tool |
| 1:45–2:00 | Close: honest scope (~10 samples/class = proof-of-concept), path to production via onset-triggered inline sensor |

**Tip:** rehearse the live-classify segment a few times before recording —
the timing is the only part with any real-world variance (mic pickup,
snap loudness), everything else is deterministic.

---

## 8. File overview

| File | Purpose |
|---|---|
| `features.py` | Core feature extraction (MFCCs, centroid, ZCR, rolloff, RMS, attack time) |
| `record_samples.py` | Prompt-and-record loop to build `data/<class>/*.wav` |
| `build_dataset.py` | Walks `data/`, extracts features, saves `dataset.npz` |
| `train_classifier.py` | Trains RandomForest, evaluates honestly given small-sample size, saves `model.joblib` |
| `rule_based.py` | 2-feature threshold fallback classifier (guaranteed-to-demo backup) |
| `live_classify.py` | Records a live snap, prints both rule-based and ML predictions |
| `visualize.py` | Waveform + spectrogram comparison plots for slides/video |
| `requirements.txt` | Python dependencies |

## 9. Honest limitations (know these before a judge asks)

- ~10 samples/class is far too small to claim a validated accuracy number —
  frame `train_classifier.py`'s output as a directional signal only.
- Recording protocol variance (mic distance, snap force, ambient noise)
  is not controlled for beyond "do it consistently" — a real deployment
  would need a fixed acoustic enclosure or a normalization step.
- The rule-based fallback's thresholds are calibrated to *your* mic and
  room — they will need recalibration on different hardware.
- This is a proof-of-concept for a *research signal* (sound correlates
  with texture), not a production-ready classifier.
