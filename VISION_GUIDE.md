# CrunchVision — Vision + Fusion Guide

This extends the acoustic-only CrunchIQ project with a vision half (cross-
section photo classification via transfer learning) and a fusion layer
combining both signals into one verdict. Read this alongside the main
`README.md`.

---

## 1. New files

| File | Purpose |
|---|---|
| `vision_dataset.py` | PyTorch `Dataset` loading `data_vision/<class>/*.jpg`, plus train/val transforms |
| `train_vision.py` | Transfer-learning training script: frozen MobileNetV2/ResNet18 backbone + new head |
| `vision_infer.py` | `predict_image()` — one photo in, (class, confidence, all_probs) out |
| `fusion.py` | Explainable agreement/disagreement fusion logic (no third model) |
| `capture_pair.py` | Paired data collection: one break -> one `.wav` + one `.jpg`, matching indices |
| `live_demo.py` | Combined live demo: snap -> audio + photo -> both models -> fused verdict |

Folder convention: `data_vision/<class_name>/*.jpg` — **use the exact same
class names as `data/<class_name>/`** (e.g. `fresh`, `stale`, `overbaked`,
optionally `broken`). Fusion compares label strings directly, so a mismatch
(`"fresh"` vs `"fresh_crisp"`) would silently break agreement detection.
I already fixed this: `rule_based.py`'s labels now match the folder names.

## 2. Workflow

```bash
pip install -r requirements.txt --break-system-packages

# Collect paired samples (one break -> audio + photo)
python capture_pair.py --class fresh     --count 15
python capture_pair.py --class stale     --count 15
python capture_pair.py --class overbaked --count 15

# Train both halves independently
python build_dataset.py && python train_classifier.py     # acoustic
python train_vision.py --epochs 15 --backbone mobilenet_v2 # vision

# Test fusion + full pipeline
python live_demo.py
```

If you already have separate acoustic samples from the earlier CrunchIQ
build, you don't need to re-record — just run `capture_pair.py` for the
photo half only, or manually add `.jpg` files under `data_vision/` with
your own naming; the vision model doesn't require the same files to exist
under `data/`, they're trained independently either way.

---

## 3. Why transfer learning is the right call (for judges)

Three sentences you can say confidently if asked:

1. *"With 40-60 images per class, training a CNN from scratch would need
   orders of magnitude more data to learn general visual features like
   edges and textures on its own — it would just memorize our training
   photos."*
2. *"MobileNetV2 already learned those general features from millions of
   ImageNet images. We freeze that backbone and only train a small new
   classification head — a few thousand parameters, not millions — so it
   fits our data volume and trains in minutes on a CPU."*
3. *"The crumb/aeration texture in a cross-section photo is exactly the
   kind of local texture and gradient pattern those pretrained features
   already respond well to, even though the backbone never saw a biscuit."*

## 4. Fusion — how to explain it in one breath

*"Neither model overrides the other. If acoustic and vision agree, we
report that class with combined confidence. If they disagree, we don't
force a fake consensus — we show both predictions, flag it for human
review, and surface the higher-confidence one as the primary read. It's
a simple rule, not another trained model, so there's nothing hidden
between the two classifiers and the final verdict."*

This is also the honest engineering reason you gave for not training a
third fusion model: with this data volume, a learned combiner is itself
an overfitting risk for marginal benefit over a transparent rule.

---

## 5. Scripting the demo around an agreement case

Your instinct to script around agreement is right — it's the cleanest,
most confident moment in the video. Suggested beats:

1. **Before recording:** run a few practice captures with `live_demo.py`
   and pick a biscuit/class combo that reliably agrees (fresh is usually
   the easiest — bright/sharp acoustically and clearly aerated visually).
2. **On camera:** snap the pre-selected biscuit, let both predictions
   print, and narrate the agreement moment: *"Both signals independently
   say fresh — [X]% from the sound, [Y]% from the crumb structure. That
   agreement is what gives us confidence in the read."*
3. **Optional second beat, if you have time:** show one disagreement case
   from a pre-recorded run (not live) to demonstrate the flagging logic
   without gambling live-demo time on a coin-flip result. Frame it as a
   feature: *"When they disagree, the system doesn't guess — it flags for
   a human to check, which is exactly the behavior you'd want on a line."*

---

## 6. Live demo vs. pre-recorded fallback — how to decide

Be honest with yourself about this the night before, not during setup:

- **Vision val accuracy comfortably above chance level** (chance = 1/n_classes,
  so >50% for 3 classes, >60-65% would feel solid) **AND** you've run
  `live_demo.py` successfully 5+ times in a row with your actual camera
  setup → live demo is reasonable risk.
- **Val accuracy is close to chance, or `train_vision.py` printed the
  "few validation images" warning and results swing a lot between runs**
  → don't gamble the whole video on a live long-shot. Pre-record 2-3 clean
  demo runs (one agreement case, ideally one disagreement case) the night
  before, and play the best one in your video with a quick "here's a live
  capture we ran earlier" framing. Judges care about the system working,
  not about real-time risk theater.
- **Middle ground:** do the audio half live (it's the more validated,
  proven half) and cut to a pre-recorded vision + fusion result. This is
  a legitimate hybrid — just say so explicitly rather than implying both
  ran live.

Whatever you choose, your one scoping sentence at the top of the video
(proof-of-concept, small paired dataset, acoustic validated, vision via
transfer learning on limited samples) makes either choice consistent —
you're not overclaiming either way.

## 7. What to say if vision accuracy is mediocre

Don't hide a weak number — reframe it honestly:

> *"On [N] validation images per class, our vision head reached [X]%
> accuracy. That's above chance but not production-grade — with 40-60
> training images per class that's expected, and it's exactly the kind
> of early signal that justifies scaling up data collection for a real
> pilot. The value right now isn't the accuracy number, it's proving that
> a frozen-backbone transfer-learning approach can pick up *any* signal
> from crumb structure at all on this little data, and that fusing it
> with the already-validated acoustic signal gives you a system that
' fails safely' — it flags disagreement instead of guessing."*

Never let a printed number stand alone without that context — it's why
`train_vision.py` always prints the small-sample caveat directly next to
the accuracy, not as a separate afterthought a judge might miss.

---

## 8. Updated 2-minute video structure (with vision + fusion)

| Time | Segment |
|---|---|
| 0:00–0:15 | Scoping sentence + problem: one break, two independent signals, zero extra product cost |
| 0:15–0:30 | Quick visual: acoustic spectrogram comparison + a couple of cross-section photos side by side |
| 0:30–0:35 | One-line fusion explanation (see Section 4) |
| 0:35–1:45 | Live (or pre-recorded, per Section 6) capture: snap -> audio + photo -> both predictions print -> fused verdict, narrated on the agreement case |
| 1:45–1:55 | Manufacturing value: one destructive test, two QC dimensions, flags disagreement instead of guessing |
| 1:55–2:00 | Honest close: proof-of-concept on a small paired dataset, clear path to a larger pilot |

## 9. Known limitations to have ready if asked

- Vision and acoustic models are trained fully independently — there's no
  guarantee the two datasets came from perfectly matched biscuits beyond
  what `capture_pair.py`'s paired-capture protocol gives you.
- The frozen backbone means the model can't learn biscuit-specific low-
  level features beyond what ImageNet already encodes — a larger dataset
  would eventually benefit from partially unfreezing the last few backbone
  layers (fine-tuning), which we deliberately skipped given the time/data
  budget.
- Cross-section photo quality (lighting, focus, framing consistency) is
  not controlled for beyond "do it the same way each time" — a real
  deployment would want a fixed light box, same as the acoustic protocol
  needs a controlled recording setup.
