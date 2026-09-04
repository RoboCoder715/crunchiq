"""
CrunchVision - Fusion logic combining acoustic + vision predictions.

Deliberately NOT a trained model - a simple, explainable rule, so there's
no black-box-on-black-box reasoning to defend in front of judges:

  - Both signals agree    -> report that class; combined confidence is the
                              average of the two model confidences.
  - Signals disagree      -> report BOTH predictions transparently, flag
                              the result for review, and surface the
                              higher-confidence prediction as the primary
                              label (clearly marked as coming from only
                              one of the two signals).

Usage (as a library):
    from fusion import fuse_predictions
    result = fuse_predictions("fresh", 0.82, "fresh", 0.75)
    print(result.summary())
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FusionResult:
    agreement: bool
    primary_label: str
    primary_confidence: float
    acoustic_label: str
    acoustic_confidence: float
    vision_label: str
    vision_confidence: float
    flagged_for_review: bool
    combined_confidence: Optional[float] = None

    def summary(self):
        if self.agreement:
            return (
                f"VERDICT: {self.primary_label}  "
                f"(both signals agree -> combined confidence {self.combined_confidence:.0%})"
            )
        return (
            f"VERDICT: {self.primary_label} [primary - higher confidence signal]  "
            f"-- FLAGGED FOR REVIEW: signals disagree "
            f"(acoustic={self.acoustic_label} {self.acoustic_confidence:.0%}, "
            f"vision={self.vision_label} {self.vision_confidence:.0%})"
        )


def fuse_predictions(acoustic_label, acoustic_confidence, vision_label, vision_confidence):
    agreement = (acoustic_label == vision_label)

    if agreement:
        combined_confidence = (acoustic_confidence + vision_confidence) / 2
        return FusionResult(
            agreement=True,
            primary_label=acoustic_label,
            primary_confidence=combined_confidence,
            acoustic_label=acoustic_label,
            acoustic_confidence=acoustic_confidence,
            vision_label=vision_label,
            vision_confidence=vision_confidence,
            flagged_for_review=False,
            combined_confidence=combined_confidence,
        )

    # Disagreement: surface both, use the higher-confidence signal as primary, flag it.
    if acoustic_confidence >= vision_confidence:
        primary_label, primary_conf = acoustic_label, acoustic_confidence
    else:
        primary_label, primary_conf = vision_label, vision_confidence

    return FusionResult(
        agreement=False,
        primary_label=primary_label,
        primary_confidence=primary_conf,
        acoustic_label=acoustic_label,
        acoustic_confidence=acoustic_confidence,
        vision_label=vision_label,
        vision_confidence=vision_confidence,
        flagged_for_review=True,
        combined_confidence=None,
    )


if __name__ == "__main__":
    # Quick manual sanity checks - no pytest needed for a 2-day sprint.
    print(fuse_predictions("fresh", 0.82, "fresh", 0.75).summary())
    print(fuse_predictions("fresh", 0.60, "stale", 0.71).summary())
