# Business Case

## Problem

Identifying a flower's species from a photograph requires botanical expertise that
most people — home gardeners, retail staff, casual hikers — don't have. Manual
identification does not scale: it requires either an expert on hand or slow,
error-prone lookup through field guides.

## Why this is a good automation target

- **Visual, repetitive, high-volume task.** Fine-grained image classification is
  exactly the kind of pattern-matching problem deep learning excels at, and where
  a trained model can process far more images per second than a human expert.
- **Pretrained backbones transfer well.** ImageNet-pretrained CNNs already encode
  general visual features (edges, textures, shapes, color patterns) that are
  directly useful for distinguishing flower species, which is why transfer
  learning is the right approach rather than training from scratch on a few
  thousand images.
- **Low cost of a wrong answer, high value of a right one.** Unlike safety-critical
  classification tasks, an incorrect flower prediction is rarely harmful — this
  makes the problem well suited to an initial automated system with human review
  as a fallback for low-confidence predictions.

## Target users

- **Consumer plant-ID apps** (e.g., gardening or nature apps) — instant species
  suggestions from a photo.
- **E-commerce / nursery catalogs** — auto-tagging product photos by species.
- **Citizen-science / biodiversity monitoring** — assisting volunteers logging
  plant sightings.
- **Education** — botany students verifying identifications.

## Value proposition

A classifier that reaches strong top-1/top-5 accuracy on 102 fine-grained species
removes the need for a botanical expert in the loop for the majority of cases,
turning an expert-dependent task into a self-serve one. Reported confidence scores
let the system flag uncertain predictions for human review, rather than silently
guessing.

## Risks and limitations

- **Fine-grained visual similarity.** Many flower species are visually similar
  (same family, similar color/shape); errors are expected to concentrate on these
  confusable pairs rather than being random — see the "most confused classes"
  analysis in the evaluation report.
- **Distribution shift.** The model is trained on Oxford Flowers 102, a curated
  dataset of common UK/garden species. Performance on photos taken in different
  conditions (poor lighting, occlusion, unfamiliar species not in the training
  set, phone-camera artifacts) is not guaranteed and would need validation before
  production use.
- **Closed-set assumption.** The model can only predict one of the 102 trained
  classes — it has no mechanism to say "none of the above" for an out-of-distribution
  flower unless an explicit confidence threshold / rejection rule is added.
- **No fairness/bias axis here**, since the task is botanical rather than about
  people — but data quality (class imbalance, image quality per class) still
  affects per-class reliability; see per-class metrics in the evaluation report.

## What would be required before production deployment

1. **Out-of-distribution handling** — a confidence threshold (or a dedicated
   "unknown" detector) so the system doesn't confidently mislabel a flower species
   outside the training distribution.
2. **Broader, representative data** — the current dataset skews toward
   well-lit, centered, single-flower photos; real user photos are noisier.
3. **Monitoring** — track prediction confidence distributions and (if feasible)
   collect user feedback/corrections to catch drift over time.
4. **Latency/footprint validation** on the target platform (mobile vs. server),
   informing whether the current backbone size is appropriate or should be
   distilled/quantized further.
5. **Legal/licensing check** on the training data and any deployed model weights.
