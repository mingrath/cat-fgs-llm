# NEEDS FROM YOU — cat-fgs-llm

Everything the pipeline can't get by itself. Code is written and the blocking gates
pass (Gate 4 decode + MPS parity, leakage-safe folds — all green on py3.11/MPS).
What's left is **keys, two decisions, and one vet sitting**. Work top to bottom.

Legend: `[ ]` = you do it · `[me]` = I do it (once unblocked) · `<FILL>` = paste/replace inline.

---

## A. Right now — 1 key (≈30 seconds, unblocks ~everything)

- [ ] **Anthropic key.** Open `.env` (gitignored, never committed) and fill the slot:
      ```
      ANTHROPIC_API_KEY=<FILL — sk-ant-...>
      ```
      Your **Roboflow key is already set** ✓. W&B is optional (leave empty to skip tracking).

- [ ] **Kaggle token (optional, for CatFLW alignment landmarks — Gate 5).**
      Download `kaggle.json` from kaggle.com → Account → Create New API Token, then:
      ```
      mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
      ```
      Status: `[ ] done`  ·  `[ ] skip for now` (Gate 5 can fall back to the Haar cropper)

> Once the Anthropic key is in, tell me and I run section B unattended.

---

## B. Then I run unattended (no vet, no decisions needed)

- [me] Fork `lia-k4jkv/cat-pain-ul7lu` → `mingraths-workspace` + regenerate a **Fit (reflect) 640**
      version (the shipped Stretch version distorts AU geometry) + export COCO/YOLO.
      *(I may surface one Roboflow UI link for you to click if the MCP fork needs confirmation.)*
- [me] Download assets: `facebook/dinov2-small` (cached ✓; the `_reg`/register variant `dinov2_vits14_reg`
      is the field default — A/B it before locking ViT-S/14, plain stays as compat), horse-grimace fixture, Haar cat-face cascade.
- [me] **Gate 0** power calc → writes `data/manifests/power.json` = the vet-budget integer + κ floors
      (this fixes **how many images you label** in section C).
- [me] **Gate 1** per-CAT merge → folds.csv · **Gate 2** confound audit · **Gate 3** freeze hold-out.
- [me] **Gate 5** crop/align audit on 30–50 images.
- [me] **VLM pre-label** the face crops (5 AUs each, 0/1/2 + rationale) → builds your review queue.

---

## C. The vet sitting — THE ONE STEP ONLY YOU CAN DO 🐾

After B, I generate a review file at `data/pilot/vet_review.csv` — one row per image, each
with the VLM's 5 AU pre-fills (ear / orbital / muzzle / whiskers / head, scored 0/1/2) **and its
rationale**. You **accept or correct**, you do NOT score from scratch.

- [ ] Review ≈ **`<N — set by Gate 0, ~120>`** images, of which **≥50 pain-positive**.
      Estimated effort: ~4–5 hours. Fill the `vet_*` columns (the template will spell out which).
- [ ] (Optional) Where the VLM and you disagree, a one-line note helps the datasheet.

When that CSV comes back, **Gate 1-B** fires: the per-AU VLM-vs-vet quadratic κ on the CI lower bound.
This κ is the project's **guarded, inspected-not-validated reliability check**, not the headline — the
headline is the power-aware, per-AU confound-attribution protocol. Gate 1-B → GO (add the graded 0-10
layer as upside on top of the floor) or HOLD at the floor (the binary-plus-abstention spine that ships
today either way). The κ check stays in either case as kill-tree insurance. Both outcomes are
publishable as a methods/protocol paper.

> **Interpretation guard:** a high κ here measures weak-labeling *capability* only if the
> vet anchor is independent **and** the rubric handed to the VLM is not the same rubric the
> vet scored from. Note that accept-or-correct over the VLM pre-fills makes the anchor
> *non-independent* unless the vet scores blind — read such a κ as rubric-following, not
> weak-labeling skill.

> Review-UI preference (pick one — I'll wire the export to match):
> `[ ]` plain CSV/spreadsheet  ·  `[ ]` Roboflow review  ·  `[ ]` CVAT / Label Studio  ·  `[ ]` other: `<FILL>`

---

## D. Two clinical decisions (you can defer — sane defaults are pre-set)

- [ ] **Undertreat : overtreat harm ratio** (drives the welfare decision curve). We have no elicited
      value, so the code **sweeps it as a range** by default — fine to leave. If you have a clinical
      view, give a range: `<FILL e.g. 3:1 to 10:1>`  → `configs/wrapper.yaml`.
- [ ] **κ floors** (Gate 1-B GO/NO-GO). Defaults: orbital/ear/head LB ≥ 0.60, muzzle/whiskers ≥ 0.40.
      Override only if you want stricter/looser: `<FILL or leave default>` → `configs/vlm_fgs.yaml`.

---

## E. Optional / deferrable

- [ ] **Colab T4** for the RF-DETR detector sweep (Phase A binary detector).
      **Deferrable:** the Roboflow dataset already ships pain/no_pain boxes, so I can crop from those
      ground-truth boxes to build and validate the whole wrapper without training a detector first.
      The detector is only needed for the final *deployable* binary-decision metric.
      Status: `[ ] have Colab`  ·  `[ ] defer the detector`

---

### TL;DR of what I'm blocked on right now
1. `ANTHROPIC_API_KEY` in `.env`  ← the only hard blocker to start section B.
2. Your **vet sitting** (section C) — the single irreplaceable human step.
Everything else has a default or is something I do.
