# Cat Pain / FGS: Field Coverage, White Space, and the Gap We Should Own

## 1. The field in one page

| Paper | What it nailed | What it left open |
|---|---|---|
| **Evangelista 2019** (FGS development) | Created and psychometrically validated the FGS (5 AUs, 0/1/2); derived the load-bearing **>0.39 analgesia cut-off** (AUC 0.94, sens 90.7 / spec 86.6); inter-rater ICC 0.89. | 100% manual human scoring; no automation; no brachy/dark-coat/kitten; cut-off never tested *interventionally*; data on request only. |
| **Feighelstein 2022** (binary PoC) | First head-to-head landmark-ML vs ResNet50 on cats; clean LOSO splits; showed 48-landmark geometry ≈ raw image (~72%). | Binary only; manual landmarks; single homogeneous cohort (young female DSH); no graded output; no foundation model / VLM. |
| **Feighelstein 2023** (explainable, noisy pop.) | Tested generalization on heterogeneous 84-cat clinic population; XAI region importance (**mouth > eyes > ears**); LDM (77%) beat DL (65%). | Binary; one frame/cat; manual landmarks; no external set; cannot separate pain from distress; no calibration/CIs. |
| **Steagall 2023** (automated FGS) | **Only graded per-AU automated FGS** (37 landmarks → 35 geometric descriptors → XGBoost; per-AU 0/1/2 + total + 0.39 decision); 95.5% binary, smartphone-arguable. | Hand-crafted geometry, not learned features, no ordinal loss; needs 8 real-time vet raters; data **withheld for app**; no calibration, no CIs, components tested in isolation; no app shipped. |
| **Martvel 2024** (video) | First end-to-end **raw-video** binary pipeline (YOLOv8 + ELD landmarks + temporal XGBoost); showed dynamics help; cross-dataset transfer attempted. | Binary; transfer largely **failed** (Finka→TiHo −0.15 acc); only acc/F1 (no ROC/CI/calibration); landmark-API only released; analgesia confound not disentangled. |
| **Marangoni 2026** (brachy ocular pain) | Showed FGS **responsiveness** + reliability in a hard population; surfaced e-collar and image-vs-real-time overestimation confounds; muzzle/whisker AUs break down in brachy. | Human raters only; no control group; automated brachy FGS = future work; no graded 0–10; single-reviewer real-time gold. |
| **Lee & Steagall 2026** (COSMIN review) | Field-level audit: FGS = highest evidence; named **responsiveness, interpretability, feasibility, ROC analgesia thresholds** as the open needs. | **Explicitly excluded all automated/AI scorers**; no meta-analysis; no new data; named gaps left unfilled by design. |
| **Namboonlue 2023** (Thai CNN thesis) | Independent non-European cohort; EfficientNetB7 79%; Grad-CAM (center-face attention); honest overfitting/confound discussion (Smudge meme false positive). | Binary; no validated scale as ground truth; no subject counts/leakage control; severe overfit; nothing released. |

**One-line synthesis:** The field has validated the human FGS + 0.39 threshold, automated *binary* pain repeatedly, automated *graded* FGS exactly once (closed, hand-crafted, no learned features), and shipped **zero** runnable artifacts, **zero** calibration/CIs, **zero** open benchmarks, and **zero** measurement of its own training-label reliability.

---

## 2. The white space — gaps NONE of the 8 filled

**Group A — Trustworthy-decision layer (the strongest, most-converged white space)**
- **Calibration + cost-sensitive operating point at 0.39** — no ML paper reports a reliability diagram, Brier/ECE, or decision-curve under undertreatment≫overtreatment asymmetry. Clinical papers derived 0.39 on *humans*; ML papers inherit it as a constant. *Novelty: high · Feasibility: high · Value: high.*
- **Uncertainty-aware defer-to-vet abstention** evaluated as accuracy-vs-coverage — Martvel filters low-confidence frames internally but nobody surfaces it as an output. *high · high · high.*
- **Pain-recall-at-fixed-high-sensitivity with bootstrap 95% CIs stated up front** (CI half-width honesty) — never reported under real prevalence. *medium · high · high.*

**Group B — Label honesty (forced on us by our data, owned by no one)**
- **Weak-label reliability as a measured quantity** — treat the VLM 5-AU labeler as a *rater*; report per-AU agreement (quadratic κ + CI) vs a vet. No paper reports reliability of its *training* labels (only of human FGS raters). *high · high · high.*
- **Label-provenance + capture-condition confound audit** — does the binary signal predict pain or brightness/blur/box-aspect/morphology? Flagged by Martvel/Namboonlue/Feighelstein, **quantified by none**. *medium–high · high · high.*

**Group C — Modeling novelty**
- **Frozen foundation-model backbone (DINOv2 ViT-S)** — every paper uses ImageNet ResNet/EfficientNet/ShuffleNet or landmark geometry; zero foundation models. *high · high · high.*
- **Learned graded per-AU output via rank-consistent ordinal heads (CORN)** — the only graded paper (Steagall) used XGBoost on descriptors, no ordinal loss, no learned features. *high · medium · high.*
- **VLM/LLM weak-labeling of the 5 AUs** — all 8 papers are pure vision; the per-AU annotation bottleneck is named repeatedly but the proposed fix is always "automate landmarks," never "semantically weak-label the AUs." *high · medium · high.*

**Group D — Open artifact**
- **Open, leakage-audited benchmark + datasheet** with honest pseudo-label provenance and a frozen 0.39 protocol — every dataset is request-only/withheld. *Caveat below in §6.* *medium–high · medium · medium.*

---

## 3. What we should NOT do (replication traps)

1. **Do NOT re-do Steagall's landmark → geometric-descriptor → XGBoost pipeline.** It is done, it is the field's graded baseline, and it needs 8 real-time vet raters we don't have. Replicating it = a worse, closed-data clone.
2. **Do NOT re-do Martvel's video/temporal pipeline.** We have no video, no timestamps, no paired frames. Temporal modeling is structurally impossible on our data.
3. **Do NOT chase the binary-accuracy leaderboard (77 / 79 / 95%).** Saturated, on tiny separable curated sets. "Beat 79%" is not a contribution and our confounded labels make it dishonest.
4. **Do NOT claim external-cohort validation.** We have one confounded Flickr-derived source (single CAT_01 camera). Every external pain/FGS set is request-only. Name single-source as our top threat-to-validity; do not pretend to fill it.
5. **Do NOT promise responsiveness / longitudinal within-cat rescoring.** It is the field's #1 named need (tempting) but requires paired pre/post-analgesia timestamped frames we do not have. Cite as out-of-reach.
6. **Do NOT promise multi-rater model-vs-human ICC/Bland-Altman.** Needs a rater panel; we have at most one vet. Borrow the vocabulary, not the claim.
7. **Do NOT headline cross-species horse transfer.** 5 horses, 3 of 5 AUs (no whiskers, no head), grouped ≈ 5 datapoints. Use it only as **pipeline-validation scaffolding** (does CORN-decode → sum → threshold work on *real* graded labels?), never as a measured transfer gain.
8. **Do NOT center a shipped mobile app or multimodal posture/audio.** App = solo-eng scope creep (a minimal HF Space demo is fine as a delivery vehicle, not the thesis); posture/audio = no body/audio data.

---

## 4. THE GAP WE SHOULD OWN

> **Build the field's first *learned graded* FGS scorer whose headline deliverable is TRUSTWORTHINESS AT THE 0.39 DECISION POINT, not accuracy — and whose every claim is explicitly conditioned on a measured weak-label-reliability number and a confound audit.**

Concretely, one vertical:

1. **Engine (modeling novelty):** frozen **DINOv2 ViT-S/14** (the `_reg` register variant, `dinov2_vits14_reg`, is the field default — registers suppress attention artifacts that hurt the localized orbital/ear/muzzle features per-AU FGS needs; A/B it before locking ViT-S/14) + **5 rank-consistent CORN ordinal heads** → per-AU 0/1/2 → 0–10 → 0.39 decision.
2. **Supervision (never attempted):** **VLM weak-labels the 5 AUs**; a vet reviews a small calibration anchor; active-learning prioritizes uncertain/severe cases. This is the *only* path from binary/web data to graded FGS without a closed expert corpus.
3. **Honesty wrapper (the actual headline):** VLM-as-rater per-AU **quadratic κ vs vet**; **calibration** (reliability diagram, Brier/ECE); **pain-recall-at-fixed-sensitivity with bootstrap 95% CIs (half-width stated first)**; **decision-curve under undertreatment≫overtreatment**; **defer-to-vet abstention curve**; **capture-condition confound audit** (trivial brightness/blur/aspect classifier); face-quality/morphology input gate.
4. **Artifact:** released weak-labeled AU annotations + frozen-backbone+CORN code/weights + frozen clip-grouped test split + **datasheet** documenting pseudo-label provenance and the confound audit.

**Why this is novel against each relevant paper:**
- **vs Steagall 2023** (the only graded work): we use *learned* features (frozen ViT) + an *ordinal loss* (CORN) instead of hand-crafted geometry + XGBoost; we replace 8 real-time vet raters with VLM-weak-labeling + a small vet anchor; and we report calibration/CIs/abstention/confound audit — *all* absent from Steagall. Our data is open; theirs is withheld.
- **vs Feighelstein 2022/2023 & Martvel 2024 & Namboonlue 2023:** they are binary; we are graded per-AU. They are ImageNet-CNN/landmark; we are foundation-model. None uses any VLM/LLM. None reports calibration, CIs, abstention, or training-label reliability.
- **vs Evangelista 2019 & Marangoni 2026:** they derived/tested 0.39 on *human* raters with no model to calibrate. We attach the calibrated, cost-sensitive, abstention-aware decision layer to an *automated* scorer.
- **vs Lee & Steagall 2026:** they named ROC thresholds + interpretability + feasibility as the field's open need and **explicitly excluded automated scorers**. We deliver exactly the automated-scorer measurement layer they could not assess.

**Why it's feasible on our stack:** frozen backbone → only light heads train (M4/Colab T4); calibration, abstention, bootstrap CIs, κ, and confound audits are all **post-hoc analyses on held-out scores** — no FGS gold corpus, no second cohort, no video, no large compute. The horse set is used once to sanity-check the CORN decode path on genuine 0/1/2 labels.

**Resolving the lens disagreement:** the five analyses circle the same point from different sides. Modeling says "DINOv2+CORN+VLM is the novel engine"; clinical/methodology/data say "calibration + label-honesty + confound audit is the unowned validity ground"; product says "ship it open." These are **not competing** — they are the engine, the wrapper, and the delivery vehicle of *one* contribution. The headline is the **validity wrapper** (it is what's genuinely unowned and non-replicable); the graded engine is what makes a wrapper worth having; the open release is the delivery. The "open benchmark" framing is demoted to "open *audit + datasheet + protocol*" (not a leaderboard implying trustworthy labels we don't have).

---

## 5. Concrete project framing (how this updates BUILD_PLAN.md)

**Target.** A graded per-AU FGS scorer (frozen DINOv2 ViT-S + 5 CORN heads, VLM-weak-labeled + vet-anchored AUs) delivered as a **calibrated, abstention-aware, confound-audited decision-support artifact** at the 0.39 cut-off.

**The claim we could make (and defend).** *"The first learned graded FGS scorer reported as a trustworthy clinical decision: we measure our own weak-label reliability (VLM-vs-vet per-AU κ), calibrate the 0.39 analgesia decision (reliability diagram, Brier/ECE), report pain-recall with bootstrap CIs and a defer-to-vet abstention policy, and audit the acquisition/morphology confound — none of which the 8 prior cat-pain works did, and which the COSMIN review named as the field's open need but could not assess for automated scorers."*

**Minimal validation to support it (the GO/NO-GO gates):**
- **Gate 1 — Label reliability:** ~120-image VLM-vs-vet per-AU quadratic κ pilot with CI. A high κ only measures weak-labeling *capability* if the vet anchor is independent and the rubric handed to the VLM differs from the vet's rubric; otherwise it measures rubric-following — state which a given run measures. If κ is too low, AUs that fail are reported as such (honest), not hidden.
- **Gate 2 — Confound audit:** trivial classifier (brightness/blur/box-aspect/CLIP) on the binary label. If it beats chance, report pain as entangled with acquisition context and condition all downstream claims on it.
- **Gate 3 — Decision layer:** calibration + pain-recall-at-fixed-sensitivity with bootstrap 95% CI half-width stated up front (~±0.13 on ~50-positive folds); 0.39 decision estimated **only on vet-confirmed labels** to break VLM-label circularity; abstention accuracy-vs-coverage curve.
- Splits: **clip-grouped StratifiedGroupKFold** (no same-cat leakage); frozen hashed test split.

**First 3 actions:**
1. **Run Gate-2 confound audit now** on the existing binary Roboflow data (trivial-shortcut classifier) — cheapest, decides whether the corpus is salvageable and frames the whole paper's honesty.
2. **Stand up the VLM 5-AU weak-labeler + the ~120-image vet-review pilot** to get the first Gate-1 per-AU κ number (the precondition every downstream claim rides on).
3. **Wire frozen DINOv2 ViT-S embedding extraction + a CORN ordinal head, validated first on the horse 0/1/2 labels** (decode → sum → threshold sanity check) before touching cat AUs.

---

## 6. Honest risks

- **Label reliability could fail Gate 1.** If VLM-vs-vet κ is poor on muzzle/whiskers (exactly where Steagall and Marangoni both report low inter-rater reliability), graded per-AU output may be untrustworthy. *Mitigation:* report κ honestly per-AU; ship the calibrated **binary** decision layer with abstention as the v1 floor — the likely v1 ship that stands today, with the graded 0-10 layer as upside conditional on Gate 1-B passing; the validity-wrapper contribution stands either way.
- **Confound audit could be damning (Gate 2).** If pain is largely predicted by brightness/blur/morphology, our labels are mostly shortcut. *This is still a publishable finding* (the first measured confound audit), but it kills any accuracy claim — which is why we don't headline accuracy.
- **"Novel" calibration could be seen as routine ML hygiene.** Calibration/abstention are standard elsewhere; a reviewer may say "applying known methods." *Mitigation:* the novelty is the *combination on graded FGS + the measured weak-label-reliability conditioning + the confound audit*, which is genuinely unowned in this research line, plus the COSMIN-named gap framing.
- **Open-benchmark trap.** Releasing a dataset whose pain labels we ourselves distrust is a liability, and CatFLW (best alignment asset) is **CC BY-NC** — contaminating a clean open release and blocking commercial use. *Mitigation:* release the **audit + datasheet + protocol + code/weights**, not a "trustworthy-labels leaderboard."
- **Single-source ceiling.** One confounded Flickr camera means no external validity; wide CIs are unavoidable. *Mitigation:* name it as the top threat-to-validity up front rather than over-claiming.
- **VLM AU-grounding may be weak.** VLMs may not reliably localize feline-specific AUs (orbital tightening, whisker change). *Mitigation:* the Gate-1 κ pilot *measures* this directly before any training spend — failure is detected cheaply, not after the model is built.