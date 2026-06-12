# GitHub Mining — Additive Briefing (beyond HF_SOLUTION.md / DATA_DECISION.md)

## 1. Did anything NEW turn up?

**Verdict on new downloadable cat-pain/FGS data: effectively NO.** No new open, directly-downloadable cat dataset with graded per-AU FGS (or even expert binary cat-pain) labels exists on GitHub. The entire published cat lineage (Steagall, Evangelista, Finka/TiHo, **and now the named Zamansky/Feighelstein 464-img GitLab repo**) is confirmed request-only or dead-linked. This does **not** change the DATA_DECISION conclusion.

**But the mining produced four genuinely new, useful assets** not in HF_SOLUTION/DATA_DECISION:

1. **`mytalbot/MGS_data`** — real **open per-AU ordinal grimace labels** (Mouse Grimace Scale, 5 AUs incl. orbital + ear + **whisker change**), 4,944 rows. The *only* openly downloadable dataset whose label shape matches our 5-AU 0/1/2 target — and it covers a whisker AU, our structural blind spot.
2. **`Barn99/Automated-RGS`** — pretrained **ViT per-AU grimace grader + YOLOv5 AU-region detector** (Rat Grimace Scale, weights on Google Drive). A working instance of our exact planned architecture (region-detect → per-AU ordinal head).
3. **Exact request target located:** `gitlab.com/is-annazam/automated-recognition-of-pain-in-cats` — the named (now 404/upon-request) repo for the **464-img balanced expert binary cat-pain set + 48 CatFACS landmarks**, author **Anna Zamansky / Martvel**. Gives DATA_DECISION §5's "email Zamansky" a precise repo+author handle.
4. **`marroyol/comp3000-marie`** — **directly downloadable** ~44-img **4-level ordinal pain-likelihood** labels keyed to CatFLW filenames, with **two passes + a second rater** (ready-made inter-rater kappa fixture). Non-expert/student labels, so not gold — but a free, multi-rater calibration fixture on real CatFLW crops.

Plus a public **Martvel landmark-detection Colab API** (the only public form of the IJCV 48-landmark cat detector — no GitHub repo exists) and concrete reusable training/wrapper code (below).

## 2. New datasets / data links (genuinely new only)

| name | url | pain-labeled? | labels | license | access |
|---|---|---|---|---|---|
| MGS_data (Mouse Grimace) | github.com/mytalbot/MGS_data | yes (mouse) | **5 ordinal AUs incl. whisker change** (orbital, nose, cheek, ear, whisker), 4,944 rows; -1=rejected | LICENSE file in repo (verify) | open `git clone` — ships scores `MGS_raw.txt`, **not** source images |
| Automated-RGS weights (Rat Grimace) | github.com/Barn99/Automated-RGS | yes (rat) | pretrained ViT per-AU grader (orbital/ear/nose) + YOLOv5 AU detector; no whisker | not stated | code open; weights via [Drive folder](https://drive.google.com/drive/folders/1Cl_5GyouX7sDLv1NUKuq_YxrrQRQMYKn) |
| comp3000-marie pain_labels | github.com/marroyol/comp3000-marie/tree/main/pain_labels | yes (cat, **non-expert**) | 4-level ordinal pain-likelihood, keyed to CatFLW filenames; pass1+pass2+rater2 | no LICENSE (images=CatFLW CC BY-NC) | open, raw.githubusercontent download |
| Zamansky/Feighelstein cat pain (464 img) | gitlab.com/is-annazam/automated-recognition-of-pain-in-cats | yes (cat, **expert binary**) | binary pain/no-pain (ovariohysterectomy protocol) + 48 CatFACS landmarks, 26 cats | unspecified/academic | **404 / request-only** — email Anna Zamansky, Tech4Animals, U. Haifa |
| htcv_mgs labels+weights | github.com/PatrickFreund/htcv_mgs | yes (mouse, **binary only**) | labels.csv binary 0/1 (NOT per-AU) + per-fold `best_model.pth` | not stated | open |
| CatFACS coding manual | animalfacs.github.io/AnimalFACS/CatFACS | no (manual) | defines canonical cat facial AU scheme | manual free; videos permission-gated | [Drive](https://drive.google.com/drive/folders/1uO0DXexVg9V62TC3fGBvHkqz84J5lEc1) |
| EquinePainFace annotations | github.com/jmalves5/EquinePainFaceDataset | yes (horse) | graded equine pain-face annotations (XLS/JSON) tied to PLOS ONE pone.0231608 | unspecified | repo + PLOS supplementary |

*(CatFLW, the horse-grimace set, and the Steagall/Evangelista request-only sets are already in DATA_DECISION/HF_SOLUTION — not repeated.)*

## 3. Repos to clone now

### (a) Paper code / weights
| repo | what it gives us | reuse value |
|---|---|---|
| github.com/martvelge/CatFLW | Official CatFLW repo; per-image JSON `{labels:(48,2), bounding_boxes}` format spec + Kaggle link (confirms the download already in DATA_DECISION) | high (format reference) |
| [Martvel landmark Colab](https://colab.research.google.com/drive/1XmTL3qJ2mMfb4FfCdwhnDW5jVUWNYTbi) | **Only public form** of the IJCV magnifying-ensemble 48-landmark cat detector (no GitHub exists). Auto-detect 48 landmarks on our Roboflow images → AU-region crops without manual annotation | high |
| github.com/martvelge/dog_emotions_LLMs | Same lab's official code (AGPL-3.0) for **prompting VLMs over cropped animal faces + YOLO11** — direct template for our VLM weak-labeling step | medium |
| github.com/Barn99/Automated-RGS | YOLOv5 AU-detector + per-AU ViT ordinal grader **with downloadable weights** — warm-start candidate / architecture proof | high |

### (b) Reusable training code (DINOv2 + CORN, face-align)
| repo | what it gives us | reuse value |
|---|---|---|
| github.com/arthur-thuy/qde-ordinality | Backbone-agnostic ordinal head (`RobertaOrdinalHead`) that takes pooled CLS embeddings and switches CORAL/CORN/OrderedLogit by config — **best template to drop onto DINOv2 `features[:,0,:]`**; wrap 5 in `nn.ModuleList` | high |
| github.com/itsprakhar/Downstream-Dinov2 | Frozen-DINOv2 extractor + swappable head + `train_classifier.py` ImageFolder entrypoint — scaffold; replace single Linear with the 5 CORN heads (MPS-friendly, backbone frozen) | high |
| github.com/marinbenc/dermatoscopy_colorimetry_eval | Concrete **CORAL-on-image-backbone training loop** (`CoralLayer` + `levels_from_labelbatch` + `coral_loss`) — copy-paste ordinal-grading recipe | high |
| github.com/chelsea23311/Cat-Face-Landmark-Detection | Complete cat-face 9-keypoint pipeline (ResNet-50, train/test/predict, **interocular-distance normalization**, PCK/NME) — ready-made eye-alignment step for Gate-5 / the §3 NME audit; weights on releases | medium |
| github.com/RobvanGastel/dinov3-finetune | DINOv2/v3 **LoRA** path for later T4 unfreezing if frozen linear-probe underfits | medium |
| github.com/Raschka-research-group/coral-pytorch | Canonical CORN/CORAL lib (already named in our plan; confirmed: `CoralLayer`, `corn_loss`, `corn_label_from_logits`, `levels_from_labelbatch`) | high (already planned) |

### (c) Trustworthiness-wrapper implementations (calibration / abstention / noisy-label)
| repo | what it gives us | reuse value |
|---|---|---|
| github.com/ENSTA-U2IS-AI/torch-uncertainty | **One-stop wrapper backbone (Apache-2.0):** temp/vector/matrix/Dirichlet scaling + ECE/SmoothECE/AdaptiveECE/**ClasswiseECE** + reliability diagrams **AND** selective-classification AURC/AUGRC/CovAt5%Risk/RiskAt80%Cov + conformal APS/RAPS. Covers 2 of 4 wrapper pillars off the shelf | high |
| github.com/cleanlab/cleanlab | Confident-learning weak-label audit (`find_label_issues`, multi-label) — ranks suspect VLM FGS labels into a send-to-vet queue (already named; now confirmed multi-label support for the 5-AU case) | high |
| github.com/dholzmueller/probmetrics | **Maintained** temperature-scaling + ECE/Brier/NLL/SmoothECE (pip, v1.3.0 2026) — use this for the actual temp fit, not the unmaintained gpleiss repo | high |
| github.com/EFS-OpenSource/calibration-framework | netcal — most complete standalone calibration toolbox + reliability-diagram plotting | high |
| github.com/apple/ml-calibration (`relplot`) | Kernel-smoothed reliability diagrams + SmoothECE (ICLR2024) — **important because our FGS set is small; binned ECE is unstable at low n** | high |
| github.com/MSKCC-Epi-Bio/dcurves | Maintained Vickers **decision-curve / net-benefit** analysis — clinical-utility argument at the 0.39 threshold | high |
| github.com/bhanML/Co-teaching | Official PyTorch **co-teaching** small-loss selection — exactly the noisy-VLM-label training already specified in HF_SOLUTION step 8 | medium |
| github.com/IdoGalil/benchmarking-uncertainty-estimation-performance | ICLR2023 AURC + **SAC** (max coverage at accuracy target) — frames "defer-to-vet" as a publishable metric | medium |

## 4. Ideas / references worth borrowing

- **Pretrain/validate the 5 CORN heads on `mytalbot/MGS_data` first** — real open per-AU ordinal labels (incl. a whisker AU) to de-risk the heads before spending cat/vet data, alongside the horse warm-start already in HF_SOLUTION. MGS covers whisker; horse does not — partial relief for our whisker/head blind spot.
- **Adopt `Barn99/Automated-RGS`'s two-stage pattern explicitly** (region-detect → per-AU ordinal head) and consider fine-tuning from its released rat ViT weights as a warm start.
- **Head module:** copy `qde-ordinality`'s `RobertaOrdinalHead` (dropout→dense→tanh→dropout→`out_proj`), feed DINOv2 CLS, `out_proj = nn.Linear(384, K-1)` with K=3; wrap 5 in `nn.ModuleList`. Loss = sum of 5 `corn_loss`; decode per-AU via `corn_label_from_logits` → 0–10 sum → 0.39 in code. CORAL-vs-CORN ablation is a one-flag config switch.
- **Use `comp3000-marie`'s pass1/pass2/rater2 as a pre-built inter-rater kappa fixture** to validate our VLM-weak-label kappa pipeline on real CatFLW crops before any vet hour.
- **Wrapper:** lean on `torch-uncertainty` for calibration+abstention; report `relplot` SmoothECE alongside binned ECE (small-n); per-AU `ClasswiseECE` so miscalibration is attributable to a specific AU; frame abstention as SAC (Galil ICLR2023); add `dcurves` net-benefit at 0.39.
- **Rubric/prompt spec (free, defensible):** pull the **CatFACS manual** (Drive) and `etho-backend`'s CatFACS→FGS AU-mapping table (`research/02_CATFACS_FGS.md`, encodes mean ≥0.39) to anchor the VLM weak-label prompt + vet rubric.
- **AU-importance ablation** (Talbot "orbital tightening alone" finding): test whether a subset of the 5 cat AUs retains most predictive power — informs which AUs to prioritize for scarce vet review and whether whiskers/head can be dropped.

## 5. Updated bottom line

**GitHub does NOT change the conclusion** that we must self-label via VLM + vet review. No new open cat FGS/per-AU dataset surfaced; the published cat lineage — now including the precisely-named **Zamansky 464-img GitLab repo** — is all request-only or dead-linked. The VLM-weak-label → vet-arbitration → frozen-DINOv2 + 5-CORN pipeline remains mandatory.

What GitHub **does** change: (a) gives DATA_DECISION §5's Zamansky email an exact repo+author target (likely bundles CatFLW-style landmarks since Martvel is co-author); (b) supplies an **open per-AU ordinal pretraining set with a whisker AU** (MGS_data) and a **transferable pretrained per-AU grimace head + weights** (Automated-RGS) to extend the horse-only warm-start; (c) provides a free multi-rater kappa fixture (comp3000-marie); (d) confirms ready-to-use code for every wrapper pillar.

### Exact next commands

```bash
# --- New per-AU ordinal pretraining + transferable grimace head/weights ---
git clone https://github.com/mytalbot/MGS_data ./datasets/mgs          # 5 mouse AUs incl. whisker (scores only)
git clone https://github.com/Barn99/Automated-RGS ./models/automated-rgs # ViT per-AU + YOLOv5; weights via Drive folder in README

# --- Multi-rater cat pain-likelihood fixture (kappa sanity) on CatFLW crops ---
git clone https://github.com/marroyol/comp3000-marie ./datasets/comp3000-marie

# --- Reusable training code: ordinal head + frozen-DINOv2 scaffold + CORAL loop + eye-align ---
git clone https://github.com/arthur-thuy/qde-ordinality ./ref/qde-ordinality
git clone https://github.com/itsprakhar/Downstream-Dinov2 ./ref/downstream-dinov2
git clone https://github.com/marinbenc/dermatoscopy_colorimetry_eval ./ref/coral-image-loop
git clone https://github.com/chelsea23311/Cat-Face-Landmark-Detection ./ref/cat-landmark-align

# --- VLM weak-label template (same lab, animal faces) ---
git clone https://github.com/martvelge/dog_emotions_LLMs ./ref/animal-vlm

# --- Trustworthiness-wrapper deps (pip; not git) ---
pip install torch-uncertainty cleanlab probmetrics netcal relplot dcurves

# --- Optional: LoRA upgrade path + co-teaching reference ---
git clone https://github.com/RobvanGastel/dinov3-finetune ./ref/dinov3-lora
git clone https://github.com/bhanML/Co-teaching ./ref/co-teaching
```

**Manual fetches (no CLI):** Martvel 48-landmark detector — open Colab `https://colab.research.google.com/drive/1XmTL3qJ2mMfb4FfCdwhnDW5jVUWNYTbi` (no GitHub repo exists); CatFACS manual — Drive `https://drive.google.com/drive/folders/1uO0DXexVg9V62TC3fGBvHkqz84J5lEc1`; Automated-RGS weights — Drive `https://drive.google.com/drive/folders/1Cl_5GyouX7sDLv1NUKuq_YxrrQRQMYKn`.

**Email target (sharpened):** Anna Zamansky / George Martvel, Tech4Animals Lab, U. Haifa — re `gitlab.com/is-annazam/automated-recognition-of-pain-in-cats` (464-img expert binary cat-pain + 48 CatFACS landmarks).

**Do NOT** treat `comp3000-marie` (non-expert student labels) or `htcv_mgs` (binary only, not per-AU) as gold supervision — fixtures/scaffolds only.

---

# Mining Pass 2 (2026-06-12) — Novelty / differentiation axis

> Sections 1–5 above answer *"what data/code can we clone?"* This pass answers a different question: *"what real GitHub techniques would make us UNIQUE, not a duplicate?"* Produced by a 63-agent grep-MCP workflow (760 grep calls, 46 findings verified, 45 confirmed real by re-grepping inside the named repos). Sorted by how much each strengthens the **trustworthiness wrapper** (the owned contribution), not the engine.

**Headline:** nothing on public GitHub builds a *learned graded* FGS scorer, and nothing pairs ordinal calibration + abstention + confound-counterfactuals on cat pain. The components exist in isolation; the assembled wrapper on a learned graded FGS scorer is unowned. CORN/DINOv2 plumbing is table stakes — adopt and move on.

## P2.1 Adopt now (high differentiation, low/med effort)

| # | Technique | Repo | What to try | Why it makes us unique | Effort |
|---|-----------|------|-------------|------------------------|--------|
| 1 | **Risk-controlled two-threshold abstention** (Learn-then-Test): below λ₁→no-pain, above λ₂→pain, middle→defer; thresholds chosen so NPV/recall is statistically guaranteed | `scikit-learn-contrib/MAPIE` (`plot_risk_control_llm_as_a_judge.py`) | Feed calibrated pain-prob into `BinaryClassificationController`; control **negative_predictive_value** (missed pain = dominant cost) at target α on the vet anchor; report `abstention_rate` next to bootstrap recall CIs | Turns the defer-to-vet curve into a **distribution-free finite-sample guarantee** on undertreatment risk. Stronger than any prior cat-pain work; seconds on CPU. **Coexists with** the frozen 0.39 cutoff (band selector λ₁,λ₂ around 0.39, not a replacement) | low |
| 2 | **Foreground-mask background counterfactual** (ImageNet-9 BG-challenge): build `mixed_rand`/`only_bg_t`, measure score shift when only background swaps | `MadryLab/backgrounds_challenge` (via `bytedance/ibot`) + `visinf/beyond-accuracy` (BG-Gap) | Mask the cat face (DINOv2 attention / SAM / grabcut — no training), composite onto same-FGS vs swapped clinic backgrounds; report **FGS-BG-Gap** = mean 0–10 shift + pain-flip rate, per AU | Makes Gate-2's confound audit **causal not correlational** — proves the score isn't reading the cage/clinic. No prior cat-pain paper proves this | medium |
| 3 | **Energy-based Pointing Game (EBPG)**: fraction of saliency energy inside an anatomical ROI vs whole map | `haofanwang/Score-CAM` (`energyPointGame.py`) + `hungntt/xai_thyroid` | Per-AU saliency (attention-rollout/Grad-CAM on frozen backbone, 1–2 passes), ROI from CatFLW landmarks; report `energy_in_ROI / energy_whole` per AU, stratified by capture condition | The scalar the audit was missing: does the "ear" head fire on the ear? Per-AU localization-faithfulness no prior cat-pain work reports | low |
| 4 | **Ordinal Krippendorff's α** over N≥3 repeated VLM weak-labelings of the same image | `prometheus-eval/prometheus-eval` (`eval/consistency.py`) | `krippendorff.alpha(reliability_data, level_of_measurement="ordinal")` on a (n_runs, n_items) matrix per AU at varied temp/seed | Vet-free reliability axis (VLM self-consistency) complementing vs-vet κ. Cheap pre-screen before vet budget + abstention signal (low-α → defer). Feeds Gate 1-B | low |

## P2.2 Borrow patterns (adjacent) — concrete code to copy

**CORN decode (point estimate + soft probs for calibration):**
- `Raschka-research-group/coral-pytorch` — `corn_loss` + `corn_label_from_logits`. **Critical:** do NOT decode straight to a hard label. Keep `probas = cumprod(sigmoid(logits))` (per-level cumulative P(rank>k)) — those soft ordinal probs feed ECE/Brier/RPS/reliability-diagram/abstention. The `>0.5` hard threshold is point-estimate only. Unit-test against their doctest (`tensor([1,3])`) — this *is* Gate 4's CORN-decode test.
- Vendoring: `ludwig-ai/ludwig` (`modules/loss_implementations/corn.py`) inlines the ~40-line `corn_loss` (torch+F only, MIT) — self-contained, auditable for the datasheet, no pip dep.
- Multi-head: `mueller-franzes/odelia_breast_mri` (`CornLossMulti`) — one wide head width Σ(K_au−1)=10, `torch.split` per AU, `corn_loss` per chunk averaged, decode+stack to [B,5]. Tiny trainable params on frozen trunk.
- Loss ablation: `SocialComplexityLab/life2vec` makes loss a config enum `{corn, cdw, smoothl1, focal}` — borrow the abstraction to A/B CORN vs CDW-CE and report calibration per loss.

**Frozen DINOv2:**
- `chandar-lab/semantic-wm` — freeze idiom verbatim: `Dinov2Model.from_pretrained` → `.requires_grad_(False)` → `.eval()`; read `config.hidden_size` (384 ViT-S); derive patch size from `embeddings.patch_embeddings.projection.stride[0]` not hardcode 14. HF returns CLS at index 0 → slice `[:,1:,:]` for patch tokens.
- `microsoft/Semi-supervised-learning` (`semilearn/nets/dinov2.py`) — `only_feat`/`only_fc` split: one `only_feat` pass to **cache pooled features to disk**, then train the 5 CORN heads off the cache (no backbone forward per epoch — big M4 win). A/B mean-pool vs CLS (AU localization may favor patch-token pooling).
- `BenediktAlkin/vtab1k-pytorch` — init each CORN head `nn.init.trunc_normal_(weight, std=2e-5)` for stable small-data linear-probe; keep their LoRA-on-backbone path as fallback if frozen-linear underperforms.

**Calibration (adapt, don't copy — these are top-label softmax, ours is ordinal):**
- `gpleiss/temperature_scaling` — the `optim.LBFGS([T], lr=0.01, max_iter=200)` NLL closure + bin-and-average ECE skeleton. Per-head scalar T (or shared); report before/after NLL+ECE; ECE on the binarized pain decision, not max-confidence.
- `EFS-OpenSource/calibration-framework` (netcal) — `netcal.metrics.ECE/ACE/MCE` + `ReliabilityDiagram` on the pain decision; **netcal.regression** `ENCE/UCE/QCE/PICP/MPIW` on the 0–10 sum as a predictive distribution (mean+var from per-AU CORN probs). `IsotonicRegression`/`VarianceScaling` to fix miscalibration post-hoc. Skip GP recalibrators (overkill on MPS).
- `awslabs/gluonts` (`discrete_distribution.py rps()`) + `PriorLabs/TabPFN` — add **RPS** (discrete ranked probability score) as the ordinal-aware Brier: `RPS = Σ_k (CDF_pred(k) − CDF_obs(k))²`. CORN emits P(y>k) so the CDF is free; rewards "close on the ordinal scale." Report per-AU + aggregate-FGS RPS with bootstrap CIs.

**Abstention:**
- `facebookresearch/reliable_vqa` — a **learned** selector head: tiny MLP over [DINOv2 CLS feat ⊕ 5 concatenated CORN logits] predicting P(correct vs vet), instead of raw confidence. Adopt their `risk_coverage/auc` as the headline selective-prediction metric (RiskTolerance→Threshold protocol); wrap with our bootstrap CIs (they don't). Max-prob as baseline so the learned selector is the delta.

**Confound audit (counterfactual engines for P2.1 #2/#3):** `visinf/beyond-accuracy` BG-Gap, `bytedance/ibot` BG-challenge variants, `haofanwang/Score-CAM` EBPG. Plus `machanic/AU_R-CNN`'s presence(≥1)/strong(≥2) cut — *invert its role*: report VLM-vs-vet κ separately at the presence and strong thresholds and feed the margin into the abstention curve.

**VLM weak-labeling (structured-output plumbing):**
- `Arize-ai/phoenix` — forced tool-use: one tool whose `input_schema` has 5 enum properties (each `[0,1,2]`) + optional per-AU confidence/abstain; `tool_choice={"type":"tool","name":...}`; read `content_block.input`. Parse-failure-free enums de-risk the κ metric.
- `D-Star-AI/dsRAG` — `instructor.from_anthropic(mode=Mode.ANTHROPIC_JSON)` + Pydantic `AUScores` (5× `Literal[0,1,2]`), base64 image source, `max_retries`; log validation failures as datasheet provenance.
- `thomasnormal/fewshot` — Pydantic image-in/`Literal`-enum-out + `GreedyFewShot(max_examples=3)` to auto-select vet-anchored in-context demos; measure κ lift before freezing labels.
- `Cohere-Labs-Community/m-rewardbench` — row-paired `cohen_kappa_score(labels=[0,1,2])`, **but add `weights="quadratic"`** (their code is nominal — `labels=` only fixes the class set, does NOT make it ordinal). This bug-to-avoid is our weak-label-reliability number.
- `Lum1104/MER-Factory` `MENTION_TO_AUS` — parse free-text VLM AU mentions into our 5 feline AUs as a cheap secondary check. Do NOT adopt `AU_TO_TEXT_MAP` (human OpenFace FACS — wrong species, reintroduces a human-FACS confound).

**Face crop (zero-training preprocessing fallback only):** `cyclomon/UNSB` / `haribaskar/CatDetection-HaarCascade` — Haar `frontalcatface` → detectMultiScale → pad ~0.3 → clamp → crop. Coarse pre-crop feeding DINOv2; log detect-failure as an abstention signal; do NOT make it the landmark pipeline (Steagall trap). Fix the swapped-xml bug in UNSB if adopted.

## P2.3 Already covered / duplicate — don't re-find

- **CORN loss/decode mechanics** (`coral-pytorch`, `ludwig`, `odelia_breast_mri`, `life2vec`) — already in BUILD_PLAN §3.3/§5. Plumbing, zero novelty.
- **Frozen-DINOv2 freeze idiom** (`semantic-wm`) — decided architecture; borrow the snippet, not research.
- **`anl13/animal_papers`** — a bibliography, not code (surfaced 4×). Re-confirms MEMORY: feline-grimace exists on GitHub only as CatFLW *citations* (landmarks, NOT AUs/FGS). Stop re-grepping; cite CatFLW for alignment only.
- **Plain Cohen's κ with `labels=[0,1,2]`** (`m-rewardbench`) — we planned quadratic-weighted κ; their omission of `weights=` is the gap, not the contribution.

## P2.4 Threats to our novelty — and how we still differentiate

Honest read: **nothing on public GitHub threatens the core thesis.** Components exist in isolation; no repo assembles them on cat pain or on a learned graded grimace scorer.

| Apparent threat | Reality | How we still own it |
|---|---|---|
| "Frozen DINOv2 + ordinal head is done" (`Semi-supervised-learning`, `vtab1k-pytorch`, `odelia_breast_mri`) | True as plumbing, on breast MRI / VTAB / SSL. None grimace, none cat, none VLM-supervised, none reports calibration/abstention/confound | Engine was never the novelty (BUILD_PLAN §0.5). The wrapper is untouched |
| "Selective prediction is solved" (`reliable_vqa`, MAPIE) | True for VQA / generic binary | None on an ordinal clinical score with undertreatment≫overtreatment + bootstrap CIs + decision-curve. The composition on cat-FGS is the artifact |
| "Calibration libraries exist" (netcal, temperature_scaling) | True | They calibrate top-label softmax. Ordinal/regression calibration of a summed CORN score (ENCE/QCE/RPS) on a clinical grimace scale is unreported; prior cat-pain work reports ZERO calibration |
| "Background-challenge is known" (Madry, ibot, beyond-accuracy) | True on ImageNet-9 | No cat-pain paper proves the score isn't reading the clinic. FGS-BG-Gap (continuous shift + pain-flip, per AU) is a new metric instance |
| Steagall 2023 / Martvel | Closed data, landmark+XGBoost / video-temporal; no learned features, no calibration, no abstention, no confound counterfactual | The explicit replication traps (BUILD_PLAN §0.5 traps 1–2) |

**Differentiation rule:** not first to any single component; first to the assembled trustworthiness wrapper on a learned graded FGS scorer, every claim conditioned on a measured weak-label-reliability number.

## P2.5 Gaps still unfilled by GitHub — our strongest claims

1. **A learned graded (0–10) FGS scorer at all** — GitHub has cat *landmarks* (CatFLW) and *binary* leaderboards; zero per-AU 0/1/2→0–10 learned scorers.
2. **VLM-as-weak-labeler → per-AU quadratic κ vs vet on cat AUs** — κ mechanics exist; the measurement does not. Gate 1-B precondition + first-in-field.
3. **Ordinal calibration of a clinical grimace score** (ENCE/QCE/RPS + reliability diagram on a summed CORN distribution).
4. **A distribution-free *guaranteed* defer-to-vet band** with NPV control (undertreatment cost) — MAPIE machinery, never applied to animal-welfare pain.
5. **Per-AU causal confound audit** (FGS-BG-Gap + per-AU EBPG, stratified by capture condition).
6. **Honest datasheet of pseudo-label provenance** + frozen clip-grouped test split + released weak-labeled AU annotations.

## P2.6 Single highest-leverage next experiment

**The Gate 1-B VLM-vs-vet per-AU quadratic-κ pilot** (~120 images, ≥50 pain-positive), instrumented with three borrowed pieces at once: (a) `Arize-ai/phoenix` forced-tool-use enum schema (parse-failure-free 5-AU calls); (b) `m-rewardbench` row-paired κ **with `weights="quadratic"`**; (c) `prometheus-eval` ordinal Krippendorff α over N≥3 repeated VLM runs for a vet-free self-consistency column.

Highest leverage because it is the GO/NO-GO gate every downstream claim rides on (BUILD_PLAN §0.5, §3.4), needs no GPU/backbone (pure API + sklearn/krippendorff on M4), and produces the first-in-field weak-label-reliability number that is the spine of the wrapper. If muzzle/whiskers κ collapses, the α pre-screen and the binary-fallback insurance (novel-contribution memory) both cover us — pure information gain, no project-killing downside.