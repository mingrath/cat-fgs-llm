# KAGGLE SWEEP BRIEFING — Does Kaggle Close the Graded-FGS Gap?

## 1. Verdict — Did Kaggle close the gap?

**No. Flatly no.** After six search strategies (topic, emotion/health, cross-species, face/landmark, competitions, kernel-trace) plus five targeted gap-fills (author/codename surnames, Roboflow-pain hunt, deep file-verification of the two best assets, non-English PT/ES/FR terms), **Kaggle contains zero datasets with graded Feline Grimace Scale labels, zero per-action-unit (CatFACS) labels, and not a single new binary cat-pain example beyond the ~260 we already hold.** Every "FGS"/"grimace"/"feline pain" token returned either empty or off-topic noise. The published academic FGS sets (Steagall 1,188-img, Evangelista 110-img, Finka/TiHo) are **not** mirrored on Kaggle under topic, author surname, or paper codename. No cross-species grimace corpus (mouse MGS, rat RGS, horse HGS, sheep SPFES, rabbit) exists on Kaggle either — the only graded animal-grimace set anywhere remains the **horse-grimace set on HF**, not Kaggle.

What Kaggle *does* add is incremental **preprocessing and unlabeled-pool** value: face-detection crops, region masks, and large permissive cat-face pools for SSL pretraining. None of it touches the label gap.

## 2. What Kaggle offers (ranked, grouped by usefulness)

NEW = surfaced this sweep; KNOWN = already in our docs (catflw, crawford).

### (a) Real pain / FGS / per-AU labels — *the ideal*
**EMPTY. No dataset in this category exists on Kaggle.** This is the decisive finding.

### (b) Preprocessing assets (detect → crop → align → AU-region ROI)
| ref | title | label type | pain/FGS? | size | license | value-to-us | status |
|---|---|---|---|---|---|---|---|
| `aleksandrdremov/cat-faces-detection` | Cat Faces Detection | face/head bbox, COCO + info.csv | No | ~7.1 GB / ~50K | **CC-BY-SA-4.0** | **HIGH** — strongest crop/detector pretrain pool; usability 1.0. Labels YOLOv8-autogen + partly hand-tuned (filter by info.csv score) | **NEW** |
| `petermirzoyan/cat-face-parts` | Cat face parts | region masks `.npy` | No | ~266 MB / 5,652 jpg + 16,622 masks | CC-BY-NC-SA-4.0 | **LOW-MED** — verified taxonomy is **Eye + Nose ONLY** (card's "ear" claim is false). Covers ~2/5 AUs loosely (eye≈orbital, nose≈muzzle); no L/R split; NC | **NEW** |
| `bloodaxe/animal-pose-dataset` | Animal Pose | 20 body+face keypoints | No | ~368 MB | **unknown / NO-GO** | **AVOID** — license unresolvable (Kaggle="unknown", source GitHub has no LICENSE, mixed scraped provenance); only eyes/nose/earbases for cats | **NEW** |
| `vladsmirno/petfacedetection` | Dog+Cat Face Detection | YOLO bbox | No | ~3.9 GB | unspecified | LOW — alt cropper, less cat-focused, license unclear | **NEW** |
| `gpreda/cat-face-detection` | Haar Cascades | OpenCV XML (no data) | No | ~200 KB | OpenCV | MINIMAL — zero-dep fallback cropper only | **NEW** |

### (c) Unlabeled cat-face pools (SSL pretraining / augmentation)
| ref | title | label type | pain/FGS? | size | license | value-to-us | status |
|---|---|---|---|---|---|---|---|
| `dseidli/lcwlabeled-cats-in-the-wild` | LCW | identity (~140k cats) | No | ~22 GB | **Apache-2.0** | MED — largest permissive SSL pool; identity only | **NEW** |
| `timost1234/cat-individuals` | Cat Individuals | identity (518 cats) | No | ~11.2 GB | **CC BY 4.0** | MED — shelter cats (some stressed), permissive | **NEW** |
| `crawford/cat-dataset` | Cat Dataset (Zhang 2008) | 9 landmarks | No | ~4.3 GB | see card | LOW — already owned; sparse landmarks | **KNOWN** |

### (d) Cross-species / affect proxies — *weak auxiliary, not pain*
| ref | title | label type | pain/FGS? | size | license | value-to-us | status |
|---|---|---|---|---|---|---|---|
| `georgemartvel/catflw` | CatFLW | 48 facial landmarks | No | ~1.4 GB | **CC BY-NC** | MED — our gold landmark/align asset (Gate-5 NME); NC | **KNOWN** |
| `georgemartvel/dogflw` | DogFLW | 46 dog landmarks | No | ~1.4 GB | CC BY-NC | LOW — cross-species landmark transfer; NC | NEW |
| `anshtanwar/pets-facial-expression-dataset` | Pet Facial Expression | coarse emotion, multi-species | No | ~40 MB | CC BY 4.0 | LOW — subjective emotion ≠ pain | NEW |
| `mohammadabdulghafara/cat-thermal-and-rgb...` | Cat Thermal+RGB | health-status (Healthy) + thermal | No | ~1.1 GB | CC BY 4.0 | LOW — thermal/health proxy, no AU | NEW |
| `trendcart/*`, `nguyenvunhuhuynh/*`, `meenalsaini/*` | Cat Emotion sets | emotion folders | No | varies | mostly unconfirmed | LOW — behavioral affect, not pain | NEW |
| `diemhuongnt12/cat-skin-roboflow` | Cat skin disease | dermatology bbox | No | ~9 MB | unspecified | NONE for FGS | NEW |
| `xuancd/sick-cat`, `smadive/pet-disease-images` | Disease | diagnosis | No | small | unconfirmed | NONE for FGS | NEW |
| `steubk/catmeows` | CatMeows | **audio** emotion | No | 13 MB | — | NONE (audio) | NEW |

## 3. What's genuinely worth downloading

Download exactly **one** thing for new value, plus the one known anchor we still need:

```bash
# WORTH IT — best new preprocessing asset: 50K cat-face/head boxes, COCO, permissive (CC-BY-SA-4.0).
# Use to train/validate a face-crop detector before FGS scoring. Filter boxes by info.csv origin/score.
kaggle datasets download -d aleksandrdremov/cat-faces-detection

# STILL NEEDED (known anchor, already in HF_SOLUTION) — gold landmarks for align/crop + Gate-5 NME audit.
kaggle datasets download -d georgemartvel/catflw
```

**Optional, only if a specific need arises:**
```bash
# Large permissive unlabeled pool IF you actually do SSL pretraining (22 GB — don't pull speculatively).
kaggle datasets download -d dseidli/lcwlabeled-cats-in-the-wild
```

**Why these and nothing else:** `cat-faces-detection` is the single asset that materially improves our detect→crop stage (50K permissive boxes vs. our sparse alternatives) and is the only NEW download with a clean commercial-grade license. `catflw` is already mandated by HF_SOLUTION for the Gate-5 alignment audit. Everything in group (d) is coarse affect/diagnosis that would only inject confound if used as a label source — do **not** download for supervision.

**Explicitly skip:** `petermirzoyan/cat-face-parts` (verified Eye+Nose only — 3 of 5 AUs absent, NC, card is wrong); `bloodaxe/animal-pose-dataset` (license NO-GO); all emotion/disease/audio sets (no pain signal, license-murky).

## 4. Updated data picture — does Kaggle change the conclusion?

**No. The conclusion in `HF_SOLUTION.md` and `DATA_DECISION.md` stands unchanged and is now triply confirmed (HF=0, Roboflow swept, Kaggle=0):** graded per-AU FGS labels **do not exist on any open platform** and must be **manufactured via the VLM-weak-label (Claude Opus 4.8, 5-AU enum 0/1/2) → cleanlab/boundary triage → vet review → frozen DINOv2 + 5 CORN heads** spine, run in parallel with request-only academic acquisition (Steagall/Evangelista/Zamansky).

Specifically, Kaggle:
- **Does NOT** supply the 5-AU 0/1/2 labels, the 0–10 sum, or the >0.39 analgesia ground truth. The spine still owns label creation.
- **Does NOT** add binary pain examples — the ~260 pain / 1,819 no_pain Roboflow base remains the ceiling, so `DATA_DECISION.md §1`'s confound caveat (Flickr breed-morphology pseudo-labels, not nociception) is unaltered.
- **Does NOT** touch the structural blind spot: **whiskers + head AUs have no cross-species or Kaggle source** (the horse set, HF-only, covers just 3/5). These two AUs remain acquirable only via real gold (Steagall) or owned vet-clinic data.
- **DOES** marginally strengthen the *preprocessing* leg HF_SOLUTION already specified: `aleksandrdremov/cat-faces-detection` is a better crop-detector pool than the Haar cascade / sparse-landmark options, and CatFLW is re-confirmed as the alignment anchor. This is a quality bump to detect→crop→align, not a change to the modeling or labeling strategy.

**Doc updates to record:**
- `DATA_DECISION.md §4–5.5`: add the Kaggle verdict — "exhaustively swept (11 strategies); zero graded FGS / per-AU / new binary pain; only preprocessing + unlabeled-pool value. Stop sweeping Kaggle." Add `aleksandrdremov/cat-faces-detection` (CC-BY-SA-4.0, 50K boxes) as a detect/crop asset alongside CatFLW.
- `HF_SOLUTION.md §5` command block: append the two `kaggle datasets download` lines above; mark `petermirzoyan/cat-face-parts` and `bloodaxe/animal-pose-dataset` as **rejected** (Eye/Nose-only; license NO-GO).
- Reaffirm: **Kaggle is a dead end for FGS labels — the spine and the Steagall email remain the only routes to ground truth.**

## 5. Open risks / licenses

- **License ceiling unchanged and tight.** `catflw`/`dogflw` are **CC BY-NC** (research/portfolio OK, commercial fatal). `cat-face-parts` is CC-BY-NC-SA. Only the genuinely useful new asset (`aleksandrdremov/cat-faces-detection`, **CC-BY-SA-4.0**), plus `LCW` (Apache-2.0) and `cat-individuals` (CC BY 4.0), are commercially permissive — and the share-alike clause on CC-BY-SA propagates to derivatives.
- **`bloodaxe/animal-pose-dataset` — legal NO-GO.** License "unknown" on Kaggle, no LICENSE at source, mixed scraped provenance (PASCAL VOC2011 / Animals-10 / web). Do not redistribute or build derivatives.
- **Card-vs-reality mismatch.** `petermirzoyan/cat-face-parts` card claims "ear and nose masks"; full enumeration of all 16,622 `.npy` shows **Eye + Nose only** — no Ear/Muzzle/Whisker/Head. Trust file listings, not cards. (General Kaggle hazard: dataset cards are JS-rendered; `WebFetch` returns only the title — always verify via `kaggle datasets metadata` / `files`.)
- **Auto-generated labels need filtering.** `cat-faces-detection` boxes are YOLOv8-autogenerated then partly hand-tuned; gate by `info.csv` origin/confidence before trusting any single box for evaluation.
- **Affect/emotion sets are a confound trap.** Using any "Angry/emotion/sick" Kaggle set as a pain proxy would inject breed-morphology/context bias exactly as `DATA_DECISION.md §1` warns. Quarantine them out of supervision and validation.
- **Compute/storage:** LCW (22 GB) and `cat-individuals` (11 GB) are large — pull only if SSL pretraining is actually scheduled, not speculatively, given local Apple-M4/MPS + Colab-T4 constraints.

**Bottom line:** Stop sweeping Kaggle for FGS data — confirmed absent. Pull `aleksandrdremov/cat-faces-detection` + `catflw`, fold them into the detect→crop→align stage, and keep the VLM-weak-label + vet-review spine (with the parallel Steagall request) as the sole path to graded ground truth.