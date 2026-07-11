# Wave 2 — FGS official scoring / VLM / vet anchor

Worker: 019f1695-cae9-7e11-9cd5-a8deed96da4a
Axis: official FGS scoring, VLM/chatbot evidence, vet-anchor requirements.

## Key findings
- FGS uses 5 AUs: ear position, orbital tightening, muzzle tension, whiskers change, head position.
- Each AU is scored 0–2; total raw score 0–10.
- Analgesia trigger is ≥4/10 or normalized >0.39.
- 2025 chatbot paper tested ChatGPT, Claude, Gemini, Perplexity; most had poor agreement; LoA crossed the 0.39 clinical boundary, including Claude despite best retest bias.
- New detector must validate against expert-veterinarian FGS scores, including AU-level agreement, threshold safety, LoA/bias, responsiveness to analgesia, breed/subgroup robustness, image/video quality robustness.
- 2026 caveats: brachycephalic ocular-pain cats may be overestimated in image scoring; muzzle/whiskers weak; e-collars can block scoring.
- Systematic review supports FGS as one of the better-supported acute pain instruments, but this validates the instrument/human scoring, not model outputs.

## Source anchors
- Official FGS practice/scoring: https://www.felinegrimacescale.com/practice-your-skills
- Validation paper: https://www.nature.com/articles/s41598-019-55693-8
- Chatbot/VLM paper: https://www.nature.com/articles/s41598-025-27404-z ; PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC12689790/
- Brachycephalic/ocular pain caveat: https://pmc.ncbi.nlm.nih.gov/articles/PMC12764747/
- Measurement review: https://pmc.ncbi.nlm.nih.gov/articles/PMC12862644/

## EXPAND markers verbatim
- subgroup performance table
- ROC / sensitivity / specificity details for the cutoff
- prospective clinical validation design for a new detector

## CLAIMS markers verbatim
- FGS = **5 AUs**, each **0–2**, total **0–10**.
- Analgesia cutoff = **≥4/10** or **>0.39 normalized**.
- Chatbots/VLMs tested in 2025 were **not reliable enough** as sole FGS scorers.
- New detectors must pass **expert-vet anchor + threshold-safety + subgroup robustness** before clinical use.
