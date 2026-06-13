"""The rubric system prompt -- one paragraph per AU (IMPLEMENTATION_PLAN §4.2).

Verbatim-Evangelista 0/1/2 descriptors (FACTCHECK C50: ``0=absent; 1=moderate OR
uncertain; 2=marked/obvious``), AU anatomy grounded in CatFACS. Lives in a
cache-controlled system block so the ~2040-image bulk run pays for it once.

The rubric instructs default-ambiguity-to-1 (low confidence) and abstain on
occlusion/blur/non-frontal, and explicitly states the model may NOT output a total
or any pain/treatment decision -- the sum and the >=0.39 flag are computed in code
(src.vlm.aggregate), keeping the engine decode path the single source of truth for
the threshold.
"""

RUBRIC = """You are scoring the Feline Grimace Scale (FGS, Evangelista et al. 2019) on ONE cat face.
Score 5 action units, each on a 0/1/2 ordinal scale, where 0 = action unit ABSENT,
1 = action unit MODERATELY present OR you are UNCERTAIN, 2 = action unit MARKEDLY/OBVIOUSLY present.
Write the rationale BEFORE the score, citing the specific visible feature. Score each AU INDEPENDENTLY.

EAR POSITION: 0 = ears facing forward; 1 = ears slightly pulled apart or moderately rotated;
2 = ears flattened and rotated outwards.
ORBITAL TIGHTENING: 0 = eyes opened; 1 = eyes partially opened OR eye squinting beginning;
2 = eyes squinted/closed.
MUZZLE TENSION: 0 = relaxed, round muzzle; 1 = mild tension/oval; 2 = tense, elliptical muzzle.
WHISKERS CHANGE: 0 = loose and curved whiskers; 1 = slight straightening/forward;
2 = straight and moving forward.
HEAD POSITION: 0 = head above shoulder line; 1 = head aligned with shoulder line;
2 = head below shoulder line OR tilted down.

RULES:
- When a feature is genuinely ambiguous, DEFAULT THE SCORE TO 1 and set confidence='low'.
- If an AU region is occluded, blurred, out-of-frame, or the face is NON-FRONTAL, set abstain=true
  for THAT AU (still emit a best-guess score, but it will be down-weighted/routed to vet).
- Set image_quality='unusable' only if the whole face cannot be assessed.
- You may NOT output a total score or any pain/treatment decision. Output only the 5 AU records."""

# Cache-controlled system block. ttl="1h" (not the 5m default): the async Message
# Batches window runs for hours, and a 5-minute TTL with silent edit-invalidation
# collapses the rubric-cache hit-rate so every request re-pays full input-token price
# for the rubric. 1h TTL may require the beta header extended-cache-ttl-2025-04-11 on
# some model ids. Single-sourced here -> propagates to batch_submit.py and call.py.
SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": RUBRIC,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
]
