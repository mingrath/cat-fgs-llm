"""Margin EXPANSION of the detector face box (IMPLEMENTATION_PLAN §3.1).

FGS reads ear tips (ear AU) and whisker splay, both of which sit OUTSIDE a tight
face box. The forked Roboflow boxes are tight head crops, so we EXPAND before
cropping; we never SHRINK (a shrunk box that drops ear tips silently zeros the
ear AU). The expansion is ASYMMETRIC: 1.4x on height for ear-tip room.

A box already at the frame edge cannot recover the ear/whisker by expansion;
that `clipped` flag is logged into the manifest as one abstention signal
(route-to-vet) — a clipped grimace AU cannot be scored honestly (§3.1).

This stage is preprocessing for the conceded DINOv2+CORN engine; it is not a
claimed-novel artifact. It exists to stop us measuring crop quality and
mistaking it for calibration (§3).
"""

from __future__ import annotations

EXPAND = 0.135            # 13.5%, midpoint of the 12-15% band (BUILD_PLAN line 125)
HEIGHT_EXPAND_MULT = 1.4  # asymmetric: extra top room for ear tips


def expand_box(cx, cy, w, h, img_w, img_h, expand=EXPAND, height_mult=HEIGHT_EXPAND_MULT):
    """Expand a normalized YOLO box and return clamped pixel xyxy + clip flag.

    Parameters
    ----------
    cx, cy, w, h : float
        Normalized YOLO box (center x/y, width, height), each in [0, 1].
    img_w, img_h : int
        Source image pixel dimensions.
    expand : float
        Fractional margin added to width (and base for height).
    height_mult : float
        Top margin multiplier: the TOP edge gets `expand * height_mult` while the
        bottom gets plain `expand` — asymmetric top bias for ear tips.

    Returns
    -------
    (x1, y1, x2, y2) : tuple[int, int, int, int]
        Clamped pixel xyxy corners (rounded to int).
    clipped : bool
        True if the *unclamped* expanded box exceeded any frame edge — the
        expansion could not fully recover the ear/whisker margin, so this is an
        abstention signal routed to vet downstream (§3.1, §3.5).
    """
    # Width: `expand` split evenly (whiskers are ~symmetric horizontally).
    # Height: ASYMMETRIC — ear tips sit above the head box, so the top edge gets
    # `expand * height_mult` and the bottom (chin) only plain `expand`.
    # NEVER shrink: every margin is added, not subtracted.
    half_w = w / 2 + w * expand / 2
    x1 = (cx - half_w) * img_w
    x2 = (cx + half_w) * img_w
    y1 = (cy - h / 2 - h * expand * height_mult) * img_h   # extra top room for ears
    y2 = (cy + h / 2 + h * expand) * img_h                  # plain bottom margin

    # clip flag: box already at frame edge, so expansion can't recover the
    # ear/whisker -> logged into the manifest as a route-to-vet reason.
    clipped = (x1 < 0) or (y1 < 0) or (x2 > img_w) or (y2 > img_h)

    # clamp to frame
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(img_w), x2)
    y2 = min(float(img_h), y2)

    return (round(x1), round(y1), round(x2), round(y2)), clipped
