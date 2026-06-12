"""Forced-tool-use enum schema + Pydantic mirror (IMPLEMENTATION_PLAN §4.1).

The VLM emits ONLY the 5 Feline Grimace Scale action-unit atoms, each in {0,1,2},
via a single forced tool call (Arize-ai/phoenix structured-output pattern). The
0-10 sum and the >=0.39 analgesia flag are computed IN CODE (src.vlm.aggregate),
NEVER by the VLM: there is NO ``sum``, ``decision``, or ``pain`` field in this
schema by construction.

This is supervision-layer plumbing only. The structured-output call is borrowed
plumbing; the portable method is the downstream per-AU VLM-vs-vet quadratic kappa
(src.eval.kappa / §4 pilot), not this file.

Claude strict structured outputs strip numeric min/max (FACTCHECK C48/C49), so the
ordinal levels are pinned with ``enum: [0, 1, 2]`` -- the only reliable way to fix
the three levels. Rationale is emitted BEFORE the score (property order is the
generation order under tool use) so the model reasons then commits.
"""

from typing import Literal

from pydantic import BaseModel

from src.constants import AU_ORDER

# Evangelista 5-AU order — canonical declaration in src.constants.AU_ORDER;
# mirrored in configs/vlm_fgs.yaml (au_order) and reused by aggregate.py /
# batch_collect.py / consistency.py so the per-AU column build never keys on
# dict iteration order.
AU_NAMES = list(AU_ORDER)


def _au_property():
    # rationale BEFORE score; confidence + abstain alongside. Order matters:
    # JSON-schema property order is the generation order under tool use.
    return {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": "1-2 sentences citing the visible feature, BEFORE scoring.",
            },
            "score": {"type": "integer", "enum": [0, 1, 2]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "abstain": {
                "type": "boolean",
                "description": "true if occluded/blurred/out-of-frame/non-frontal for THIS AU.",
            },
        },
        "required": ["rationale", "score", "confidence", "abstain"],
        "additionalProperties": False,
    }


FGS_TOOL = {
    "name": "record_fgs_action_units",
    "description": (
        "Record the Feline Grimace Scale action-unit scores for one cat face. "
        "Score EACH of the 5 AUs independently on the 0/1/2 ordinal scale. "
        "Do NOT compute a total or any pain decision."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            **{au: _au_property() for au in AU_NAMES},
            "image_quality": {
                "type": "string",
                "enum": ["frontal_clear", "partial", "unusable"],
            },
        },
        "required": AU_NAMES + ["image_quality"],
        "additionalProperties": False,
    },
}

# Forced; exactly one tool_use block (disable_parallel_tool_use).
TOOL_CHOICE = {
    "type": "tool",
    "name": "record_fgs_action_units",
    "disable_parallel_tool_use": True,
}


# Pydantic mirror -- validates tool input in-code; logs validation failures as
# datasheet provenance (dsRAG pattern). Parse-failure-free by design under the
# forced-enum schema, but we still assert the contract.
class AU(BaseModel):
    rationale: str
    score: Literal[0, 1, 2]
    confidence: Literal["low", "medium", "high"]
    abstain: bool


class FGSResult(BaseModel):
    ear: AU
    orbital: AU
    muzzle: AU
    whiskers: AU
    head: AU
    image_quality: Literal["frontal_clear", "partial", "unusable"]
