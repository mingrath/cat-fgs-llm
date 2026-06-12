"""Single-image forced-tool call (IMPLEMENTATION_PLAN §4.1).

Used by the Gate 1-B pilot and the Gate-4-adjacent smoke test. One forced tool_use
block -> validated FGSResult. The 0-10 sum and the 0.39 flag are NOT computed here:
they live in src.vlm.aggregate, the single threshold definition.

``anthropic`` is imported lazily so importing this module (and src.vlm at large) is
network-/SDK-free; only the actual API call requires the dependency + API key.
Model id is read from configs/vlm_fgs.yaml (model_id), falling back to the VLM_MODEL
env var. Local-only work is the Pillow re-encode -- no CUDA, no local VLM.
"""

import base64
import io
import os
from functools import lru_cache
from pathlib import Path

import yaml
from PIL import Image

from src.vlm.rubric import SYSTEM_BLOCKS  # cache-controlled rubric block(s), §4.2
from src.vlm.schema import FGS_TOOL, TOOL_CHOICE, FGSResult

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "vlm_fgs.yaml"
_MAX_TOKENS = 1200


@lru_cache(maxsize=1)
def _config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_model(model=None):
    if model:
        return model
    return os.environ.get("VLM_MODEL") or _config()["model_id"]


@lru_cache(maxsize=1)
def _client():
    # Lazy import: keep `import src.vlm.call` SDK-free until a call is actually made.
    from anthropic import Anthropic

    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def encode_face(path, long_edge=896):
    """Base64-encode a face crop at a capped long edge (crops are small; cap for cost)."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((long_edge, long_edge))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def score_face(path, model=None, temperature=0.0, seed_tag=""):
    """Single forced-tool call -> validated FGSResult (raises on contract violation)."""
    model = _resolve_model(model)
    msg = _client().messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=temperature,
        system=SYSTEM_BLOCKS,  # cached rubric block(s)
        tools=[FGS_TOOL],
        tool_choice=TOOL_CHOICE,  # forced; exactly one tool_use block
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encode_face(path),
                        },
                    },
                    {"type": "text", "text": f"Score this cat face. tag={seed_tag}"},
                ],
            }
        ],
    )
    block = next(b for b in msg.content if b.type == "tool_use")
    return FGSResult(**block.input)  # raises on contract violation -> log + abstain-route
