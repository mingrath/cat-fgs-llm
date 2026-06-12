"""Bulk weak-labeling submit: Message Batches API + prompt caching (IMPLEMENTATION_PLAN §4.4).

Builds one batch request per image (per run for the self-consistency subset). The
cached rubric system block is shared across all requests (~free after the first
cache write); Batches give ~50% off and an async 24h window. Hosted-API only -- no
local VLM -- so it runs identically from the M4/MPS box or a Colab T4.

The full-corpus run is BLOCKED until Gate 1-B returns GO on orbital/ear/head; on
NO-GO we pivot to the binary-plus-wrapper fallback (the kappa numbers ship as a
method finding regardless). The VLM emits only the 5 atoms; sum/flag are computed
in code at collect time (src.vlm.batch_collect via src.vlm.aggregate).

``anthropic`` is imported lazily so importing this module is SDK-free.
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from src.vlm.call import encode_face
from src.vlm.rubric import SYSTEM_BLOCKS
from src.vlm.schema import FGS_TOOL, TOOL_CHOICE

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
    from anthropic import Anthropic

    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def build_requests(manifest_csv, n_runs=1, model=None):
    """One request per (image, run). Manifest columns: image_id, path.

    n_runs >= 3 only for the self-consistency subset (§4.5); temperature is spread to
    1.0 for those runs so ordinal Krippendorff alpha has variation to measure.
    Folds (cat-disjoint) are assumed already assigned upstream.
    """
    model = _resolve_model(model)
    df = pd.read_csv(manifest_csv)
    reqs = []
    for _, r in df.iterrows():
        for k in range(n_runs):
            reqs.append(
                {
                    "custom_id": f"{r.image_id}__run{k}",
                    "params": {
                        "model": model,
                        "max_tokens": _MAX_TOKENS,
                        "temperature": 0.0 if n_runs == 1 else 1.0,  # spread for alpha
                        "system": SYSTEM_BLOCKS,  # cache_control rides here
                        "tools": [FGS_TOOL],
                        "tool_choice": TOOL_CHOICE,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": encode_face(r.path),
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": f"Score this cat face. tag={r.image_id}__run{k}",
                                    },
                                ],
                            }
                        ],
                    },
                }
            )
    return reqs


def submit(manifest_csv, n_runs=1, model=None):
    """Create a Message Batch over the manifest; returns the batch id."""
    batch = _client().messages.batches.create(
        requests=build_requests(manifest_csv, n_runs=n_runs, model=model)
    )
    print("batch.id =", batch.id, "status =", batch.processing_status)
    return batch.id
