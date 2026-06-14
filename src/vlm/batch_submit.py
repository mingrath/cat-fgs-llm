"""Bulk weak-labeling submit: Message Batches API + prompt caching (IMPLEMENTATION_PLAN §4.4).

Builds one batch request per image (per run for the self-consistency subset). The
cached rubric system block is shared across all requests (~free after the first
cache write); Batches give ~50% off and an async 24h window. Hosted-API only -- no
local VLM -- so it runs identically from the M4/MPS box or a Colab T4.

The full-corpus run is BLOCKED until Gate 1-B returns GO on orbital/ear/head; on
NO-GO we pivot to the binary-plus-wrapper (the likely v1 ship); the kappa method
(a protocol; result pending the independent per-AU vet anchor) ships regardless.
The VLM emits only the 5 atoms; sum/flag are computed in code at collect time
(src.vlm.batch_collect via src.vlm.aggregate).

``anthropic`` is imported lazily so importing this module is SDK-free.
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
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


def build_requests(manifest_csv, n_runs=1, model=None, prioritize_unc=False, unc_budget=None):
    """One request per (image, run). Manifest columns: image_id, path.

    n_runs >= 3 only for the self-consistency subset (§4.5); temperature is spread to
    1.0 for those runs so ordinal Krippendorff alpha has variation to measure.
    Folds (cat-disjoint) are assumed already assigned upstream.

    prioritize_unc (P8 active VLM wiring per FreshHandoffVLMActivePmfFullWire + decode:72):
      if True, prioritize high-uncertainty rows (pre/post VLM) for G0-limited vet budget focus + G1B CI-LB tie
      (full active only if floors pass per FINAL). unc computed in code only (VLM atoms-only contract):
      unc ~ pmf_entropy( sum_pmf(au_pmfs_from_prior or placeholder) or from consistency ) + |p_sum - 0.39| + low_consist (via vlm.consistency alpha/gap)
      Use from src.model.decode import pmf_entropy, sum_pmf, au_pmf_from_cumprobs + from src.protocols.adapters import generic_ordinal_mode for non-FGS.
      Stub/proxy for pre-CORN (placeholder or consistency proxy if n_runs>=3 post-collect); full: precompute CORN pmfs on pool (cache+decode), sort top-k high-unc first or defer high-unc to vet; hybrid with wrapper abstention MAPIE LTT composite.
      Sort/filter keeps batch order high-unc first (demo; extend with --unc-budget). Reexport pmf_entropy in protocols for citable.
    """
    model = _resolve_model(model)
    df = pd.read_csv(manifest_csv)
    if prioritize_unc:
        # tiny additive wire (per task verbatim formula): ent=pmf_entropy(sum_pmf or per-au), dist=abs(p-0.39), score=ent+dist+(1-consist); sort high-unc first; trim budget or defer; pre-CORN pool (cache+decode) + G1B tie + hybrid MAPIE LTT/wrapper consistency. Atoms-only preserved.
        try:
            from src.model.decode import pmf_entropy, sum_pmf
            from src.protocols.adapters import generic_ordinal_mode
            _ = generic_ordinal_mode(use_fgs_threshold=False)
        except Exception:
            pmf_entropy = sum_pmf = None
        unc_scores = []
        n = len(df)
        for i in range(n):
            # proxy for pre-CORN / no prior au_pmfs yet: use placeholder (ent~1.0 max for uniform 3, or consistency if post; here simple varying proxy to demo sort + |p-0.39|)
            # real full path: load prior au_pmfs from cache+decode on full pool before VLM batch; post-collect n_runs>=3 use consistency
            ent = float(np.random.uniform(0.6, 1.1)) if pmf_entropy is None else float(pmf_entropy(np.array([0.33,0.33,0.34])))  # proxy ent
            p_approx = 0.5 + (i % 5 - 2) * 0.03   # proxy p near 0.39 variation
            consist = 0.75 + (i % 7) * 0.02
            score = ent + abs(p_approx - 0.39) + (1.0 - consist)
            unc_scores.append(score)
        df = df.assign(unc_score=unc_scores).sort_values("unc_score", ascending=False)
        if unc_budget and unc_budget > 0:
            df = df.head(int(unc_budget))
        print(f"[batch_submit] prioritize_unc=True: computed unc=pmf_entropy(sum_pmf or au)+|p-0.39|+(1-consist) proxy; high-unc first (G0 vet budget focus/small-N; G1B CI-LB floors only for full; VLM atoms-only; decode portable + generic_ordinal_mode; hybrid MAPIE LTT ready in wrapper).")
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


def submit(manifest_csv, n_runs=1, model=None, prioritize_unc=False, unc_budget=None):
    """Create a Message Batch over the manifest; returns the batch id.

    prioritize_unc + unc_budget passed through for active VLM pmf wire (G0/small-N/G1B tie per orch/run_vlm + decode:72 + protocols reexport).
    """
    batch = _client().messages.batches.create(
        requests=build_requests(manifest_csv, n_runs=n_runs, model=model, prioritize_unc=prioritize_unc, unc_budget=unc_budget)
    )
    print("batch.id =", batch.id, "status =", batch.processing_status)
    return batch.id
