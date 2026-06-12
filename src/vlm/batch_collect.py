"""Bulk weak-labeling collect: poll + stream results to parquet (IMPLEMENTATION_PLAN §4.4).

Polls the batch to the SDK terminal state (``ended``), then streams the per-image
results, parses the single forced tool_use block, validates against FGSResult
(schema failures are logged as datasheet provenance, never silently dropped), and
writes per-AU score/abstain columns plus the in-code fgs_sum / analgesia_flag /
any_abstain (from src.vlm.aggregate -- the single threshold definition).

The VLM emits only the 5 atoms; the sum and the 0.39 flag are computed HERE in code.

``AU_NAMES`` is imported from src.vlm.schema so the per-AU column build is keyed on
the fixed 5-AU order, not on dict iteration. ``anthropic`` is imported lazily.
"""

import os
import time
from functools import lru_cache

import pandas as pd

from src.vlm.aggregate import analgesia_flag, any_abstain, fgs_sum
from src.vlm.schema import AU_NAMES, FGSResult


@lru_cache(maxsize=1)
def _client():
    from anthropic import Anthropic

    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def wait_and_collect(batch_id, out_parquet, poll_s=60):
    """Block until the batch ends, then write all results to ``out_parquet``."""
    client = _client()
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":  # SDK terminal state is "ended"
            break
        print(b.request_counts)
        time.sleep(poll_s)

    rows = []
    for entry in client.messages.batches.results(batch_id):  # streams .jsonl by custom_id
        cid = entry.custom_id
        if entry.result.type != "succeeded":
            rows.append({"custom_id": cid, "error": entry.result.type})
            continue
        msg = entry.result.message
        tu = next((c for c in msg.content if c.type == "tool_use"), None)
        if tu is None:
            rows.append({"custom_id": cid, "error": "no_tool_use"})
            continue
        try:
            res = FGSResult(**tu.input)
        except Exception as e:  # noqa: BLE001 -- record provenance, do not drop
            rows.append({"custom_id": cid, "error": f"schema:{e}"})
            continue
        d = res.model_dump()
        s = fgs_sum(d)
        rows.append(
            {
                "custom_id": cid,
                **{f"{au}_score": d[au]["score"] for au in AU_NAMES},
                **{f"{au}_abstain": d[au]["abstain"] for au in AU_NAMES},
                "image_quality": d["image_quality"],
                "fgs_sum": s,
                "analgesia_flag": analgesia_flag(s),
                "any_abstain": any_abstain(d),
            }
        )
    pd.DataFrame(rows).to_parquet(out_parquet)
    return out_parquet
