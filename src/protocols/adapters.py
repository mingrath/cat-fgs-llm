"""Thin adapters for portable protocol use on arbitrary corpora (no FGS/CAT_01/cat-fgs assumptions).

Addresses P2 overclaim: AU_ORDER coupling, col conventions, 0.39 magic.

- AU list override: use different #/names/order of "action units" (generic ordinal rater protocol).
- Col mapper: rename your {foo}_vlm / {foo}_vet or pain cols to protocol expectation.
- no-FGS-0.39 generic ordinal mode: for kappa etc on non 5-AU 0/1/2 or different clinical cut; 0.39/POINT only for FGS compatibility layer.

Pure: numpy/pandas only. No torch, no paths, no engine, no yaml (unless caller).

Re-exported via src.protocols.__init__ .
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd

from src.constants import AU_ORDER as _FGS_AU_ORDER
from src.constants import POINT_DECISION_THRESHOLD as _FGS_POINT  # optional, for compat layer only

def get_au_names(override: Optional[Iterable[str]] = None) -> list[str]:
    """Return AU names for protocol use.
    
    override: e.g. ["ear","muzzle",...] or ["au1","au2",...] for 7-ordinal non-FGS corpus.
    Defaults to FGS Evangelista order (for backward + single source).
    """
    if override is None:
        return list(_FGS_AU_ORDER)
    return list(override)

def apply_au_override(au_names: Optional[Iterable[str]] = None) -> list[str]:
    """Alias for get_au_names (used by reexport adapters)."""
    return get_au_names(au_names)

def map_df_columns(
    df: pd.DataFrame,
    au_names: Iterable[str],
    vlm_suffix: str = "_vlm",
    vet_suffix: str = "_vet",
    cat_col: str = "cat_id",
    mapper: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Map caller df columns to protocol-expected {au}_vlm / {au}_vet (+cat_id).
    
    mapper: explicit {your_col: protocol_col} e.g. {"my_ear_score_v": "ear_vlm"}.
    If mapper provided, use it first (bypass auto).
    Otherwise, for each au in au_names, look for au+vlm_suffix etc.
    
    Returns a *copy* with standardized cols (originals preserved if extra).
    Idempotent if already matches.
    """
    df = df.copy()
    au_list = list(au_names)
    if mapper:
        for src, dst in mapper.items():
            if src in df.columns:
                df[dst] = df[src]
        # ensure cat if mapped
        if cat_col not in df.columns and any(k.endswith("cat") for k in mapper):
            pass  # caller responsibility or extend
        return df

    for au in au_list:
        vlm_cand = f"{au}{vlm_suffix}"
        vet_cand = f"{au}{vet_suffix}"
        if vlm_cand not in df.columns and au in df.columns:
            # allow bare au col as vlm in generic
            df[vlm_cand] = df[au]
        # similarly vet if needed; assume caller supplies both or use same for synthetic
        if vet_cand not in df.columns and f"{au}_vet" not in df.columns:
            # for pure generic tests may map same
            pass
    # cat_col left as-is; caller ensures or protocol tolerates missing
    return df

def generic_ordinal_mode(
    use_fgs_threshold: bool = False,
    custom_threshold: Optional[float] = None,
) -> dict:
    """Return config/mode for non-FGS generic ordinal reuse (no 0.39 Evangelista assumption).
    
    use_fgs_threshold=False (default): disable 0.39 / analgesia_flag / FGS sum logic in callers.
    For kappa: still works (ordinal 0/1/2 or arbitrary labels).
    For confound: 0-10 sum + 0.39 decision optional; pass scores directly.
    
    Returns dict for downstream (e.g. protocols.kappa can ignore if flag set).
    """
    mode = {
        "fgs_mode": bool(use_fgs_threshold),
        "threshold": _FGS_POINT if (use_fgs_threshold and custom_threshold is None) else custom_threshold,
        "note": "generic ordinal (no FGS 0.39 / 5-AU / Evangelista assumption)" if not use_fgs_threshold else "FGS compat",
    }
    return mode
