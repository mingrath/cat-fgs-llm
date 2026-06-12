"""Duplicate + near-duplicate audit (IMPLEMENTATION_PLAN §1.1 step 5 / §2.2).

Two distinct jobs, kept separate on purpose:

1. ``collapse_exact_dups`` -- pixel-identical (md5) dedup. The export carries
   493 exact-duplicate filenames (2040 records / ~1547 distinct). Collapse them
   BEFORE splitting so a duplicate image cannot straddle train and test.

2. ``assert_no_neardup_straddles_split`` -- ``imagehash`` Hamming-<=10 union-find
   used ONLY as a near-duplicate detector, so a near-dup pair cannot land on
   opposite sides of a split. This is explicitly NOT cat re-ID: pHash/CLIP carry
   no individual-identity signal here (see Gate 1). The per-cat merge is the
   group key; this is a leakage guard, nothing more.

Data plumbing for the binary spine; makes no validated claim.
"""

from __future__ import annotations

import collections
import hashlib
import pathlib
from typing import Iterable

# imagehash is optional at import time (the near-dup guard is the only consumer).
try:  # pragma: no cover - exercised only when a real export is present
    import imagehash
    from PIL import Image
except Exception:  # pragma: no cover
    imagehash = None
    Image = None

# Default near-duplicate Hamming threshold (§2.2 "imagehash Hamming <=10").
NEARDUP_HAMMING = 10


def md5_of_file(path: pathlib.Path) -> str:
    """md5 of the raw file bytes (pixel-identity for our export)."""
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def collapse_exact_dups(root: str | pathlib.Path, pattern: str = "*.jpg") -> dict[str, list[str]]:
    """Map each md5 -> list of paths sharing it (the exact-dup collapse map).

    Returns the full hash->paths map so the caller can pick one canonical path
    per hash and log the dup map to ``provenance/dup_map.json``. The distinct
    pixel-hash count should land near 1547; the caller asserts that floor.
    """
    seen: dict[str, list[str]] = collections.defaultdict(list)
    for p in sorted(pathlib.Path(root).rglob(pattern)):
        seen[md5_of_file(p)].append(str(p))
    return dict(seen)


def canonical_paths(dup_map: dict[str, list[str]]) -> list[str]:
    """One canonical (lexicographically first) path per distinct pixel-hash."""
    return [sorted(paths)[0] for paths in dup_map.values()]


class _UnionFind:
    """Minimal union-find over hashable items."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict:
        out: dict = collections.defaultdict(set)
        for x in list(self.parent):
            out[self.find(x)].add(x)
        return dict(out)


def neardup_components(
    paths: Iterable[str | pathlib.Path],
    hamming: int = NEARDUP_HAMMING,
) -> dict:
    """Union-find over perceptual-hash near-dups; returns root -> component set.

    Pairs with pHash Hamming distance <= ``hamming`` are unioned. Requires
    ``imagehash`` + Pillow (only needed when a real export is present).
    """
    if imagehash is None:
        raise ImportError("imagehash + Pillow required for the near-dup detector")
    items = [str(p) for p in paths]
    phash = {p: imagehash.phash(Image.open(p)) for p in items}
    uf = _UnionFind()
    for p in items:
        uf.find(p)  # ensure singletons register
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if (phash[items[i]] - phash[items[j]]) <= hamming:
                uf.union(items[i], items[j])
    return uf.groups()


def assert_no_neardup_straddles_split(
    split_of: dict[str, str],
    components: dict,
) -> None:
    """Assert no near-dup component spans more than one split.

    ``split_of`` maps a path/image_id -> its split name (e.g. "train"/"val");
    ``components`` is the output of ``neardup_components`` keyed by paths. Raises
    ``AssertionError`` naming the offending component if a near-dup pair
    straddles a split. NOT a re-ID check -- a leakage guard only.

    ``split_of`` MUST cover every component member: a key-scheme mismatch (paths vs
    image_ids) would otherwise make every membership lookup miss and the guard pass
    vacuously, so coverage is asserted FIRST.
    """
    all_members = {m for members in components.values() for m in members}
    uncovered = sorted(m for m in all_members if m not in split_of)
    assert not uncovered, (
        "near-dup leakage guard cannot run: split_of is missing "
        f"{len(uncovered)} component members (key-scheme mismatch?): {uncovered[:5]}"
    )
    for root, members in components.items():
        splits = {split_of[m] for m in members}
        assert len(splits) <= 1, (
            f"near-dup component {root!r} straddles splits {sorted(splits)}: "
            f"{sorted(members)}"
        )
