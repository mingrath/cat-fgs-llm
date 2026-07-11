# Verification — local repo contracts

Command:

```bash
uv run python .omo/ulw-research/20260630-102642/verify-local-contracts.py
```

Output:

```
{
  "backbone_support": [
    "dinov2_vits14_reg",
    "dinov2_vits14",
    "dinov3_vits16"
  ],
  "configured_backbone": "dinov2_vits14_reg",
  "point_decision_threshold": 0.39,
  "cache_feature_dim_contract": 384
}
```

Verdict: CONFIRMED — repo has DINOv3 support path, DINOv2-reg default/config, shared 0.39 threshold, and 384-d feature cache contract.
