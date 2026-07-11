# Verification — targeted local tests

Command:

```bash
uv run pytest tests/test_backbone_variant.py tests/test_threshold_single_source.py tests/test_gate4_decode.py -q
```

Output:

```text
............                                                             [100%]
12 passed in 1.86s
```

Verdict: CONFIRMED — local backbone variant contract, shared 0.39 threshold, and CORN decode/point decision tests pass.
