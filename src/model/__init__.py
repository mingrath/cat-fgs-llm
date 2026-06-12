"""ENGINE — conceded plumbing, never claimed novel.

Frozen DINOv2 ViT-S/14 (384-d) + 5 per-AU CORN heads. Outputs feed src.wrapper
(the frame) and src.eval (the two portable methods); they never produce a
claimed-novel artifact. Distributional CORN decode is the default; hard argmax
decode is used ONLY for the 0.39 point decision.
"""
