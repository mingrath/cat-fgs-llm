"""Evaluation: kappa / confound / distributional (RPS + ClasswiseECE) / bootstrap CI.

Hosts the two portable-method headlines (kappa, confound). The kappa gate and
the confound audit are one-directional and fire on the CI lower bound. No binned
reliability diagram exists on the 11-atom 0-10 sum.

The kappa bootstraps cluster-resample by cat_id (groups=) so the gate-feeding lower
bound respects within-cat correlation. The confound protocol has three axes: image-side
bg_gap + ebpg, and the judge-side judge_bias probe (VLM-rater prompt-perturbation bias).
"""
