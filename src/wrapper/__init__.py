"""THE FRAME: operating_point / decision_curve / abstention.

Operating point is a FIXED pain-recall >= 0.90 (Evangelista anchor), never
Youden-J/F1. Abstention is a one-sided 95% NPV lower bound; the word
"guaranteed" is banned. Sens/spec at 0.39 are estimated on vet-confirmed
labels ONLY (circularity firewall).
"""
