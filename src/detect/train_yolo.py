"""Phase A — YOLOv11s RUNNER-UP runner (IMPLEMENTATION_PLAN §2.5).

Thin wrapper that builds/prints the `yolo detect train` command with the FGS-safe
augmentation knobs (§2.3). YOLOv11s is the fallback only if RF-DETR training is
unstable on this tiny, imbalanced set. ALWAYS fine-tune from the COCO `.pt` —
never random init. Backbone is plumbing, never claimed novel.

FGS-safe augmentation: subtle orbital/muzzle/whisker cues are the whole signal,
so `flipud=0.0` (banned — inverts grimace geometry), `degrees<=10`, `hsv_h~0`
(coat-color is a confound, don't amplify). Selection metric is PAIN RECALL, never
mAP; operating point = FIXED pain-recall >= 0.90 (set LAST, threshold §2.6 rung 4).

Colab T4 primary, M4/MPS fallback (`device=mps`). `ultralytics` is the opt-in
'detect' group dep; this wrapper only assembles a shell command and does not import it.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "detect_yolo.yaml"

# FGS-safe augmentation knobs (§2.3). flipud BANNED; degrees<=10; hsv_h~0.
FGS_SAFE_AUG = {
    "fliplr": 0.5,      # safe (face is L/R symmetric)
    "flipud": 0.0,      # BANNED — inverts grimace geometry
    "degrees": 10,      # larger destroys ear/whisker angle cues
    "translate": 0.1,
    "scale": 0.4,
    "hsv_h": 0.0,       # coat-color is a confound, don't amplify
    "hsv_s": 0.4,       # modest
    "hsv_v": 0.3,       # modest
    "close_mosaic": 10,  # final epochs see the real distribution
}


def load_config(config_path=DEFAULT_CONFIG):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_command(fold: int, device="0", config_path=DEFAULT_CONFIG):
    """Assemble the `yolo detect train` command for one cat-grouped fold.

    Args:
        fold: fold index K -> ``fold{K}.yaml`` data file.
        device: "0" on Colab T4 (CUDA); "mps" on the M4 fallback.
    """
    cfg = load_config(config_path)
    parts = [
        "yolo", "detect", "train",
        f"model={cfg['model']}.pt",               # fine-tune from COCO — never random init
        f"data=fold{fold}.yaml",
        f"imgsz={cfg['resolution']}",
        f"batch={cfg['batch_size']}",
        f"epochs={cfg['epochs']}",
        f"patience={cfg.get('patience', 25)}",
        "optimizer=auto",
    ]
    for k, v in FGS_SAFE_AUG.items():
        parts.append(f"{k}={v}")
    parts.append(f"device={device}")
    return " ".join(parts)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Print the YOLOv11s runner-up train command (§2.5).")
    ap.add_argument("--fold", type=int, required=True, help="fold index K in {0..4}")
    ap.add_argument("--device", default="0", help='"0" for Colab T4 / CUDA, "mps" for M4 fallback')
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    print(build_command(args.fold, device=args.device, config_path=args.config))


if __name__ == "__main__":
    main()
