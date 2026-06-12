"""Phase A — RF-DETR-Nano fold trainer (IMPLEMENTATION_PLAN §2.4).

Phase A is the **binary spine** (pain / no_pain) and the only Phase that emits a
*validated* number under the firewall: detector P/R and the fixed-recall operating
point are reported on vet-confirmable binary labels, NEVER on VLM-derived AU labels.
The RF-DETR backbone (DINOv2-family) is **plumbing, never claimed novel**.

Full 5-fold training runs on Colab T4 (CUDA): there the `device` arg is OMITTED so
RF-DETR auto-detects CUDA. The *same* call with `device="mps"`, `epochs=1`, and a
small subset is the M4 smoke run (Gate-4 MPS-correctness half for Phase A). CUDA
pins live ONLY in notebooks/colab_train_rfdetr.ipynb, never here.

`rfdetr` is an opt-in `detect`-group dependency; it is imported LAZILY inside the
trainer so this module imports clean without it.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "detect_rfdetr.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    with open(config_path) as f:
        return yaml.safe_load(f)


def num_classes_from_class_names(class_names):
    """Derive ``num_classes`` from the data, never hard-code 2 (the §2.4 footgun box).

    RF-DETR sizes its classification head from ``num_classes`` and the value must
    cover the max COCO ``category_id`` present (RF-DETR reserves the high index for
    background; a 2-category 1-based file can require 3). We let the data set it:
    ``len(class_names)`` for the sorted ['no_pain','pain'] case; override only if
    the head errors against the actual ``category_id`` scheme.
    """
    return len(class_names)


def train_fold(
    fold: int,
    dataset_dir=None,
    class_names=None,
    device=None,
    config_path=DEFAULT_CONFIG,
    output_dir=None,
):
    """Train RF-DETR-Nano on one cat-grouped fold (run once per k in {0..4}).

    Args:
        fold: fold index K. Used to resolve ``./datasets/fold{K}`` and ``./out/fold{K}``.
        dataset_dir: COCO fold dir (train/ valid/ test/ each with _annotations.coco.json).
            Defaults to ``datasets/fold{K}`` relative to cwd (the Colab layout).
        class_names: sorted COCO class names from the datamodule, e.g. ['no_pain','pain'].
            ``num_classes`` is derived from this (footgun box); pass it after
            ``dm.setup('fit'); dm.class_names``.
        device: OMIT (None) on Colab to auto-detect CUDA; pass "mps" for the M4 smoke run.
        config_path: configs/detect_rfdetr.yaml (knobs read from here, never hardcoded).
        output_dir: checkpoint/log dir; defaults to ``out/fold{K}``.
    """
    # Lazy import: rfdetr is the opt-in 'detect' group dep (heavy). Keeps this module
    # import-clean without it. Swap RFDETRSmall only if Nano's pain-recall plateaus.
    from rfdetr import RFDETRNano

    cfg = load_config(config_path)

    if class_names is None:
        raise ValueError(
            "class_names is required: derive num_classes from datamodule.class_names "
            "(the §2.4 footgun box). Build the COCO datamodule, then pass "
            "dm.class_names (expected sorted ['no_pain','pain'])."
        )
    num_classes = num_classes_from_class_names(class_names)  # let the data set it

    if dataset_dir is None:
        dataset_dir = f"./datasets/fold{fold}"
    if output_dir is None:
        output_dir = f"./out/fold{fold}"

    model = RFDETRNano(num_classes=num_classes)  # COCO weights auto-download (cat = COCO id 15)

    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        resolution=cfg["resolution"],            # 512 default; 576 if T4 holds it at batch 4
        epochs=cfg["epochs"],                    # expect earlier convergence than 100
        batch_size=cfg["batch_size"],
        grad_accum_steps=cfg["grad_accum_steps"],  # effective batch = batch_size * grad_accum_steps
        lr=cfg["lr"],
        lr_encoder=cfg["lr_encoder"],            # backbone/encoder LR; drop to 7.5e-5 if unstable
        weight_decay=cfg.get("weight_decay", 1e-4),
        use_ema=cfg.get("use_ema", True),
        early_stopping=cfg.get("early_stopping", True),
        early_stopping_patience=cfg.get("early_stopping_patience", 10),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 0.005),  # 0.5% mAP; 0.001 is val noise
        early_stopping_use_ema=cfg.get("early_stopping_use_ema", True),
        num_workers=cfg.get("num_workers", 2),
        output_dir=str(output_dir),
        progress_bar=cfg.get("progress_bar", "rich"),
        tensorboard=cfg.get("tensorboard", True),
        seed=cfg.get("seed", 42),
    )

    # device omitted -> auto-detect CUDA on Colab; device="mps" only for the M4 smoke run.
    if device is not None:
        train_kwargs["device"] = device

    model.train(**train_kwargs)
    return model
