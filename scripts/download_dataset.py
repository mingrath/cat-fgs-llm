#!/usr/bin/env python3
"""Download the forked cat-pain dataset (version 1) from Roboflow.

Usage:
    pip install roboflow python-dotenv
    python scripts/download_dataset.py [--format yolov8] [--version 1]

Reads ROBOFLOW_API_KEY / workspace / project from the project-root .env.
Object-detection project (classes: pain / no_pain). Default export = yolov8.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="yolov8",
                    help="export format: yolov8 | coco | voc | folder | ...")
    ap.add_argument("--version", type=int,
                    default=int(os.getenv("ROBOFLOW_VERSION") or 1))
    ap.add_argument("--location", default=str(ROOT / "datasets"))
    args = ap.parse_args()

    key = os.environ["ROBOFLOW_API_KEY"]
    workspace = os.getenv("ROBOFLOW_WORKSPACE_OWNED", "mingraths-workspace")
    project_id = os.getenv("ROBOFLOW_PROJECT_FORK", "cat-pain-ul7lu-p3rtl")

    rf = Roboflow(api_key=key)
    project = rf.workspace(workspace).project(project_id)
    version = project.version(args.version)
    dataset = version.download(args.format, location=args.location)
    print(f"Downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
