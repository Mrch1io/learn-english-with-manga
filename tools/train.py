"""Train the bubble detector or fine-tune an existing local checkpoint.

Examples (paths are relative to the repository, independent of the current cwd)::

    python tools/train.py --data dataset/data.yaml
    python tools/train.py --data dataset/data.yaml \
        --weights output/training/bubble-detector/weights/best.pt \
        --epochs 100 --name bubble-detector-finetuned

Only the default yolov8m.pt may be downloaded automatically by Ultralytics.
Dataset files and custom checkpoints must already exist locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return number


def run_name(value: str) -> str:
    if not value.strip() or value in {".", ".."} or any(c in value for c in "/\\"):
        raise argparse.ArgumentTypeError("must be a single nonempty directory name")
    return value


def repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune the YOLO bubble detector.",
        epilog=("Relative file paths are resolved from the repository root. "
                "Pass a previous best.pt with --weights to fine-tune it. "
                "Run directories are incremented by Ultralytics when a name already exists."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="local dataset YAML file")
    parser.add_argument(
        "--weights", default="yolov8m.pt",
        help="local checkpoint; default is reused from the root or cached under output/models",
    )
    parser.add_argument("--epochs", type=positive_int, default=30, help="training epochs")
    parser.add_argument("--imgsz", type=positive_int, default=640, help="training image size")
    parser.add_argument("--batch", type=positive_int, default=8, help="images per batch")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0, 0,1, or cpu")
    parser.add_argument("--workers", type=nonnegative_int, default=2, help="data-loading workers")
    parser.add_argument("--project", default="output/training", help="training output directory")
    parser.add_argument("--name", type=run_name, default="bubble-detector", help="run directory name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    # Validate local inputs before importing torch/Ultralytics or allocating a model.
    data = repository_path(args.data)
    if not data.is_file():
        parser.error(f"dataset YAML not found: {data}")
    weights = repository_path(args.weights)
    if not weights.is_file():
        if args.weights != "yolov8m.pt" or weights.exists():
            parser.error(f"weights file not found: {weights}")
        weights = PROJECT_ROOT / "output" / "models" / "yolov8m.pt"
        if weights.exists() and not weights.is_file():
            parser.error(f"weights path is not a file: {weights}")
    project = repository_path(args.project)
    if project.exists() and not project.is_dir():
        parser.error(f"output path is not a directory: {project}")
    if not args.device.strip():
        parser.error("--device must not be empty")

    try:
        from ultralytics import YOLO

        weights.parent.mkdir(parents=True, exist_ok=True)
        project.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(weights))
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=args.name,
        )
    except KeyboardInterrupt:
        print("Training interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print("Training complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
