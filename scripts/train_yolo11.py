import argparse
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_json_to_yolo import convert_json_to_yolo


def resolve_device(device_arg):
    if device_arg is None:
        device_arg = "auto"

    if device_arg == "auto":
        if torch.cuda.is_available():
            return 0
        return "cpu"

    if isinstance(device_arg, int):
        return device_arg

    if device_arg in {"cpu", "mps", "cuda"}:
        return device_arg

    try:
        return int(device_arg)
    except ValueError as exc:
        raise ValueError(f"Unsupported device value: {device_arg}") from exc


def get_device_name(device_arg):
    if device_arg == "cpu":
        return "CPU"

    if device_arg == "mps":
        return "Apple MPS"

    if device_arg == 0 and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return f"CUDA:{device_arg}" if isinstance(device_arg, int) else str(device_arg)


def train(
    epochs=50,
    imgsz=640,
    batch=8,
    workers=4,
    base_model="yolo11n.pt",
    device_arg="auto",
    project="runs",
    name="license_plate_detector",
    patience=20,
):
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "Dataset"
    data_yaml_path = project_root / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {data_yaml_path}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    labels_dir = dataset_dir / "labels"
    images_dir = dataset_dir / "images"
    if not labels_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(
            f"Expected both {labels_dir} and {images_dir} to exist before training."
        )

    convert_json_to_yolo(str(dataset_dir))

    import yaml
    try:
        with open(data_yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f) or {}
        yaml_content['path'] = str(dataset_dir.resolve())
        with open(data_yaml_path, 'w') as f:
            yaml.safe_dump(yaml_content, f)
        print(f"Updated dataset path in {data_yaml_path} to absolute: {yaml_content['path']}")
    except Exception as e:
        print(f"Warning: Could not update data.yaml path dynamically: {e}")

    device = resolve_device(device_arg)
    print(f"Using device: {get_device_name(device)}")

    if device == "cpu":
        batch = min(batch, 2)
        workers = 0
    elif device == 0:
        torch.cuda.empty_cache()

    train_kwargs = dict(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        cache=False,
        device=device,
        optimizer="AdamW",
        project=str(project),
        name=name,
        patience=patience,
        amp=True,
        plots=True,
        save=True,
        val=True,
        seed=42,
        lr0=0.01,
        close_mosaic=10,
    )

    print(f"Starting YOLO11 training using {base_model} for {epochs} epochs")
    model = YOLO(base_model)

    try:
        model.train(**train_kwargs)
    except RuntimeError as exc:
        error_message = str(exc).lower()
        if "cuda" in error_message or "cudnn" in error_message or "out of memory" in error_message:
            print("CUDA-related training failure detected. Retrying with smaller batch size and CPU offload fallback.")
            torch.cuda.empty_cache()
            train_kwargs["batch"] = max(1, train_kwargs["batch"] // 2)
            train_kwargs["workers"] = 0
            train_kwargs["device"] = 0 if torch.cuda.is_available() else "cpu"
            try:
                model.train(**train_kwargs)
            except RuntimeError:
                print("Retry failed. Falling back to CPU training.")
                train_kwargs["device"] = "cpu"
                train_kwargs["batch"] = 2
                train_kwargs["amp"] = False
                model.train(**train_kwargs)
            return
        raise

    print(f"Training complete. Best weights saved in {project_root / project / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 on the ANPR license plate dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n.pt",
        help="YOLO11 base checkpoint to fine-tune (recommended: yolo11n.pt for GTX 1650)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, mps, cuda, or an integer GPU index such as 0",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        base_model=args.base_model,
        device_arg=args.device,
    )
