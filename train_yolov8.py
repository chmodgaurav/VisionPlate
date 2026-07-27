"""Train script that uses YOLOv8 for detection training.

Usage:
    python train_yolov8.py --data data.yaml --epochs 50 --imgsz 640

This script trains from scratch when the `--from-scratch` flag is used, otherwise it
loads the YOLOv8n architecture/weights and begins training.
"""
import argparse
import sys
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='yolov8_train', help='Run name for training')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model name or local path (e.g. yolov8n.pt)')
    parser.add_argument('--from-scratch', action='store_true', help='Train from scratch with a YOLOv8 architecture YAML if available')
    args = parser.parse_args()

    model_name = args.model
    if args.from_scratch and os.path.exists(model_name) and model_name.endswith('.yaml'):
        print(f'Loading YOLOv8 architecture config from {model_name} for scratch training')
    else:
        print(f'Loading YOLOv8 model: {model_name}')

    try:
        model = YOLO(model_name)
    except Exception as e:
        print('Error loading YOLOv8 model:', e)
        print('Ensure the model is a valid ultralytics YOLOv8 configuration or weights file.')
        sys.exit(1)

    print('Starting YOLOv8 training...')
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=args.name, exist_ok=True)
    print(f'Training complete. Weights saved under {args.project}/{args.name}/weights/')


if __name__ == '__main__':
    main()
