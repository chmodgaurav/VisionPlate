# Knight-Sight — ANPR & Vehicle Intelligence Pipeline

A modular, edge-friendly Automatic Number Plate Recognition (ANPR) pipeline for research and small deployments. Combines vehicle detection, plate localization, and OCR-based text extraction, with a Streamlit demo and YOLOv8 training utilities.

## Pipeline

```
Image / Frame
     │
     ▼
Vehicle Detector   (models/vehicle_detector.py)
     │
     ▼
Plate Detector     (models/plate_detector.py)   ← YOLOv8
     │
     ▼
ANPR Engine        (models/anpr_engine.py)      ← OCR text extraction
     │
     ▼
Annotated image + plate text + OCR confidence
```

`pipeline.py` also applies CLAHE-based low-light enhancement and adaptive-threshold glare mitigation to plate crops before OCR.

## Project Structure

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Web demo for upload-and-inspect inference |
| `pipeline.py` | `VehicleIntelligencePipeline` — end-to-end orchestration |
| `models/` | Vehicle detector, plate detector, ANPR/OCR engine |
| `train_yolov8.py` | YOLOv8 training script for the plate detector |
| `data.yaml` | Ultralytics dataset config (single class: `license_plate`) |
| `Dataset/` | Training images and YOLO-format labels |
| `yolov8n.pt` | Pretrained YOLOv8 nano weights |

## Requirements

- Python 3.9+
- A CUDA-capable GPU is recommended for training (CPU works for inference and quick tests)
- Tesseract OCR system binary, if the ANPR engine is configured to use it (`sudo apt-get install tesseract-ocr`)

## Installation

```bash
git clone https://github.com/chmodgaurav/Knight-Sight.git
cd Knight-Sight
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, upload an image, and view detected vehicles, plates, and OCR results.

### Programmatic use

```python
import cv2
from pipeline import VehicleIntelligencePipeline

pipeline = VehicleIntelligencePipeline()
image = cv2.imread("path/to/image.jpg")
results, vehicles, plates = pipeline.process_image(image_array=image)

for r in results:
    print(r.get("plate_text"), r.get("ocr_confidence"))

annotated = pipeline.annotate_image(image, results, vehicles)
cv2.imwrite("output.jpg", annotated)
```

## Data & Training

Dataset lives under `Dataset/images` and `Dataset/labels` in YOLO format, referenced by `data.yaml`:

```yaml
path: ./Dataset
train: images
val: images
nc: 1
names:
  0: 'license_plate'
```

Train the plate detector:

```bash
# GPU
python train_yolov8.py --data data.yaml --epochs 50 --batch 8 --imgsz 640 --model yolov8n.pt

# CPU quick test
python train_yolov8.py --data data.yaml --epochs 1 --batch 2 --imgsz 640
```

Trained weights are saved under `runs/train/<run-name>/weights/`. `models/plate_detector.py` and the Streamlit demo expect a weights file such as `yolov8n.pt` or a checkpoint from a completed run.

## License

Built with Ultralytics YOLOv8, Streamlit, and OCR tooling. See repository for license details.
