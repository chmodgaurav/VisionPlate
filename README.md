# KnightSight ANPR Pipeline

A modular, edge-friendly Automatic Number Plate Recognition (ANPR) and Vehicle Intelligence pipeline for research and small deployments. The project integrates vehicle detection, plate localization, and OCR-based text extraction with a Streamlit demo and training utilities.

## Highlights

- End-to-end: vehicle detection → plate localization → OCR.
- Lightweight models: designed to work with Ultralytics YOLO (v8) lightweight backbones for edge use.
- Modular: separate components under `models/` for vehicle and plate detection and the ANPR OCR engine.
- Demo: Streamlit interface available for quick testing and visualization.

## Quick Start

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit demo:

```bash
streamlit run streamlit_app.py
```

Open the shown local URL (usually `http://localhost:8501`) to upload images and run inference.

## Using the Pipeline in Code

Import and run the pipeline from `pipeline.py`:

```python
import cv2
from pipeline import VehicleIntelligencePipeline

pipeline = VehicleIntelligencePipeline()
image = cv2.imread('path/to/image.jpg')
results, vehicles, plates = pipeline.process_image(image_array=image)

for r in results:
    print(r.get('plate_text'), r.get('ocr_confidence'))

annotated = pipeline.annotate_image(image, results, vehicles)
cv2.imwrite('output.jpg', annotated)
```

## Data Preparation & Training

- Dataset location: `Dataset/images` and `Dataset/labels` (YOLO format). Verify these exist before training.
- Convert JSON annotations to YOLO format (if needed):

```bash
python scripts/convert_json_to_yolo.py
```

- Training (example using the included `train_yolov8.py`):

```bash
# GPU training (adjust device/batch/epochs as needed)
python train_yolov8.py --epochs 50 --batch 8 --img 640 --device 0

# CPU quick test
python train_yolov8.py --epochs 1 --batch 2 --img 640 --device cpu
```

Trained weights are saved under `runs/` by default. The demo and `models/plate_detector.py` will look for model files like `yolov8n.pt` or your trained checkpoint.

## Notes

- If you use Tesseract for OCR, install the system binary (e.g., `sudo apt-get install tesseract-ocr`).
- The repository contains: `streamlit_app.py`, `pipeline.py`, `train_yolov8.py`, `models/`, and `Dataset/`.

## License & Acknowledgements
Built with Ultralytics YOLOv8, Streamlit, and OCR tools.

If you want, I can also add a brief troubleshooting section or a contributing guide.
