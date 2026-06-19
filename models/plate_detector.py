import os
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path=None):
        """
        Initialize the plate detector using YOLO11 weights. This implementation
        requires `yolo11.pt` (or a path provided via `model_path` or the
        `YOLO11_WEIGHTS` environment variable). No fallback to YOLOv8 is performed
        to honor the strict YOLO11 requirement.
        """
        if model_path is None:
            model_path = os.environ.get('YOLO11_WEIGHTS', '')
            if not model_path:
                custom_path = 'runs/license_plate_detector/weights/best.pt'
                if os.path.exists(custom_path):
                    model_path = custom_path
                else:
                    model_path = 'yolo11n.pt'
                    print(f"Warning: Custom model not found, falling back to base model: {model_path}")

        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect license plates in an image.
        Returns a list of dictionaries with bounding box and confidence.
        """
        results = self.model(image, verbose=False)

        plates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                plates.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })

        return plates
