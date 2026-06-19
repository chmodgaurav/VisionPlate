import cv2
import os
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path=None):
        """
        Initialize the vehicle detector. This loader expects YOLO11 weights
        (strict). Provide a path via `model_path` or set `YOLO11_WEIGHTS` env var.
        """
        if model_path is None:
            model_path = os.environ.get('YOLO11_WEIGHTS', '')
            if not model_path:
                model_path = 'yolo11n.pt'
                print(f"Loading vehicle detector base model: {model_path}")

        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect vehicles in an image.
        Returns a list of dictionaries with bounding box and confidence.
        """
        # COCO class IDs: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        results = self.model(image, classes=[2, 3, 5, 7], verbose=False)

        vehicles = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                vehicles.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })

        return vehicles
