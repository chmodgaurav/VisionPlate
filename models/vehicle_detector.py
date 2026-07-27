import os
import cv2
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the vehicle detector with a base YOLO model or optimized ONNX model.
        """
        onnx_path = model_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_path):
            print(f"Loading optimized ONNX vehicle detector: {onnx_path}")
            self.model = YOLO(onnx_path, task='detect')
        else:
            self.model = YOLO(model_path)
    
    def detect(self, image, track=False):
        """
        Detect vehicles in an image.
        If track=True, uses YOLO's native tracking to return object IDs.
        Returns a list of dictionaries with bounding box, confidence, and track_id.
        """
        # COCO class IDs: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        if track:
            results = self.model.track(image, classes=[2, 3, 5, 7], persist=True, verbose=False)
        else:
            results = self.model(image, classes=[2, 3, 5, 7], verbose=False)
        
        vehicles = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                
                vehicles.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'track_id': track_id
                })
                
        return vehicles
