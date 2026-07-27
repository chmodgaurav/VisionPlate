import os
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path='runs/license_plate_detector/weights/best.pt'):
        """
        Initialize the plate detector with the custom trained YOLO model or optimized ONNX model.
        Falls back to base YOLOv8 model if the custom weights are not found.
        """
        onnx_path = model_path.replace('.pt', '.onnx')
        
        if os.path.exists(onnx_path):
            print(f"Loading optimized ONNX plate detector: {onnx_path}")
            self.model = YOLO(onnx_path, task='detect')
            self.is_custom = True
        elif os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.is_custom = True
        else:
            print(f"Warning: Custom plate model not found, falling back to base YOLO.")
            onnx_fallback = 'yolov8n.onnx'
            if os.path.exists(onnx_fallback):
                self.model = YOLO(onnx_fallback, task='detect')
            else:
                self.model = YOLO('yolov8n.pt')
            self.is_custom = False
            
    def detect(self, image, track=False):
        """
        Detect license plates in an image.
        If track=True, uses YOLO's native tracking to return object IDs.
        Returns a list of dictionaries with bounding box, confidence, and track_id.
        """
        if track:
            results = self.model.track(image, persist=True, verbose=False)
        else:
            results = self.model(image, verbose=False)
            
        plates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                
                if self.is_custom:
                    # Custom model: only allow class 0 (license plate)
                    if cls_id != 0:
                        continue
                else:
                    # Fallback model: allow vehicles to simulate plate crops for demonstration
                    if cls_id not in [2, 3, 5, 7]:
                        continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                
                plates.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'track_id': track_id
                })
                
        return plates
