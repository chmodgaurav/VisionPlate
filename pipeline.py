import cv2
from models.vehicle_detector import VehicleDetector
from models.plate_detector import PlateDetector
from models.anpr_engine import ANPREngine

class VehicleIntelligencePipeline:
    def __init__(self):
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.anpr_engine = ANPREngine()

    def process_image(self, image_path=None, image_array=None):
        """
        Process an image end-to-end.
        """
        if image_array is not None:
            image = image_array
        elif image_path:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image.")
        else:
            raise ValueError("Must provide image_path or image_array")

        output_data = []

        # 1. Detect Vehicles
        vehicles = self.vehicle_detector.detect(image)
        
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle['box']
            # Optionally, we can crop the vehicle to find plates only within it
            # vehicle_crop = image[vy1:vy2, vx1:vx2]
            # But here we run plate detection on the whole image and then map
            
            # Since plate detection models are usually trained on full frames:
            pass

        # For simplicity and robustness, detect plates directly on the full frame
        plates = self.plate_detector.detect(image)
        
        for plate in plates:
            px1, py1, px2, py2 = plate['box']
            plate_conf = plate['confidence']
            
            # Ensure valid bounds
            py1, py2 = max(0, int(py1)), min(image.shape[0], int(py2))
            px1, px2 = max(0, int(px1)), min(image.shape[1], int(px2))
            
            plate_crop = image[py1:py2, px1:px2]
            
            # Avoid processing empty or extremely small crops
            if plate_crop.size == 0 or plate_crop.shape[0] < 5 or plate_crop.shape[1] < 5:
                continue
                
            # 3. Read Text
            text, ocr_conf = self.anpr_engine.extract_text(plate_crop)
            
            # Map plate to a vehicle (simple IoU or overlap logic could go here)
            associated_vehicle = None
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v['box']
                # If plate center is inside vehicle box
                pc_x = (px1 + px2) / 2
                pc_y = (py1 + py2) / 2
                if vx1 <= pc_x <= vx2 and vy1 <= pc_y <= vy2:
                    associated_vehicle = v['box']
                    break
            
            output_data.append({
                "vehicle_box": associated_vehicle,
                "plate_box": [px1, py1, px2, py2],
                "plate_confidence": float(plate_conf),
                "plate_text": text,
                "ocr_confidence": float(ocr_conf)
            })

        return output_data, vehicles, plates

    def annotate_image(self, image, results, vehicles):
        """
        Draw premium bounding boxes and text overlays on the image.
        """
        annotated = image.copy()
        
        # Color palette (BGR format)
        vehicle_color = (255, 144, 30) # Premium Sky Blue/Orange
        plate_color = (0, 215, 255) # Gold/Amber
        text_bg_color = (15, 15, 15)
        text_color = (255, 255, 255)
        
        # Draw vehicle boxes
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v['box']
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), vehicle_color, 2)
            
            # Label
            label = f"Vehicle: {v['confidence']:.1%}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Ensure label fits within image bounds
            lbl_y1 = max(h + 15, vy1)
            cv2.rectangle(annotated, (vx1, lbl_y1 - h - 10), (vx1 + w + 10, lbl_y1), text_bg_color, -1)
            cv2.putText(annotated, label, (vx1 + 5, lbl_y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                        
        # Draw plate boxes and text
        for res in results:
            px1, py1, px2, py2 = res['plate_box']
            text = res['plate_text']
            conf = res['plate_confidence']
            
            cv2.rectangle(annotated, (px1, py1), (px2, py2), plate_color, 3)
            
            # Label
            label = f"Plate: {text} ({conf:.1%})" if text else f"Plate: {conf:.1%}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lbl_y1 = max(h + 15, py1)
            cv2.rectangle(annotated, (px1, lbl_y1 - h - 10), (px1 + w + 10, lbl_y1), text_bg_color, -1)
            cv2.putText(annotated, label, (px1 + 5, lbl_y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                        
        return annotated
