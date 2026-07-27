import cv2
from models.vehicle_detector import VehicleDetector
from models.plate_detector import PlateDetector
from models.anpr_engine import ANPREngine

class VehicleIntelligencePipeline:
    def __init__(self):
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.anpr_engine = ANPREngine()

    def preprocess_image_clahe(self, image):
        """Apply CLAHE for low-light enhancement without amplifying noise."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def mitigate_glare(self, plate_crop):
        """Reduce glare using adaptive thresholding and morphological operations."""
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def process_image(self, image_path=None, image_array=None, is_video=False):
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

        # Low light enhancement
        enhanced_image = self.preprocess_image_clahe(image)

        output_data = []
        plate_crops = []

        # 1. Detect Vehicles (use tracking if video)
        vehicles = self.vehicle_detector.detect(enhanced_image, track=is_video)

        # 2. Detect plates
        plates = self.plate_detector.detect(enhanced_image, track=is_video)
        
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
                
            # Apply glare mitigation
            processed_crop = self.mitigate_glare(plate_crop)
            
            # 3. Read Text
            text, ocr_conf, is_indian = self.anpr_engine.extract_text(processed_crop)
            
            # 4. Association Logic with Tracking
            associated_vehicle_box = None
            associated_vehicle_id = None
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v['box']
                # If plate center is inside vehicle box
                pc_x = (px1 + px2) / 2
                pc_y = (py1 + py2) / 2
                if vx1 <= pc_x <= vx2 and vy1 <= pc_y <= vy2:
                    associated_vehicle_box = v['box']
                    associated_vehicle_id = v.get('track_id')
                    break
            
            output_data.append({
                "vehicle_box": associated_vehicle_box,
                "vehicle_track_id": associated_vehicle_id,
                "plate_box": [px1, py1, px2, py2],
                "plate_track_id": plate.get('track_id'),
                "plate_confidence": float(plate_conf),
                "plate_text": text,
                "is_indian_plate": is_indian,
                "ocr_confidence": float(ocr_conf)
            })

            # Prepare an annotated crop image with a clear edge/border around the plate
            try:
                crop_annot = plate_crop.copy()
                # Draw a thick green border (edge) on the crop to highlight plate
                h, w = crop_annot.shape[:2]
                border_thickness = max(2, int(round(min(w, h) * 0.03)))
                cv2.rectangle(crop_annot, (0, 0), (w - 1, h - 1), (0, 255, 0), border_thickness)
                # Put OCR text on the crop (top-left)
                text_label = f"{text} ({ocr_conf:.2f})"
                cv2.putText(crop_annot, text_label, (5, max(15, border_thickness + 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception:
                crop_annot = plate_crop

            plate_crops.append({
                "crop": plate_crop,
                "annotated_crop": crop_annot,
                "plate_text": text,
                "ocr_confidence": float(ocr_conf),
                "plate_confidence": float(plate_conf),
                "plate_box": [px1, py1, px2, py2]
            })

        return output_data, vehicles, plates, plate_crops

    def annotate_image(self, image, results, vehicles):
        """
        Draw bounding boxes and text on the image for visualization.
        """
        annotated = image.copy()
        
        # Draw vehicle boxes
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v['box']
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
            label = f"Veh ID:{v['track_id']}" if v.get('track_id') else f"Veh {v['confidence']:.2f}"
            cv2.putText(annotated, label, (vx1, max(10, vy1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
        # NOTE: Plate boxes are not drawn on the full image; cropped plate images contain the highlighted edge.
                        
        return annotated

    def process_video(self, video_path, output_path, skip_frames=2, progress_callback=None):
        """
        Process a video end-to-end, writing an annotated video to output_path.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use 'mp4v' codec for saving video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / (skip_frames + 1), (width, height))
        
        frame_idx = 0
        all_results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % (skip_frames + 1) == 0:
                results, vehicles, plates = self.process_image(image_array=frame, is_video=True)
                annotated_frame = self.annotate_image(frame, results, vehicles)
                out.write(annotated_frame)
                
                if results:
                    all_results.append({
                        "frame": frame_idx,
                        "detections": results
                    })
                    
            if progress_callback and total_frames > 0:
                # Ensure we don't exceed 1.0 (100%) due to cv2 frame count inaccuracies
                progress_callback(min(1.0, frame_idx / total_frames))
                
            frame_idx += 1
            
        cap.release()
        out.release()
        
        return all_results
