import easyocr

class ANPREngine:
    def __init__(self):
        """
        Initialize the OCR engine using EasyOCR.
        gpu=False is used for maximum compatibility on CPU/edge devices by default,
        though it could be set to True if hardware allows.
        """
        # Load English language reader
        self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='models/easyocr_models', user_network_directory='models/easyocr_network', download_enabled=True)
        
    def extract_text(self, plate_image):
        """
        Extract text from a cropped license plate image.
        Returns the concatenated text and the average confidence.
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
            
        results = self.reader.readtext(plate_image)
        
        if not results:
            return "", 0.0
            
        text = ""
        avg_conf = 0.0
        
        for (bbox, t, prob) in results:
            text += t.strip() + " "
            avg_conf += prob
            
        avg_conf /= len(results)
        
        return text.strip(), avg_conf
