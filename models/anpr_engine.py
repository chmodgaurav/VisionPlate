import easyocr
import re

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
            return "", 0.0, False
            
        results = self.reader.readtext(plate_image)
        
        if not results:
            return "", 0.0, False
            
        text = ""
        avg_conf = 0.0
        
        for (bbox, t, prob) in results:
            text += t.strip() + " "
            avg_conf += prob
            
        avg_conf /= len(results)
        
        # Clean and format the extracted text
        formatted_text = self.format_plate(text)
        is_indian = self.is_indian_plate(text)
        
        return formatted_text, avg_conf, is_indian

    def format_plate(self, text):
        """
        Clean and format the text into an Indian license plate format.
        Handles typical lengths:
        - 10 chars: XX-00-XX-0000 (e.g. MH-12-AB-1234)
        - 9 chars: XX-00-X-0000
        - 7 chars: ABC-1234 (generic fallback requested)
        """
        # Keep only alphanumeric characters and uppercase them
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if len(clean_text) == 10:
            return f"{clean_text[:2]}-{clean_text[2:4]}-{clean_text[4:6]}-{clean_text[6:]}"
        elif len(clean_text) == 9:
            return f"{clean_text[:2]}-{clean_text[2:4]}-{clean_text[4:5]}-{clean_text[5:]}"
        elif len(clean_text) == 7:
            return f"{clean_text[:3]}-{clean_text[3:]}"
            
        return clean_text

    def is_indian_plate(self, text):
        """
        Validates whether the raw alphanumeric text conforms to typical Indian license plate formats.
        Format: 2 Letters (State), 1-2 Digits (District), 1-2 Letters (Series), 1-4 Digits (Number)
        Example: MH12AB1234
        """
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        # Regex for State + District + Series + Number
        pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$"
        return bool(re.match(pattern, clean_text))
