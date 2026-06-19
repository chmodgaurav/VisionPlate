import os
import json
from PIL import Image

def convert_json_to_yolo(dataset_dir=None):
    if dataset_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        dataset_dir = os.path.join(root_dir, 'Dataset')
    
    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, 'images')
    # Ensure the labels directory exists; create it if missing
    if not os.path.isdir(labels_dir):
        os.makedirs(labels_dir, exist_ok=True)
        print(f"Created missing labels directory: {labels_dir}")
    # Warn if images directory is missing (cannot proceed without images)
    if not os.path.isdir(images_dir):
        print(f"Warning: images directory not found at {images_dir}. No conversions will run.")
    
    import glob
    # List files only if the directory contains items
    if os.path.isdir(labels_dir):
        for filename in os.listdir(labels_dir):
            if not filename.endswith('.json'):
                continue

            json_path = os.path.join(labels_dir, filename)
            base_name = filename.replace('.json', '')

            # Find matching image file dynamically
            image_files = glob.glob(os.path.join(images_dir, f"{base_name}.*"))
            image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"Warning: Image not found for {filename}")
                continue

            image_path = image_files[0]

            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue

            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error parsing {json_path}")
                    continue

            yolo_lines = []
            for obj in data:
                # Assuming JSON format: [{"class_id": 1, "x": 317, "y": 179, "width": 80, "height": 53}]
                # Assuming x, y is top-left
                class_id = 0  # Map class_id 1 -> 0

                x_min = obj['x']
                y_min = obj['y']
                width = obj['width']
                height = obj['height']

                # Convert to YOLO format (center_x, center_y, width, height) normalized
                x_center = x_min + (width / 2.0)
                y_center = y_min + (height / 2.0)

                # Normalize
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

            txt_path = os.path.join(labels_dir, filename.replace('.json', '.txt'))
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
    print("Conversion to YOLO format completed.")

if __name__ == "__main__":
    convert_json_to_yolo()