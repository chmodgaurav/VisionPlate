import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# VehicleDetector (reuse the class from models/vehicle_detector.py)
# ---------------------------------------------------------------------------
class VehicleDetector:
    """Simple wrapper around a YOLO model to detect vehicles.

    The underlying model is a YOLO11 nano checkpoint (yolo11n.pt) which is
    lightweight (≈ 5.5 MB) and works well on CPU. It detects the COCO vehicle
    classes: car, motorcycle, bus and truck.
    """

    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        # Load the YOLO model – the library will download the weights if missing.
        self.model = YOLO(model_path)

    def detect(self, image: Any) -> List[Dict[str, Any]]:
        """Detect vehicles in a single image.

        Parameters
        ----------
        image: np.ndarray or str
            Either an already‑loaded image (numpy array) or a path to an image file.
            The YOLO API accepts both.

        Returns
        -------
        List[Dict]
            A list of detections, each dictionary containing:
            ``box`` – [x1, y1, x2, y2] (pixel coordinates)
            ``confidence`` – detection confidence score (0‑1)
        """
        # COCO class IDs for vehicles: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        results = self.model(image, classes=[2, 3, 5, 7], verbose=False)

        vehicles: List[Dict[str, Any]] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                vehicles.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                })
        return vehicles

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def get_model_size(detector: VehicleDetector) -> dict:
    """Return model file size in megabytes and kilobytes.

    Returns a dictionary with keys:
        - ``mb``: size in megabytes (rounded to 2 decimal places)
        - ``kb``: size in kilobytes (rounded to 2 decimal places)
    """
    model_path = getattr(detector.model, 'ckpt_path', 'yolo11n.pt')
    if not model_path or not os.path.exists(model_path):
        return {"mb": 0.0, "kb": 0.0}
        
    size_bytes = os.path.getsize(model_path)
    mb = round(size_bytes / (1024 * 1024), 2)
    kb = round(size_bytes / 1024, 2)
    return {"mb": mb, "kb": kb}

def infer_on_image(detector: VehicleDetector, img_path: str) -> Dict[str, Any]:
    """Run inference on a single image and measure time.

    Returns a dictionary with keys:
    - ``image_path``
    - ``detections`` (list from ``VehicleDetector.detect``)
    - ``inference_time`` (seconds, float)
    """
    start = time.time()
    detections = detector.detect(img_path)
    elapsed = time.time() - start
    return {
        "image_path": img_path,
        "detections": detections,
        "inference_time": round(elapsed, 4),
    }

def infer_on_folder(detector: VehicleDetector, folder: str, out_json: str) -> List[Dict[str, Any]]:
    """Run inference on every image inside ``folder`` (recursively).

    The results are written to ``out_json`` (a JSON list) and also returned.
    """
    folder_path = Path(folder)
    image_paths = list(folder_path.rglob("*.png")) + list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.jpeg"))
    results: List[Dict[str, Any]] = []
    for img_path in image_paths:
        res = infer_on_image(detector, str(img_path))
        results.append(res)
    # Persist results
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    return results

def infer_on_video(detector: VehicleDetector, video_path: str, out_path: str = None, fps: int = 5) -> List[Dict[str, Any]]:
    """Run vehicle detection on a video.

    Parameters
    ----------
    video_path: str
        Path to the input video.
    out_path: str (optional)
        If provided, a video with drawn bounding boxes will be written.
    fps: int
        Frame sampling rate – only one frame every ``fps`` seconds is processed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

    frame_idx = 0
    results: List[Dict[str, Any]] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % fps == 0:
            start = time.time()
            detections = detector.detect(frame)
            elapsed = time.time() - start
            results.append({
                "frame": frame_idx,
                "detections": detections,
                "inference_time": round(elapsed, 4),
            })
            # Draw boxes if output video requested
            if writer:
                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    conf = det["confidence"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                writer.write(frame)
        frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    return results

# ---------------------------------------------------------------------------
# Simple mAP calculation (optional, requires pycocotools)
# ---------------------------------------------------------------------------
def compute_map(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> float:
    """Compute mean Average Precision (mAP) using COCO‑style evaluation.

    This function expects ``predictions`` and ``ground_truth`` to be lists where each
    element corresponds to a single image with the following schema:
    {
        "image_id": str,
        "boxes": [[x1, y1, x2, y2], ...],
        "scores": [float, ...]
    }
    ``ground_truth`` does not contain scores.

    The implementation uses ``pycocotools`` if available; otherwise it falls back
    to a very naive IoU‑based approximation.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise ImportError("pycocotools is required for proper mAP calculation. Install it via 'pip install pycocotools'.")

    # Helper to convert our format into COCO JSON structures.
    def to_coco_dict(items: List[Dict[str, Any]], is_pred: bool) -> Dict:
        images = []
        annotations = []
        for idx, item in enumerate(items):
            img_id = idx
            images.append({"id": img_id, "file_name": item.get("image_path", f"img_{idx}.jpg")})
            for det in item["detections"]:
                x1, y1, x2, y2 = det["box"]
                w = x2 - x1
                h = y2 - y1
                ann = {
                    "id": len(annotations) + 1,
                    "image_id": img_id,
                    "category_id": 1,  # vehicle class
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                }
                if is_pred:
                    ann["score"] = det["confidence"]
                else:
                    ann["iscrowd"] = 0
                annotations.append(ann)
        return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "vehicle"}]}

    gt_coco = to_coco_dict(ground_truth, is_pred=False)
    pred_coco = to_coco_dict(predictions, is_pred=True)

    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(pred_coco["annotations"])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # mAP@IoU=0.50:0.95

# ---------------------------------------------------------------------------
# Demonstration / entry‑point when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vehicle detection utility – Task 1")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Image folder mode
    img_parser = subparsers.add_parser("images", help="Run detection on a folder of images")
    img_parser.add_argument("folder", type=str, help="Path to folder containing images")
    img_parser.add_argument("--gt", type=str, help="Path to ground truth JSON for mAP evaluation")
    img_parser.add_argument("--out", type=str, default="detections.json", help="Output JSON file with results")

    # Video mode
    vid_parser = subparsers.add_parser("video", help="Run detection on a video file")
    vid_parser.add_argument("video", type=str, help="Path to input video")
    vid_parser.add_argument("--gt", type=str, help="Path to ground truth JSON for mAP evaluation")
    vid_parser.add_argument("--out", type=str, default=None, help="Optional output video with drawn boxes")
    vid_parser.add_argument("--sample-fps", type=int, default=1, help="Process one frame every N frames")

    args = parser.parse_args()
    detector = VehicleDetector()

    import statistics

    if args.mode == "images":
        results = infer_on_folder(detector, args.folder, args.out)
        print(f"Processed {len(results)} images. Results saved to {args.out}")
        size_info = get_model_size(detector)
        print(f"Model size: {size_info['mb']} MB ({size_info['kb']} KB)")
        times = [r["inference_time"] for r in results]
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
            print(f"Inference time (s): avg={avg_time:.4f}, min={min_time:.4f}, max={max_time:.4f}, std={std_time:.4f}")
        else:
            print("No inferences were made (empty folder?).")
        if args.gt:
            with open(args.gt) as f:
                gt = json.load(f)
            map_metrics = compute_map(results, gt)
            print(f"mAP metrics: {map_metrics}")
    elif args.mode == "video":
        video_results = infer_on_video(detector, args.video, out_path=args.out, fps=args.sample_fps)
        print(f"Processed video {args.video}. Frame count: {len(video_results)}")
        size_info = get_model_size(detector)
        print(f"Model size: {size_info['mb']} MB ({size_info['kb']} KB)")
        times = [r["inference_time"] for r in video_results]
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
            print(f"Inference time per processed frame (s): avg={avg_time:.4f}, min={min_time:.4f}, max={max_time:.4f}, std={std_time:.4f}")
        else:
            print("No inferences were made (empty video?).")
        if args.gt:
            with open(args.gt) as f:
                gt = json.load(f)
            map_metrics = compute_map(video_results, gt)
            print(f"mAP metrics: {map_metrics}")
