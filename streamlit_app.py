"""
KnightSight ANPR — standalone, no external pipeline module.

Dependencies:
    pip install streamlit ultralytics easyocr opencv-python-headless Pillow numpy pandas

Model bootstrap (first run downloads automatically):
    - YOLOv8n (ultralytics) — vehicle detection
    - EasyOCR (en) — plate OCR

If you have a custom YOLO weights file for plate detection, set:
    PLATE_WEIGHTS = "path/to/plate.pt"
Otherwise the same vehicle-detection model is used and plate crops are
heuristically extracted from the bottom-centre of each vehicle box.
"""

from __future__ import annotations

import os
import time
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Lazy-import heavy deps so Streamlit can show the UI immediately ───────────
@st.cache_resource(show_spinner=False)
def _load_models():
    from ultralytics import YOLO
    import easyocr

    vehicle_weights = os.getenv("VEHICLE_WEIGHTS", "yolov8n.pt")
    plate_weights   = os.getenv("PLATE_WEIGHTS", "")        # "" → use heuristic crop

    vehicle_model = YOLO(vehicle_weights)
    plate_model   = YOLO(plate_weights) if plate_weights else None
    ocr_reader    = easyocr.Reader(["en"], gpu=False, verbose=False)

    return vehicle_model, plate_model, ocr_reader


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass
class VehicleBox:
    x1: int; y1: int; x2: int; y2: int
    confidence: float
    label: str


@dataclass
class PlateResult:
    plate_text: str
    ocr_confidence: float
    plate_confidence: float
    vehicle_box: Optional[VehicleBox] = None
    plate_crop: Optional[np.ndarray] = None   # BGR crop for display


# ── Core inference ────────────────────────────────────────────────────────────

# YOLO class IDs that correspond to vehicles (COCO: 2=car,3=motorcycle,5=bus,7=truck)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def _detect_vehicles(model, frame: np.ndarray, conf_thresh: float) -> list[VehicleBox]:
    results = model(frame, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASS_IDS:
            continue
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[cls]
        boxes.append(VehicleBox(x1, y1, x2, y2, conf, label))
    return boxes


def _heuristic_plate_crop(frame: np.ndarray, vbox: VehicleBox) -> np.ndarray:
    """
    Naive heuristic: take the bottom-30% × centre-60% of a vehicle bounding box.
    Works reasonably well when no dedicated plate detector is available.
    """
    h = vbox.y2 - vbox.y1
    w = vbox.x2 - vbox.x1
    crop_y1 = vbox.y1 + int(h * 0.70)
    crop_x1 = vbox.x1 + int(w * 0.20)
    crop_x2 = vbox.x1 + int(w * 0.80)
    return frame[crop_y1:vbox.y2, crop_x1:crop_x2]


def _detect_plates(plate_model, frame: np.ndarray, conf_thresh: float):
    """Returns list of (crop_bgr, plate_confidence, xyxy)."""
    results = plate_model(frame, verbose=False)[0]
    plates = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        plates.append((crop, conf, (x1, y1, x2, y2)))
    return plates


def _ocr_plate(reader, crop: np.ndarray) -> tuple[str, float]:
    """Run EasyOCR on a plate crop; return (text, confidence)."""
    if crop.size == 0:
        return "", 0.0
    # Upscale small crops — EasyOCR accuracy drops below ~32px height
    h, w = crop.shape[:2]
    if h < 32:
        scale = 32 / h
        crop = cv2.resize(crop, (int(w * scale), 32), interpolation=cv2.INTER_CUBIC)

    detections = reader.readtext(crop)
    if not detections:
        return "", 0.0

    # Pick the highest-confidence detection
    best = max(detections, key=lambda d: d[2])
    text = "".join(c for c in best[1] if c.isalnum()).upper()
    return text, float(best[2])


def process_frame(
    frame: np.ndarray,
    vehicle_model,
    plate_model,
    ocr_reader,
    conf_thresh: float,
) -> tuple[list[PlateResult], list[VehicleBox]]:
    vehicles = _detect_vehicles(vehicle_model, frame, conf_thresh)

    plate_results: list[PlateResult] = []

    if plate_model:
        # Dedicated plate detector path
        raw_plates = _detect_plates(plate_model, frame, conf_thresh)
        for crop, plate_conf, (px1, py1, px2, py2) in raw_plates:
            text, ocr_conf = _ocr_plate(ocr_reader, crop)
            if not text:
                continue
            # Associate with nearest vehicle (IOU heuristic)
            matched_vbox = _match_vehicle(px1, py1, px2, py2, vehicles)
            plate_results.append(PlateResult(text, ocr_conf, plate_conf, matched_vbox, crop))
    else:
        # Heuristic crop path — one crop per vehicle
        for vbox in vehicles:
            crop = _heuristic_plate_crop(frame, vbox)
            text, ocr_conf = _ocr_plate(ocr_reader, crop)
            if not text:
                continue
            plate_results.append(PlateResult(text, ocr_conf, vbox.confidence, vbox, crop))

    return plate_results, vehicles


def _match_vehicle(px1, py1, px2, py2, vehicles: list[VehicleBox]) -> Optional[VehicleBox]:
    """Return the vehicle box that most contains the plate centre."""
    pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
    for v in vehicles:
        if v.x1 <= pcx <= v.x2 and v.y1 <= pcy <= v.y2:
            return v
    return None


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    plates: list[PlateResult],
    vehicles: list[VehicleBox],
) -> np.ndarray:
    out = frame.copy()

    for v in vehicles:
        cv2.rectangle(out, (v.x1, v.y1), (v.x2, v.y2), (255, 200, 0), 2)
        cv2.putText(out, f"{v.label} {v.confidence:.2f}",
                    (v.x1, max(v.y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA)

    for p in plates:
        if p.vehicle_box:
            vb = p.vehicle_box
            # Plate label at top of vehicle box
            label = f"{p.plate_text} ({p.ocr_confidence:.0%})"
            cv2.putText(out, label,
                        (vb.x1, vb.y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)

    return out


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="KnightSight ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

.ks-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 1.25rem;
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 12px; margin-bottom: 1.5rem;
}
.ks-title  { font-size: 1.6rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
.ks-sub    { font-size: 0.85rem; opacity: 0.55; margin: 2px 0 0; }
.ks-badge  { font-size: 0.72rem; padding: 4px 12px; border-radius: 99px;
             border: 1px solid rgba(34,197,94,0.4); color: #22c55e; white-space: nowrap; }

.ks-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.5rem; }
.ks-metric  { padding: 0.9rem 1rem; border: 1px solid rgba(128,128,128,0.12); border-radius: 10px; }
.ks-metric-label { font-size: 0.75rem; opacity: 0.5; text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 4px; }
.ks-metric-value { font-size: 1.6rem; font-weight: 700; }
.ks-metric-unit  { font-size: 0.85rem; font-weight: 400; opacity: 0.6; margin-left: 2px; }
.ks-metric-sub   { font-size: 0.72rem; opacity: 0.4; margin-top: 2px; }

.ks-plate-row { display: flex; align-items: center; justify-content: space-between;
                padding: 10px 0; border-bottom: 1px solid rgba(128,128,128,0.1); }
.ks-plate-row:last-child { border-bottom: none; }
.ks-plate-num   { font-family: monospace; font-size: 1rem; font-weight: 600; letter-spacing: 0.1em; }
.ks-plate-assoc { font-size: 0.75rem; opacity: 0.45; margin-top: 2px; }
.ks-badges { display: flex; gap: 6px; }
.ks-pill   { font-size: 0.7rem; padding: 3px 9px; border-radius: 99px; font-weight: 500; }
.ks-pill-green { background: rgba(34,197,94,0.12); color: #22c55e; }
.ks-pill-amber { background: rgba(234,179,8,0.12);  color: #ca8a04; }
.ks-pill-red   { background: rgba(239,68,68,0.12);  color: #ef4444; }

.ks-section { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
              opacity: 0.4; margin: 1.5rem 0 0.75rem; }
.ks-status  { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; opacity: 0.65; }
.ks-dot     { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def confidence_pill(value: float) -> str:
    pct = f"{value:.1%}"
    cls = "ks-pill-green" if value >= 0.75 else "ks-pill-amber" if value >= 0.50 else "ks-pill-red"
    return f'<span class="ks-pill {cls}">{pct}</span>'


def render_metrics(inference_s: float, n_vehicles: int, n_plates: int, suffix: str = "image mode"):
    st.markdown(f"""
    <div class="ks-metrics">
      <div class="ks-metric">
        <div class="ks-metric-label">Inference time</div>
        <div class="ks-metric-value">{inference_s:.3f}<span class="ks-metric-unit">s</span></div>
        <div class="ks-metric-sub">{suffix}</div>
      </div>
      <div class="ks-metric">
        <div class="ks-metric-label">Vehicles detected</div>
        <div class="ks-metric-value">{n_vehicles}</div>
        <div class="ks-metric-sub">conf ≥ threshold</div>
      </div>
      <div class="ks-metric">
        <div class="ks-metric-label">Plates recognized</div>
        <div class="ks-metric-value">{n_plates}</div>
        <div class="ks-metric-sub">with OCR match</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_plate_results(results: list[PlateResult]):
    if not results:
        st.info("No plates detected above the confidence threshold.")
        return

    st.markdown('<div class="ks-section">Extracted plates</div>', unsafe_allow_html=True)
    rows_html = ""
    for p in results:
        assoc = "Vehicle matched" if p.vehicle_box else "No vehicle match"
        rows_html += f"""
        <div class="ks-plate-row">
          <div>
            <div class="ks-plate-num">{p.plate_text}</div>
            <div class="ks-plate-assoc">{assoc}</div>
          </div>
          <div class="ks-badges">
            {confidence_pill(p.ocr_confidence)}
            {confidence_pill(p.plate_confidence)}
          </div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    with st.expander("Raw JSON output"):
        st.json([
            {
                "plate_text": p.plate_text,
                "ocr_confidence": p.ocr_confidence,
                "plate_confidence": p.plate_confidence,
                "vehicle_matched": p.vehicle_box is not None,
            }
            for p in results
        ])


# ── Sidebar ───────────────────────────────────────────────────────────────────

def build_sidebar() -> tuple[str, int, float]:
    with st.sidebar:
        st.markdown("### KnightSight")
        st.markdown('<div class="ks-status"><div class="ks-dot"></div>Pipeline ready</div>',
                    unsafe_allow_html=True)
        st.divider()

        st.markdown("**Media source**")
        media_type = st.radio("Media", ["Image", "Video"], label_visibility="collapsed")

        st.divider()
        st.markdown("**Inference settings**")
        min_confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05,
                                   help="Minimum detection confidence to include a result.")
        skip_frames = st.slider("Video downsample rate", 1, 30, 5,
                                help="Process 1 frame every N frames.")

        st.divider()
        st.caption("YOLOv8 + EasyOCR · Indian ANPR")

    return media_type, skip_frames, min_confidence


# ── Image mode ────────────────────────────────────────────────────────────────

def run_image_mode(vehicle_model, plate_model, ocr_reader, min_confidence: float):
    uploaded = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    if uploaded is None:
        st.markdown("""
        <div style="border:1px dashed rgba(128,128,128,0.25);border-radius:12px;
                    padding:3rem;text-align:center;opacity:0.5;margin-top:1rem;">
          Drop a JPG, JPEG, or PNG here
        </div>""", unsafe_allow_html=True)
        return

    image = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Running pipeline…"):
        t0 = time.perf_counter()
        plates, vehicles = process_frame(img_bgr, vehicle_model, plate_model, ocr_reader, min_confidence)
        elapsed = time.perf_counter() - t0

    annotated = cv2.cvtColor(annotate_frame(img_bgr, plates, vehicles), cv2.COLOR_BGR2RGB)

    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.image(image, caption="Original", use_container_width=True)
    with col_ann:
        st.image(annotated, caption="Annotated", use_container_width=True)

    render_metrics(elapsed, len(vehicles), len(plates))
    render_plate_results(plates)

    if plates:
        st.markdown('<div class="ks-section">Table view</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{
                "#": i + 1,
                "Plate text": p.plate_text,
                "OCR confidence": f"{p.ocr_confidence:.1%}",
                "Detection confidence": f"{p.plate_confidence:.1%}",
                "Vehicle matched": "Yes" if p.vehicle_box else "No",
            } for i, p in enumerate(plates)]),
            use_container_width=True, hide_index=True,
        )


# ── Video mode ────────────────────────────────────────────────────────────────

def run_video_mode(vehicle_model, plate_model, ocr_reader, skip_frames: int, min_confidence: float):
    uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"],
                                label_visibility="collapsed")
    if uploaded is None:
        st.markdown("""
        <div style="border:1px dashed rgba(128,128,128,0.25);border-radius:12px;
                    padding:3rem;text-align:center;opacity:0.5;margin-top:1rem;">
          Drop an MP4, AVI, or MOV here
        </div>""", unsafe_allow_html=True)
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        os.unlink(tmp_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    col_feed, col_log = st.columns([3, 2])
    with col_feed:
        frame_ph  = st.empty()
        progress  = st.progress(0)
        status_txt = st.empty()
    with col_log:
        st.markdown('<div class="ks-section">Live detection log (last 10)</div>',
                    unsafe_allow_html=True)
        log_ph = st.empty()

    history: list[dict] = []
    seen: set[str] = set()
    frame_idx = processed = 0
    t0 = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            processed += 1
            plates, vehicles = process_frame(frame, vehicle_model, plate_model, ocr_reader, min_confidence)
            annotated = cv2.cvtColor(annotate_frame(frame, plates, vehicles), cv2.COLOR_BGR2RGB)
            frame_ph.image(annotated, caption=f"Frame {frame_idx}", use_container_width=True)

            ts = frame_idx / fps
            for p in plates:
                key = p.plate_text
                if key and key not in seen:
                    seen.add(key)
                    history.append({
                        "Time": f"{ts:.1f}s",
                        "Frame": frame_idx,
                        "Plate": p.plate_text,
                        "OCR conf": f"{p.ocr_confidence:.1%}",
                        "Det conf": f"{p.plate_confidence:.1%}",
                    })
                    log_ph.dataframe(pd.DataFrame(history).tail(10),
                                     use_container_width=True, hide_index=True)

        pct = min(1.0, frame_idx / max(1, total_frames))
        progress.progress(pct)
        status_txt.caption(f"Frame {frame_idx} / {total_frames}  ·  {pct:.0%}")
        frame_idx += 1

    cap.release()
    os.unlink(tmp_path)

    elapsed = time.perf_counter() - t0
    progress.progress(1.0)
    status_txt.caption(f"Done — {processed} frames processed in {elapsed:.1f}s")
    render_metrics(elapsed, 0, len(seen),
                   suffix=f"{processed} frames · {elapsed / max(1, processed):.3f}s/frame")

    if history:
        st.markdown('<div class="ks-section">Full detection history</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.warning("No plates detected above the confidence threshold.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="ks-header">
      <div>
        <p class="ks-title">🚗 KnightSight ANPR</p>
        <p class="ks-sub">YOLOv8 + EasyOCR · Indian license plate recognition</p>
      </div>
      <span class="ks-badge">● Pipeline ready</span>
    </div>
    """, unsafe_allow_html=True)

    media_type, skip_frames, min_confidence = build_sidebar()

    with st.spinner("Loading models…"):
        vehicle_model, plate_model, ocr_reader = _load_models()

    if media_type == "Image":
        run_image_mode(vehicle_model, plate_model, ocr_reader, min_confidence)
    else:
        run_video_mode(vehicle_model, plate_model, ocr_reader, skip_frames, min_confidence)


if __name__ == "__main__":
    main()