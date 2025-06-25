import os
import mss
import numpy as np
import tempfile
from PIL import Image, ImageDraw


def read_api_key_from_file():
    """Return API key from local api_key.txt if present."""
    path = os.path.join(os.path.dirname(__file__), "api_key.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def read_11_labs_key_from_file():
    """Return ElevenLabs API key from 11_labs_api.txt if present."""
    path = os.path.join(os.path.dirname(__file__), "11_labs_api.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def capture_screenshot(sct, region_phys):
    """Grab screen region using mss."""
    print("[capture_screenshot] region_phys =", region_phys)
    return sct.grab(region_phys)


def screenshot_to_array(sshot):
    """Convert an MSS screenshot to RGB numpy array."""
    img = np.array(sshot)
    img = img[:, :, :3][:, :, ::-1]  # drop alpha, BGR -> RGB
    return img


def parse_yolo_results(results_list, capture_w, capture_h):
    """Return list of bbox/polygon detections in physical coordinates."""
    output_raw = []
    polygon_indices = set()

    for r in results_list:
        if r.boxes is not None and len(r.boxes) > 0:
            box_xyxy = r.boxes.xyxy.cpu().numpy()
            box_conf = r.boxes.conf.cpu().numpy()
            box_cls = r.boxes.cls.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(box_xyxy):
                conf_score = float(box_conf[i]) if i < len(box_conf) else 0.0
                class_id = int(box_cls[i]) if i < len(box_cls) else -1
                class_name = r.names.get(class_id, str(class_id)) if r.names else str(class_id)
                output_raw.append({
                    "type": "bbox",
                    "coords": (x1, y1, x2, y2),
                    "confidence": conf_score,
                    "class_id": class_id,
                    "class_name": class_name,
                    "det_index": i,
                })
        if r.masks is not None and r.masks.xyn is not None and len(r.masks.xyn) > 0:
            mask_conf = r.boxes.conf.cpu().numpy() if (r.boxes and r.boxes.conf is not None) else []
            mask_cls = r.boxes.cls.cpu().numpy() if (r.boxes and r.boxes.cls is not None) else []
            for i, poly_points in enumerate(r.masks.xyn):
                conf_score = float(mask_conf[i]) if i < len(mask_conf) else 0.0
                class_id = int(mask_cls[i]) if i < len(mask_cls) else -1
                class_name = r.names.get(class_id, str(class_id)) if r.names else str(class_id)
                polygon_indices.add(i)
                abs_pts = [(px * capture_w, py * capture_h) for (px, py) in poly_points]
                output_raw.append({
                    "type": "polygon",
                    "coords": abs_pts,
                    "confidence": conf_score,
                    "class_id": class_id,
                    "class_name": class_name,
                    "det_index": i,
                })

    final_list = [item for item in output_raw if not (item["type"] == "bbox" and item["det_index"] in polygon_indices)]
    print("[parse_yolo_results] Found", len(final_list), "detections after polygon filtering.")
    return final_list


def crop_polygon_or_bbox(screenshot_rgb, detection):
    """Return temporary path with cropped detection area."""
    if screenshot_rgb is None:
        return None

    h, w, _ = screenshot_rgb.shape
    pil_full = Image.fromarray(screenshot_rgb, "RGB")

    if detection["type"] == "bbox":
        x1, y1, x2, y2 = map(int, detection["coords"])
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        sub = pil_full.crop((x1, y1, x2, y2))
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        sub.save(tf, "PNG")
        tf.close()
        return tf.name

    pts = detection["coords"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = int(min(xs)), int(max(xs))
    miny, maxy = int(min(ys)), int(max(ys))
    minx, maxx = max(0, min(w, minx)), max(0, min(w, maxx))
    miny, maxy = max(0, min(h, miny)), max(0, min(h, maxy))
    if maxx <= minx or maxy <= miny:
        return None
    sub = pil_full.crop((minx, miny, maxx, maxy))
    offset_pts = [(px - minx, py - miny) for (px, py) in pts]
    mask = Image.new("L", (maxx - minx, maxy - miny), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(offset_pts, fill=255)
    sub_rgba = sub.convert("RGBA")
    sub_rgba.putalpha(mask)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    sub_rgba.save(tf, "PNG")
    tf.close()
    return tf.name

