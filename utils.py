import os
import datetime
import threading
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


def read_openai_key_from_file():
    """Return OpenAI API key from OPENAI_API_KEY or openai_api_key.txt if present."""
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    path = os.path.join(os.path.dirname(__file__), "openai_api_key.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


_DEBUG_LOG_ENABLED = False
_DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "debug_log.txt")
_DEBUG_LOG_LOCK = threading.Lock()


def set_debug_logging(enabled: bool, path: str | None = None):
    """Enable/disable debug logging and optionally set a custom path."""
    global _DEBUG_LOG_ENABLED, _DEBUG_LOG_PATH
    _DEBUG_LOG_ENABLED = bool(enabled)
    if path:
        _DEBUG_LOG_PATH = path


def debug_log(message: str):
    """Write a timestamped line to the debug log when enabled."""
    if not _DEBUG_LOG_ENABLED:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"{ts} | {message}\n"
    with _DEBUG_LOG_LOCK:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)


def prepare_image_for_api(path: str, max_dim: int = 640, jpeg_quality: int = 70):
    """Return a resized/compressed image path and info for faster API calls."""
    info = {
        "orig_size": None,
        "new_size": None,
        "scale": None,
        "format": None,
    }
    try:
        img = Image.open(path)
        orig_w, orig_h = img.size
        info["orig_size"] = (orig_w, orig_h)
        scale = min(1.0, max_dim / float(max(orig_w, orig_h)))
        info["scale"] = scale
        if scale < 1.0:
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h
        info["new_size"] = (new_w, new_h)

        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )
        if has_alpha:
            bg = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.split()[-1]
            bg.paste(img, mask=alpha)
            img = bg
        else:
            img = img.convert("RGB")

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(tf, "JPEG", quality=jpeg_quality, optimize=True)
        tf.close()
        info["format"] = "jpg"
        return tf.name, info
    except Exception as e:
        debug_log(f"prepare_image_for_api error={e}")
        return path, info


def read_11_labs_key_from_file():
    """Return ElevenLabs API key from 11_labs_api.txt if present."""
    path = os.path.join(os.path.dirname(__file__), "11_labs_api.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def capture_screenshot(sct, region_phys, quiet=False):
    """Grab screen region using mss."""
    if not quiet:
        print("[capture_screenshot] region_phys =", region_phys)
    return sct.grab(region_phys)


def screenshot_to_array(sshot):
    """Convert an MSS screenshot to RGB numpy array."""
    img = np.array(sshot)
    img = img[:, :, :3][:, :, ::-1]  # drop alpha, BGR -> RGB
    return img


def parse_yolo_results(results_list, capture_w, capture_h, quiet=False):
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
    if not quiet:
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

