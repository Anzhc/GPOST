from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, QRect, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QFontMetrics,
    QPainterPath,
    QPainterPathStroker,
    QPolygonF,
)
import math
import win32gui
import win32con


def poly_area(pts):
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Check if a point is inside a polygon using the ray casting algorithm."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def rect_in_polygon(rect: QRectF, poly: list[tuple[float, float]]) -> bool:
    """Return True if all rect corners lie inside the polygon."""
    corners = [
        (rect.left(), rect.top()),
        (rect.right(), rect.top()),
        (rect.right(), rect.bottom()),
        (rect.left(), rect.bottom()),
    ]
    return all(point_in_polygon(x, y, poly) for x, y in corners)


class TranslationOverlay(QWidget):
    """Overlay drawing translated text with optional overlap avoidance."""

    def __init__(self, region_logical, capture_w, capture_h, expand_margin=0,
                 use_bbox_instead_of_polygons=False, avoid_overlap=False,
                 font_offset=0):
        super().__init__()
        self.region_logical = region_logical
        self.dpr = region_logical["dpr"]
        self.capture_w = capture_w
        self.capture_h = capture_h
        self.expand_margin = expand_margin
        self.use_bbox_instead_of_polygons = use_bbox_instead_of_polygons
        self.avoid_overlap = avoid_overlap
        self.font_offset = font_offset
        self.translated_items = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self.setGeometry(
            region_logical["left"],
            region_logical["top"],
            region_logical["width"],
            region_logical["height"],
        )
        self.make_window_click_through()
        self.show()
        print("[TranslationOverlay] Created at", region_logical)

    def make_window_click_through(self):
        hwnd = self.winId().__int__()
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        ex_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

    def update_text_for_id(self, det_id, new_text):
        for item in self.translated_items:
            if item["id"] == det_id:
                item["text"] = new_text.strip()
                break
        self.update()

    def paintEvent(self, _):
        def make_path_rect_clip(det_item, force_bbox=False):
            inner_pad = 10
            grow_px = max(0, int(self.expand_margin))

            if det_item["type"] == "bbox":
                x1, y1, x2, y2 = det_item["coords"]
                x1 /= self.dpr
                y1 /= self.dpr
                x2 /= self.dpr
                y2 /= self.dpr
                rx1, ry1 = int(x1) + inner_pad, int(y1) + inner_pad
                rx2, ry2 = int(x2) - inner_pad, int(y2) - inner_pad
                rx1 -= grow_px
                ry1 -= grow_px
                rx2 += grow_px
                ry2 += grow_px
                rect = QRect(rx1, ry1, rx2 - rx1, ry2 - ry1)
                path = QPainterPath()
                path.addRect(QRectF(rect))
                return path, rect, None

            # Use the polygon points provided by YOLO directly. Sorting the
            # points can distort concave shapes and lead to very spiky paths.
            # The YOLO polygons are already ordered, so simply convert them.
            pts = det_item["polygon"]
            if self.use_bbox_instead_of_polygons or force_bbox:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                minx, maxx = min(xs) / self.dpr, max(xs) / self.dpr
                miny, maxy = min(ys) / self.dpr, max(ys) / self.dpr
                rx1, ry1 = int(minx) + inner_pad, int(miny) + inner_pad
                rx2, ry2 = int(maxx) - inner_pad, int(maxy) - inner_pad
                rx1 -= grow_px
                ry1 -= grow_px
                rx2 += grow_px
                ry2 += grow_px
                rect = QRect(rx1, ry1, rx2 - rx1, ry2 - ry1)
                path = QPainterPath()
                path.addRect(QRectF(rect))
                return path, rect, None

            poly_f = QPolygonF([QPointF(px / self.dpr, py / self.dpr) for (px, py) in pts])
            path = QPainterPath()
            path.addPolygon(poly_f)
            if grow_px:
                stroker = QPainterPathStroker()
                stroker.setWidth(grow_px * 2)
                stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
                stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                path = path.united(stroker.createStroke(path)).simplified()
            rect = path.boundingRect().toRect()
            return path, rect, [pt for pt in path.toFillPolygon()]

        def centre_of(path: QPainterPath) -> QPointF:
            r = path.boundingRect()
            return QPointF(r.center())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text_bg = QColor(0, 0, 0, 180)
        draw_items = []
        for itm in self.translated_items:
            if not itm.get("text", "").strip():
                continue
            main_path, main_rect, poly = make_path_rect_clip(itm)
            bbox_path, bbox_rect, _ = make_path_rect_clip(itm, force_bbox=True)
            draw_items.append({
                "item": itm,
                "path": main_path,
                "rect": main_rect,
                "poly": poly,
                "bbox_path": bbox_path,
                "bbox_rect": bbox_rect,
                "dx": 0.0,
                "dy": 0.0,
                "text": itm["text"],
            })

        if self.avoid_overlap and len(draw_items) > 1:
            STEP, max_iter = 3, 200
            for _ in range(max_iter):
                moved = False
                for i in range(len(draw_items)):
                    for j in range(i + 1, len(draw_items)):
                        pi, pj = draw_items[i]["path"], draw_items[j]["path"]
                        if not pi.intersects(pj):
                            continue
                        ci, cj = centre_of(pi), centre_of(pj)
                        dx, dy = cj.x() - ci.x(), cj.y() - ci.y()
                        if abs(dx) < 1 and abs(dy) < 1:
                            dx, dy = STEP, 0
                        mag = (dx * dx + dy * dy) ** 0.5 or 1
                        dx_step, dy_step = STEP * dx / mag, STEP * dy / mag
                        pi.translate(-dx_step / 2, -dy_step / 2)
                        pj.translate(+dx_step / 2, +dy_step / 2)
                        draw_items[i]["dx"] -= dx_step / 2
                        draw_items[i]["dy"] -= dy_step / 2
                        draw_items[j]["dx"] += dx_step / 2
                        draw_items[j]["dy"] += dy_step / 2
                        moved = True
                if not moved:
                    break
            for d in draw_items:
                d["rect"] = d["path"].boundingRect().toRect()
                if d["bbox_path"] is not d["path"]:
                    d["bbox_path"].translate(d["dx"], d["dy"])
                    d["bbox_rect"] = d["bbox_path"].boundingRect().toRect()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(text_bg))
        for d in draw_items:
            use_bbox = (
                d["item"]["type"] == "bbox" or self.use_bbox_instead_of_polygons
            )
            path = d["bbox_path"] if use_bbox else d["path"]
            rect = d["bbox_rect"] if use_bbox else d["rect"]
            poly = None if use_bbox else d["poly"]
            painter.drawPath(path)
            font_pt = self.draw_wrapped_text(
                painter,
                d["text"],
                rect,
                poly,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )
            if (
                font_pt == 0
                and not use_bbox
                and d["item"]["type"] != "bbox"
            ):
                self.draw_wrapped_text(
                    painter,
                    d["text"],
                    d["bbox_rect"],
                    None,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                )
        painter.end()

    def draw_wrapped_text(self, painter, text, bounding_rect, polygon, align_flags):
        """Draw text attempting to fit within the polygon without clipping."""
        MARGIN, MAX_PT, MIN_PT = 1, 72, 6
        bounding_rect = bounding_rect.adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)

        base_font = painter.font()
        flags = align_flags | Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextDontClip

        def calc_rect(pt: int) -> tuple[QFont, QFontMetrics, QRectF]:
            font = QFont(base_font.family(), pt)
            metrics = QFontMetrics(font)
            rect = metrics.boundingRect(bounding_rect, flags, text)
            return font, metrics, QRectF(rect)

        def fits(rect: QRectF, poly_ok: bool) -> bool:
            if rect.width() > bounding_rect.width() or rect.height() > bounding_rect.height():
                return False
            if poly_ok and polygon is not None:
                pts = [(p.x(), p.y()) for p in polygon]
                return rect_in_polygon(rect, pts)
            return True

        low, high = MIN_PT, MAX_PT
        best_font, best_rect = None, None
        while low <= high:
            mid = (low + high) // 2
            font, _metrics, rect = calc_rect(mid)
            if fits(rect, True):
                best_font, best_rect = font, rect
                low = mid + 1
            else:
                high = mid - 1

        if best_font is None:
            low, high = MIN_PT, MAX_PT
            while low <= high:
                mid = (low + high) // 2
                font, _metrics, rect = calc_rect(mid)
                if fits(rect, False):
                    best_font, best_rect = font, rect
                    low = mid + 1
                else:
                    high = mid - 1

        if best_font is None:
            best_font, _metrics, best_rect = calc_rect(MIN_PT)

        if self.font_offset:
            new_pt = max(MIN_PT, min(MAX_PT, best_font.pointSize() + self.font_offset))
            best_font.setPointSize(new_pt)
            _, _metrics, best_rect = calc_rect(new_pt)

        if polygon is not None:
            # Evaluate polygon fit but ignore the result to allow overflow
            rect_in_polygon(best_rect, [(p.x(), p.y()) for p in polygon])

        painter.setFont(best_font)
        outline_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        for dx, dy in outline_offsets:
            painter.setPen(QPen(Qt.GlobalColor.black))
            painter.drawText(bounding_rect.translated(dx, dy), flags, text)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(bounding_rect, flags, text)

        return best_font.pointSize()
