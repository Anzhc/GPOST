from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, QRect, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygon, QFont, QFontMetrics,
    QPainterPath, QPainterPathStroker, QPolygonF, QRegion,
    QImage, QPixmap
)
import win32gui
import win32con


class DetectionOverlay(QWidget):
    """Transparent overlay showing bounding boxes/polygons."""

    def __init__(self, region_logical, capture_w, capture_h):
        super().__init__()
        self.region_logical = region_logical
        self.dpr = region_logical["dpr"]
        self.capture_w = capture_w
        self.capture_h = capture_h
        self.detections = []

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
        print("[DetectionOverlay] Created at", region_logical)

    def make_window_click_through(self):
        hwnd = self.winId().__int__()
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        ex_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(255, 0, 0, 180))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        text_pen = QPen(Qt.GlobalColor.white)
        text_bg = QColor(0, 0, 0, 160)

        for det in self.detections:
            cname = det["class_name"]
            cconf = det["confidence"]
            label_str = f"{cname} {cconf:.2f}"

            if det["type"] == "bbox":
                x1, y1, x2, y2 = det["coords"]
                x1 /= self.dpr
                y1 /= self.dpr
                x2 /= self.dpr
                y2 /= self.dpr
                w = x2 - x1
                h = y2 - y1
                painter.drawRect(int(x1), int(y1), int(w), int(h))

                if label_str.strip():
                    r_txt = painter.fontMetrics().boundingRect(label_str)
                    r_txt.moveTo(int(x1), int(y1) - r_txt.height())
                    r_txt.setWidth(r_txt.width() + 6)
                    r_txt.setHeight(r_txt.height() + 6)
                    painter.setBrush(QBrush(text_bg))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(r_txt)

                    painter.setPen(text_pen)
                    painter.drawText(
                        r_txt,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                        label_str,
                    )
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                pts = det["coords"]
                if len(pts) < 3:
                    continue
                qpoly = QPolygon([QPoint(int(px / self.dpr), int(py / self.dpr)) for (px, py) in pts])
                painter.drawPolygon(qpoly)

                if label_str.strip():
                    vx, vy = qpoly[0].x(), qpoly[0].y()
                    r_txt = painter.fontMetrics().boundingRect(label_str)
                    r_txt.moveTo(vx, vy - r_txt.height())
                    r_txt.setWidth(r_txt.width() + 6)
                    r_txt.setHeight(r_txt.height() + 6)
                    painter.setBrush(QBrush(text_bg))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(r_txt)
                    painter.setPen(text_pen)
                    painter.drawText(
                        r_txt,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                        label_str,
                    )
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.end()



class RegionSelectOverlay(QWidget):
    """Simple overlay to select a rectangular sub-area."""

    regionSelected = pyqtSignal(dict)

    def __init__(self, region_mon_phys, region_mon_logical, parent=None):
        super().__init__(parent)
        self.region_mon_phys = region_mon_phys
        self.region_mon_logical = region_mon_logical
        self.dpr = region_mon_logical["dpr"]

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.setGeometry(
            region_mon_logical["left"],
            region_mon_logical["top"],
            region_mon_logical["width"],
            region_mon_logical["height"],
        )

        self.dragging = False
        self.start_pos = None
        self.end_pos = None
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.end_pos = event.pos()
            x1_log = min(self.start_pos.x(), self.end_pos.x())
            y1_log = min(self.start_pos.y(), self.end_pos.y())
            x2_log = max(self.start_pos.x(), self.end_pos.x())
            y2_log = max(self.start_pos.y(), self.end_pos.y())
            x1_phys = int(x1_log * self.dpr)
            y1_phys = int(y1_log * self.dpr)
            x2_phys = int(x2_log * self.dpr)
            y2_phys = int(y2_log * self.dpr)
            sub_left = self.region_mon_phys["left"] + x1_phys
            sub_top = self.region_mon_phys["top"] + y1_phys
            sub_w = x2_phys - x1_phys
            sub_h = y2_phys - y1_phys
            region_sub_phys = {"left": sub_left, "top": sub_top, "width": sub_w, "height": sub_h}
            self.regionSelected.emit(region_sub_phys)
            self.close()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
        if self.dragging and self.start_pos and self.end_pos:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRect(self.start_pos, self.end_pos))


class GradientOverlay(QWidget):
    """Overlay visualizing the combined gradient map used for ordering."""

    def __init__(self, region_logical, capture_w, capture_h, mode):
        super().__init__()
        self.region_logical = region_logical
        self.dpr = region_logical["dpr"]
        self.capture_w = capture_w
        self.capture_h = capture_h
        self.mode = mode

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

        self.gradient_pix = self._build_gradient_pixmap()
        self.show()
        print("[GradientOverlay] Created at", region_logical)

    def make_window_click_through(self):
        hwnd = self.winId().__int__()
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        ex_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

    def _build_gradient_pixmap(self):
        img = QImage(self.capture_w, self.capture_h, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)

        max_val = 0.0
        for y in range(self.capture_h):
            ny = y / float(self.capture_h)
            if self.mode in ("lr_bt", "rl_bt"):
                ny = 1.0 - ny
            for x in range(self.capture_w):
                nx = x / float(self.capture_w)
                if self.mode in ("rl_tb", "rl_bt"):
                    nx = 1.0 - nx
                diag = (nx + ny) * 0.5
                vert = ny
                val = diag * vert
                if val > max_val:
                    max_val = val

        if max_val == 0.0:
            max_val = 1.0

        for y in range(self.capture_h):
            ny = y / float(self.capture_h)
            if self.mode in ("lr_bt", "rl_bt"):
                ny = 1.0 - ny
            for x in range(self.capture_w):
                nx = x / float(self.capture_w)
                if self.mode in ("rl_tb", "rl_bt"):
                    nx = 1.0 - nx
                diag = (nx + ny) * 0.5
                vert = ny
                val = diag * vert
                val = max(0.0, val / max_val)
                c = int(val * 255)
                img.setPixel(x, y, QColor(c, c, 0, 120).rgba())

        return QPixmap.fromImage(img)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.gradient_pix)
        painter.end()

