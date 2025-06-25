import sys
import math
import os
import mss
import numpy as np
from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QComboBox, QLabel, QMessageBox,
    QGroupBox, QSpinBox, QCheckBox, QSlider, QScrollArea,
    QPlainTextEdit, QSplitter, QInputDialog, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QPoint
from PyQt6.QtGui import QGuiApplication

from utils import (
    read_api_key_from_file,
    read_11_labs_key_from_file,
    capture_screenshot,
    screenshot_to_array,
    parse_yolo_results,
)
from model_helper import prepare_latest_model
from overlays import DetectionOverlay, RegionSelectOverlay, GradientOverlay
from translation_overlay import TranslationOverlay
from translation_threads import (
    GEMINI_AVAILABLE,
    GeminiTranslationThread,
    GeminiBatchTranslationThread,
    ElevenLabsTTSThread,
    GeminiTTSThread,
)

try:
    import keyboard as keyboard_module
    KEYBOARD_AVAILABLE = True
except ImportError:
    keyboard_module = None
    KEYBOARD_AVAILABLE = False

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except Exception:
    Polygon = None
    SHAPELY_AVAILABLE = False


class MainWindow(QMainWindow):
    requestRunInference = pyqtSignal()
    requestTranslate = pyqtSignal()
    requestClearOverlays = pyqtSignal()
    requestSelectArea = pyqtSignal()
    requestRunAll = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPOST - General Purpose On-Screen Translator")

        self.region_device = None
        self.region_logical = None
        self.screenshot_rgb = None
        self.last_detections = []
        self.model = None

        self.detection_overlay = None
        self.translation_overlay = None
        self.gradient_overlay = None
        self.translation_threads = []
        self.tts_threads = []
        self.eleven_api_key = ""
        self.region_select_overlay = None

        self.current_keys_pressed = set()
        self.console_lines = []
        self.translation_history = []

        self.init_ui()
        mp = prepare_latest_model()
        if mp:
            print(f"[MainWindow] Auto-loading YOLO model: {mp.name}")
            try:
                self.model = YOLO(str(mp))
                class_names = [self.model.names[i] for i in sorted(self.model.names)]
                self.populate_class_checkboxes(class_names)
                print("[MainWindow] Model loaded.")
            except Exception as e:
                print(f"[MainWindow] Failed to load auto model: {e}")

        self.requestRunInference.connect(self.on_run_inference)
        self.requestTranslate.connect(self.on_translate)
        self.requestClearOverlays.connect(self.on_clear_overlays)
        self.requestSelectArea.connect(self.on_select_area)
        self.requestRunAll.connect(self.on_run_all)

        self.close_timer = QTimer(self)
        self.close_timer.timeout.connect(self.periodic_check)
        self.close_timer.start(200)

        if KEYBOARD_AVAILABLE:
            self.register_global_hotkeys()

        self._check_api_key()

    # ------------------------------------------------------------------
    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        outer = QHBoxLayout(cw)

        self.main_split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self.main_split)

        left_w = QWidget(); left = QVBoxLayout(left_w)
        mid_w = QWidget(); mid = QVBoxLayout(mid_w)
        right_w = QWidget(); right = QVBoxLayout(right_w)
        self.main_split.addWidget(left_w)
        self.main_split.addWidget(mid_w)
        self.main_split.addWidget(right_w)

        mid.addWidget(QLabel("Translation Output"))
        self.console_edit = QPlainTextEdit()
        self.console_edit.setReadOnly(True)
        mid.addWidget(self.console_edit)

        self.order_combo = QComboBox()
        self.order_combo.addItem("Left-right Top-bottom", "lr_tb")
        self.order_combo.addItem("Right-left Top-bottom", "rl_tb")
        self.order_combo.addItem("Left-right Bottom-top", "lr_bt")
        self.order_combo.addItem("Right-left Bottom-top", "rl_bt")
        mid.addWidget(self.order_combo)

        ctx_group = QGroupBox("Context Settings")
        cg = QVBoxLayout(ctx_group)
        self.context_check = QCheckBox("Send previous translations as context")
        cg.addWidget(self.context_check)
        rel_layout = QHBoxLayout()
        rel_layout.addWidget(QLabel("History size:"))
        self.context_spin = QSpinBox(); self.context_spin.setRange(0, 20); self.context_spin.setValue(0)
        rel_layout.addWidget(self.context_spin)
        cg.addLayout(rel_layout)
        self.context_relevant_check = QCheckBox("Context relevant to current request")
        cg.addWidget(self.context_relevant_check)
        self.btn_clear_history = QPushButton("Clear History")
        self.btn_clear_history.clicked.connect(self.on_clear_history)
        cg.addWidget(self.btn_clear_history)
        mid.addWidget(ctx_group)

        top_label = QLabel("Select Monitor or sub-area, then load YOLO, run inference, and translate.")
        left.addWidget(top_label)

        self.monitor_combo = QComboBox()
        self.monitors = []
        with mss.mss() as sct:
            for i, mon in enumerate(sct.monitors):
                desc = f"Monitor {i}: {mon['width']}x{mon['height']} @({mon['left']},{mon['top']})"
                self.monitor_combo.addItem(desc)
                self.monitors.append(mon)
        left.addWidget(self.monitor_combo)

        hl = QHBoxLayout()
        self.btn_use_mon = QPushButton("Use Full Monitor")
        self.btn_use_mon.clicked.connect(self.on_select_monitor)
        hl.addWidget(self.btn_use_mon)

        self.btn_sel_area = QPushButton("Select Sub-Area (ctrl+alt+S+A)")
        self.btn_sel_area.clicked.connect(self.requestSelectArea.emit)
        hl.addWidget(self.btn_sel_area)
        left.addLayout(hl)

        self.btn_load_model = QPushButton("Load YOLO Model")
        self.btn_load_model.clicked.connect(self.on_load_model)
        left.addWidget(self.btn_load_model)

        self.btn_infer = QPushButton("Run Inference (ctrl+alt+R+T)")
        self.btn_infer.clicked.connect(self.requestRunInference.emit)
        left.addWidget(self.btn_infer)

        self.bypass_yolo_check = QCheckBox("Bypass YOLO (translate full area)")
        left.addWidget(self.bypass_yolo_check)
        self.bypass_overlay_check = QCheckBox("Overlay text when bypassing")
        self.bypass_overlay_check.setEnabled(False)
        left.addWidget(self.bypass_overlay_check)
        self.bypass_yolo_check.stateChanged.connect(
            lambda: self.bypass_overlay_check.setEnabled(self.bypass_yolo_check.isChecked())
        )

        self.language_combo = QComboBox()
        languages = [
            "English","Spanish","French","German","Chinese","Japanese","Korean","Arabic",
            "Portuguese","Russian","Italian","Dutch","Greek","Hebrew","Hindi","Bengali",
            "Polish","Swedish","Turkish","Vietnamese","Indonesian","Malay","Thai","Czech",
            "Slovak","Ukrainian","Romanian","Hungarian","Finnish","Norwegian","Danish"
        ]
        self.language_combo.addItems(languages)
        self.language_combo.setMaxVisibleItems(10)
        left.addWidget(self.language_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ])
        self.model_combo.setCurrentIndex(2)
        left.addWidget(self.model_combo)

        self.btn_translate = QPushButton("Translate (ctrl+alt+G+H)")
        self.btn_translate.clicked.connect(self.requestTranslate.emit)
        left.addWidget(self.btn_translate)

        self.btn_run_all = QPushButton("Run Clean → Inference → Translate (ctrl+alt+B+N)")
        self.btn_run_all.clicked.connect(self.requestRunAll.emit)
        left.addWidget(self.btn_run_all)

        self.btn_clear = QPushButton("Clear Overlays (ctrl+alt+C+V)")
        self.btn_clear.clicked.connect(self.requestClearOverlays.emit)
        left.addWidget(self.btn_clear)

        modifiers_group = QGroupBox("Text Overlay Modifiers")
        mg = QVBoxLayout(modifiers_group)
        mg.addWidget(QLabel("Expand bounding area by X pixels:"))
        self.margin_spin = QSpinBox(); self.margin_spin.setRange(0,999)
        mg.addWidget(self.margin_spin)

        self.bbox_only_check = QCheckBox("Use bounding box for polygon translations")
        mg.addWidget(self.bbox_only_check)
        self.avoid_overlap_check = QCheckBox("Avoid overlapping overlays")
        mg.addWidget(self.avoid_overlap_check)

        mg.addWidget(QLabel("Font size offset (points, can be negative):"))
        self.font_offset_spin = QSpinBox(); self.font_offset_spin.setRange(-64,64); self.font_offset_spin.setValue(0)
        mg.addWidget(self.font_offset_spin)

        self.merge_overlap_check = QCheckBox("Merge overlapping detections (larger box wins)")
        self.merge_overlap_check.setChecked(True)
        mg.addWidget(self.merge_overlap_check)
        self.alt_overlap = QCheckBox("Precise merging")
        mg.addWidget(self.alt_overlap)

        mg.addWidget(QLabel("IoU threshold for merging:"))
        self.merge_iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.merge_iou_slider.setRange(10,100)
        self.merge_iou_slider.setTickInterval(5)
        self.merge_iou_slider.setValue(40)
        mg.addWidget(self.merge_iou_slider)

        self.batch_processing_check = QCheckBox("Batch request (single API call)")
        self.batch_processing_check.setChecked(True)
        mg.addWidget(self.batch_processing_check)

        left.addWidget(modifiers_group)

        self.class_group = QGroupBox("Class Filters")
        self.class_group.setVisible(False)
        cg_lay = QVBoxLayout(self.class_group)
        self.class_scroll = QScrollArea(); self.class_scroll.setWidgetResizable(True)
        self.class_widget = QWidget(); self.class_layout = QVBoxLayout(self.class_widget)
        self.class_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.class_widget.setLayout(self.class_layout); self.class_scroll.setWidget(self.class_widget)
        cg_lay.addWidget(self.class_scroll)
        right.addWidget(self.class_group)

        self.gradient_overlay_check = QCheckBox("Show gradient overlay")
        right.addWidget(self.gradient_overlay_check)

        self.gemini_tts_check = QCheckBox("Queue TTS (Gemini)")
        self.gemini_tts_check.stateChanged.connect(self.on_gemini_tts_toggle)
        right.addWidget(self.gemini_tts_check)

        self.gemini_temp_label = QLabel("Gemini TTS temperature:")
        self.gemini_temp_label.setEnabled(False)
        right.addWidget(self.gemini_temp_label)
        self.gemini_temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.gemini_temp_slider.setRange(0, 200)
        self.gemini_temp_slider.setTickInterval(5)
        self.gemini_temp_slider.setValue(150)
        self.gemini_temp_slider.setEnabled(False)
        right.addWidget(self.gemini_temp_slider)

        self.gemini_voice_combo = QComboBox()
        self.gemini_voice_combo.addItems([
            "Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede",
            "Callirrhoe","Autonoe","Enceladus","Iapetus","Umbriel","Algieba",
            "Despina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar",
            "Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi",
            "Vindemiatrix","Sadachbia","Sadaltager","Sulafat"
        ])
        self.gemini_voice_combo.setCurrentText("Leda")
        self.gemini_voice_combo.setEnabled(False)
        right.addWidget(self.gemini_voice_combo)

        self.gemini_model_combo = QComboBox()
        self.gemini_model_combo.addItem("2.5-pro", "gemini-2.5-pro-preview-tts")
        self.gemini_model_combo.addItem("2.5-flash", "gemini-2.5-flash-preview-tts")
        self.gemini_model_combo.setCurrentIndex(1)
        self.gemini_model_combo.setEnabled(False)
        right.addWidget(self.gemini_model_combo)

        self.gemini_instruction_edit = QLineEdit()
        self.gemini_instruction_edit.setPlaceholderText(
            "Gemini emotions/instructions (optional)"
        )
        self.gemini_instruction_edit.setEnabled(False)
        right.addWidget(self.gemini_instruction_edit)

        self.tts_check = QCheckBox("Queue TTS (11Labs)")
        self.tts_check.stateChanged.connect(self.on_tts_toggle)
        right.addWidget(self.tts_check)

        self.voice_id_edit = QLineEdit()
        self.voice_id_edit.setPlaceholderText("11Labs voice ID (optional)")
        self.voice_id_edit.setEnabled(False)
        right.addWidget(self.voice_id_edit)

        self.class_checkboxes = {}

    # ------------------------------------------------------------------
    def register_global_hotkeys(self):
        if not keyboard_module:
            return
        keyboard_module.add_hotkey('ctrl+alt+r+t', self.requestRunInference.emit)
        keyboard_module.add_hotkey('ctrl+alt+g+h', self.requestTranslate.emit)
        keyboard_module.add_hotkey('ctrl+alt+c+v', self.requestClearOverlays.emit)
        keyboard_module.add_hotkey('ctrl+alt+s+a', self.requestSelectArea.emit)
        keyboard_module.add_hotkey('ctrl+alt+b+n', self.requestRunAll.emit)

    def populate_class_checkboxes(self, class_names):
        while self.class_layout.count():
            item = self.class_layout.takeAt(0)
            if w := item.widget():
                w.deleteLater()
        self.class_checkboxes.clear()
        for name in class_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.class_layout.addWidget(cb)
            self.class_checkboxes[name] = cb
        self.class_layout.addStretch(1)
        self.class_group.setVisible(True)

    def get_enabled_classes(self):
        if not self.class_checkboxes:
            return None
        return {name for name, cb in self.class_checkboxes.items() if cb.isChecked()}

    def _stop_translation_threads(self, force_kill=False):
        for thr in self.translation_threads:
            thr.stop()
            if force_kill:
                thr.requestInterruption()
                thr.terminate()
                thr.wait(100)
            thr.deleteLater()
        self.translation_threads.clear()
        for thr in self.tts_threads:
            thr.stop()
            if force_kill:
                thr.requestInterruption()
                thr.terminate()
                thr.wait(100)
            else:
                thr.wait()
            thr.deleteLater()
        self.tts_threads.clear()

    def _log_console(self, text):
        if not self.console_edit:
            return
        if text:
            self.console_lines.extend(text.splitlines())
        self.console_lines = self.console_lines[-2000:]
        self.console_edit.setPlainText("\n".join(self.console_lines))
        self.console_edit.verticalScrollBar().setValue(self.console_edit.verticalScrollBar().maximum())

    def _add_history(self, translation, original=""):
        entry = f"{translation} - {original}" if original else translation
        self.translation_history.append(entry)
        if len(self.translation_history) > 1000:
            self.translation_history = self.translation_history[-1000:]

    def _start_tts(self, lines: list[str]):
        if not lines:
            return

        for thr in self.tts_threads:
            thr.stop()
            thr.wait()
            thr.deleteLater()
        self.tts_threads.clear()

        if self.gemini_tts_check.isChecked():
            if read_api_key_from_file() or self._check_api_key():
                voice = self.gemini_voice_combo.currentText().strip() or "Leda"
                instructions = self.gemini_instruction_edit.text().strip()
                model = self.gemini_model_combo.currentData() or "gemini-2.5-flash-preview-tts"
                temperature = self.gemini_temp_slider.value() / 100.0
                thr = GeminiTTSThread(
                    lines,
                    read_api_key_from_file(),
                    voice_name=voice,
                    instruction=instructions,
                    model_name=model,
                    temperature=temperature,
                )
                thr.start()
                self.tts_threads.append(thr)
            else:
                self.gemini_tts_check.setChecked(False)

        if self.tts_check.isChecked():
            if not self._check_11labs_key():
                self.tts_check.setChecked(False)
            else:
                voice_id = self.voice_id_edit.text().strip()
                thr = ElevenLabsTTSThread(lines, self.eleven_api_key, voice_id=voice_id)
                thr.start()
                self.tts_threads.append(thr)

    def _sort_pts_clockwise(self, pts):
        if len(pts) < 3:
            return pts
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    def _ordered_detections(self):
        """Return detections sorted using a combined gradient map."""
        if not self.last_detections or self.screenshot_rgb is None:
            return []

        w = self.screenshot_rgb.shape[1]
        h = self.screenshot_rgb.shape[0]
        mode = self.order_combo.currentData()

        def center(det):
            if det['type'] == 'bbox':
                x1, y1, x2, y2 = det['coords']
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0
            xs = [p[0] for p in det['coords']]
            ys = [p[1] for p in det['coords']]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        def det_area(det):
            if det['type'] == 'bbox':
                x1, y1, x2, y2 = det['coords']
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)
            a = 0.0
            pts = det['coords']
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                a += x1 * y2 - x2 * y1
            return abs(a) * 0.5

        def weight(det):
            cx, cy = center(det)
            nx = cx / w
            ny = cy / h

            if mode in ('rl_tb', 'rl_bt'):
                nx = 1.0 - nx
            if mode in ('lr_bt', 'rl_bt'):
                ny = 1.0 - ny

            # Diagonal gradient from chosen corner to the opposite corner
            diag = (nx + ny) * 0.5
            # Additional vertical gradient along the middle
            vert = ny

            area_norm = det_area(det) / float(w * h)

            combined = diag * vert + area_norm * 0.01
            return combined

        return sorted(self.last_detections, key=weight)

    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        if KEYBOARD_AVAILABLE:
            super().keyPressEvent(event)
            return
        if event.isAutoRepeat():
            return
        self.current_keys_pressed.add(event.key())
        self.check_infocus_shortcuts()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() in self.current_keys_pressed:
            self.current_keys_pressed.remove(event.key())

    def check_infocus_shortcuts(self):
        keys = self.current_keys_pressed
        ctrl = Qt.Key.Key_Control
        alt = Qt.Key.Key_Alt
        if not {ctrl, alt}.issubset(keys):
            return
        if {ctrl, alt, Qt.Key.Key_R, Qt.Key.Key_T}.issubset(keys):
            self.requestRunInference.emit()
        elif {ctrl, alt, Qt.Key.Key_G, Qt.Key.Key_H}.issubset(keys):
            self.requestTranslate.emit()
        elif {ctrl, alt, Qt.Key.Key_C, Qt.Key.Key_V}.issubset(keys):
            self.requestClearOverlays.emit()
        elif {ctrl, alt, Qt.Key.Key_S, Qt.Key.Key_A}.issubset(keys):
            self.requestSelectArea.emit()
        elif {ctrl, alt, Qt.Key.Key_B, Qt.Key.Key_N}.issubset(keys):
            self.requestRunAll.emit()

    # ------------------------------------------------------------------
    def _dpr_for_point(self, x, y):
        pt = QPoint(int(x), int(y))
        scr = QGuiApplication.screenAt(pt)
        if not scr:
            scr = QGuiApplication.primaryScreen()
        return scr.devicePixelRatio()

    def _make_logical_rect(self, r_phys):
        dpr = self._dpr_for_point(r_phys['left'], r_phys['top'])
        return {
            'left': int(r_phys['left'] / dpr),
            'top': int(r_phys['top'] / dpr),
            'width': int(r_phys['width'] / dpr),
            'height': int(r_phys['height'] / dpr),
            'dpr': dpr
        }

    # ------------------------------------------------------------------
    def postProcessMergeByArea(self, detections, overlap_threshold=0.6, use_alt_merge=False):
        def poly_area(pts):
            a = 0.0
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                a += x1 * y2 - x2 * y1
            return abs(a) * 0.5

        def det_area(det):
            if det['type'] == 'bbox':
                x1, y1, x2, y2 = det['coords']
                return max(0, x2 - x1) * max(0, y2 - y1)
            return poly_area(det['coords'])

        def bbox_of(det):
            if det['type'] == 'bbox':
                return det['coords']
            xs = [p[0] for p in det['coords']]
            ys = [p[1] for p in det['coords']]
            return (min(xs), min(ys), max(xs), max(ys))

        def intersection_area(a, b):
            x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
            x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
            return max(0, x2 - x1) * max(0, y2 - y1)

        def angle(p, c):
            return math.atan2(p[1] - c[1], p[0] - c[0])

        def hull(points):
            points = sorted(set(points))
            if len(points) < 4:
                return points
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            lower = []
            for p in points:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            upper = []
            for p in reversed(points):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            return lower[:-1] + upper[:-1]

        if use_alt_merge and SHAPELY_AVAILABLE:
            def union_poly(p1, p2):
                p1 = self._sort_pts_clockwise(p1)
                p2 = self._sort_pts_clockwise(p2)
                poly1 = Polygon(p1)
                poly2 = Polygon(p2)
                if not poly1.is_valid:
                    poly1 = poly1.buffer(0)
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)
                try:
                    u = poly1.union(poly2)
                except Exception:
                    return hull(p1 + p2)
                if u.geom_type == 'MultiPolygon':
                    u = max(u.geoms, key=lambda g: g.area)
                return [(x, y) for x, y in u.exterior.coords[:-1]]
        else:
            if use_alt_merge and not SHAPELY_AVAILABLE:
                print('shapely.geometry failed to import, using default merging')
            def union_poly(p1, p2):
                return hull(p1 + p2)

        dets = sorted(detections, key=det_area, reverse=True)
        kept = []
        for cur in dets:
            merged = False
            for prev in kept:
                if cur['type'] != prev['type']:
                    continue
                r_cur = bbox_of(cur)
                r_prev = bbox_of(prev)
                inter = intersection_area(r_cur, r_prev)
                if inter == 0:
                    continue
                ios = inter / min(det_area(cur), det_area(prev))
                if ios >= overlap_threshold:
                    if cur['type'] == 'bbox':
                        x1 = min(r_cur[0], r_prev[0])
                        y1 = min(r_cur[1], r_prev[1])
                        x2 = max(r_cur[2], r_prev[2])
                        y2 = max(r_cur[3], r_prev[3])
                        prev['coords'] = (x1, y1, x2, y2)
                    else:
                        prev['coords'] = union_poly(prev['coords'], cur['coords'])
                    merged = True
                    break
            if not merged:
                kept.append(cur)
        return kept

    # ------------------------------------------------------------------
    @pyqtSlot()
    def on_select_monitor(self):
        idx = self.monitor_combo.currentIndex()
        if idx < 0 or idx >= len(self.monitors):
            return
        mon = self.monitors[idx]
        self.region_device = {
            'left': mon['left'],
            'top': mon['top'],
            'width': mon['width'],
            'height': mon['height']
        }
        self.region_logical = self._make_logical_rect(self.region_device)
        print('[on_select_monitor] region_device =>', self.region_device)
        print('[on_select_monitor] region_logical =>', self.region_logical)
        self.close_overlays()

    @pyqtSlot()
    def on_select_area(self):
        idx = self.monitor_combo.currentIndex()
        if idx < 0 or idx >= len(self.monitors):
            QMessageBox.warning(self, 'Error', 'Pick a valid monitor first.')
            return
        if self.region_select_overlay:
            return
        mon = self.monitors[idx]
        region_mon_phys = {
            'left': mon['left'],
            'top': mon['top'],
            'width': mon['width'],
            'height': mon['height']
        }
        region_mon_logical = self._make_logical_rect(region_mon_phys)
        self.region_select_overlay = RegionSelectOverlay(region_mon_phys, region_mon_logical)
        self.region_select_overlay.regionSelected.connect(self.on_area_selected)

    @pyqtSlot(dict)
    def on_area_selected(self, sub_phys):
        print('[on_area_selected] sub_phys =>', sub_phys)
        self.region_device = sub_phys
        self.region_logical = self._make_logical_rect(self.region_device)
        print('[on_area_selected] region_device =>', self.region_device)
        print('[on_area_selected] region_logical =>', self.region_logical)
        self.region_select_overlay.close()
        self.region_select_overlay = None
        self.close_overlays()

    @pyqtSlot()
    def on_load_model(self):
        dlg = QFileDialog(self, 'Select YOLO Model')
        dlg.setNameFilter('YOLO model files (*.pt *.onnx *.pth)')
        if dlg.exec():
            chosen = dlg.selectedFiles()
            if chosen:
                mp = chosen[0]
                print('[on_load_model] Loading YOLO =>', mp)
                self.model = YOLO(mp)
                print('[on_load_model] Model loaded.')
                class_names = [self.model.names[i] for i in sorted(self.model.names)]
                self.populate_class_checkboxes(class_names)

    @pyqtSlot()
    def on_run_inference(self):
        if not self.region_device:
            QMessageBox.warning(self, 'No region', 'Select a monitor or sub-area first.')
            return
        self.close_overlays()
        with mss.mss() as sct:
            sshot = capture_screenshot(sct, self.region_device)
        self.screenshot_rgb = screenshot_to_array(sshot)
        w = self.screenshot_rgb.shape[1]
        h = self.screenshot_rgb.shape[0]

        if self.bypass_yolo_check.isChecked():
            self.last_detections = [{'type': 'bbox', 'coords': (0, 0, w, h), 'confidence': 1.0, 'class_id': -1, 'class_name': 'full_image'}]
            print('[on_run_inference] Bypassing YOLO, using full image.')
            return

        if not self.model:
            QMessageBox.warning(self, 'No model', 'Load a YOLO model first.')
            return
        print('[on_run_inference] YOLO inference...')
        results = self.model(self.screenshot_rgb, conf=0.25, iou=0.5)
        print('[on_run_inference] done.')
        self.last_detections = parse_yolo_results(results, w, h)
        print('[on_run_inference] # detections =>', len(self.last_detections))
        enabled_classes = self.get_enabled_classes()
        if enabled_classes is not None:
            before = len(self.last_detections)
            self.last_detections = [d for d in self.last_detections if d['class_name'] in enabled_classes]
            print(f'[on_run_inference] after class-filter ⇒ {len(self.last_detections)} (dropped {before - len(self.last_detections)})')
        if self.merge_overlap_check.isChecked():
            print('Running iou merger')
            is_alt = self.alt_overlap.isChecked()
            thr = self.merge_iou_slider.value() / 100.0
            self.last_detections = self.postProcessMergeByArea(self.last_detections, overlap_threshold=thr, use_alt_merge=is_alt)

        if self.gradient_overlay_check.isChecked():
            self.gradient_overlay = GradientOverlay(self.region_logical, w, h, self.order_combo.currentData())
        else:
            self.gradient_overlay = None
        self.detection_overlay = DetectionOverlay(self.region_logical, w, h)
        self.detection_overlay.detections = self.last_detections
        self.detection_overlay.update()

    @pyqtSlot()
    def on_translate(self):
        if not self.last_detections:
            print('No detections'); return
        if self.screenshot_rgb is None:
            QMessageBox.information(self, 'No screenshot', 'No screenshot to translate from!'); return
        if not GEMINI_AVAILABLE:
            QMessageBox.warning(self, 'Gemini missing', 'google.genai not installed!'); return
        if self.translation_threads:
            print('[on_translate] Translation in progress – ignoring duplicate call.'); return
        self._stop_translation_threads()
        self._log_console('━' * 40)
        ordered = self._ordered_detections()
        show_overlay = not (self.bypass_yolo_check.isChecked() and not self.bypass_overlay_check.isChecked())
        if show_overlay:
            if not self.translation_overlay:
                w, h = self.screenshot_rgb.shape[1], self.screenshot_rgb.shape[0]
                self.translation_overlay = TranslationOverlay(
                    self.region_logical, w, h,
                    expand_margin=self.margin_spin.value(),
                    use_bbox_instead_of_polygons=self.bbox_only_check.isChecked(),
                    avoid_overlap=self.avoid_overlap_check.isChecked(),
                    font_offset=self.font_offset_spin.value(),
                )
            self.translation_overlay.translated_items.clear()
            for idx, det in enumerate(ordered):
                if det['type'] == 'bbox':
                    x1, y1, x2, y2 = det['coords']
                    self.translation_overlay.translated_items.append({'id': idx, 'type': 'bbox', 'coords': (x1, y1, x2, y2), 'text': ''})
                else:
                    self.translation_overlay.translated_items.append({'id': idx, 'type': 'polygon', 'polygon': det['coords'], 'text': ''})
            self.translation_overlay.update()
        else:
            if self.translation_overlay:
                self.translation_overlay.close()
                self.translation_overlay = None
        if self.detection_overlay:
            self.detection_overlay.hide()
        tgt_lang = self.language_combo.currentText()
        gem_model = self.model_combo.currentText()

        ctx_enabled = self.context_check.isChecked()
        ctx_num = self.context_spin.value()
        ctx_relevant = self.context_relevant_check.isChecked()
        prev_ctx = "\n".join(self.translation_history[-ctx_num:]) if ctx_enabled and ctx_num > 0 else ""
        if ctx_enabled and ctx_num > 0:
            print('[on_translate] Sending context:\n' + prev_ctx)

        self.translation_threads = []
        if self.batch_processing_check.isChecked():
            thr = GeminiBatchTranslationThread(
                ordered,
                self.screenshot_rgb,
                tgt_lang,
                gem_model,
                previous_context=prev_ctx,
                use_context=ctx_enabled,
                context_relevant=ctx_relevant,
            )
            thr.finished_signal.connect(self.on_batch_results)
            thr.start()
            self.translation_threads.append(thr)
            print('[on_translate] Batch Gemini request launched.')
        else:
            for idx, det in enumerate(ordered):
                thr = GeminiTranslationThread(
                    idx,
                    det,
                    self.screenshot_rgb,
                    tgt_lang,
                    gem_model,
                    previous_context=prev_ctx,
                    use_context=ctx_enabled,
                    context_relevant=ctx_relevant,
                )
                thr.finished_signal.connect(self.on_final_result)
                thr.start()
                self.translation_threads.append(thr)
            print(f'[on_translate] Spawned {len(self.translation_threads)} per-crop threads.')

    @pyqtSlot(dict, str)
    def on_batch_results(self, mapping, error_msg):
        if error_msg:
            self._log_console(error_msg)
            return
        if self.translation_overlay:
            for idx, itm in enumerate(self.translation_overlay.translated_items, start=1):
                entry = mapping.get(str(idx), {})
                if isinstance(entry, dict):
                    trans = (entry.get('translation') or '').strip()
                    orig = (entry.get('original') or '').strip()
                else:
                    trans = (entry or '').strip()
                    orig = ''
                itm['text'] = trans
                self._log_console(f"{idx}. {trans}")
                self._add_history(trans, orig)
            self.translation_overlay.update()
        else:
            for idx_str, val in sorted(mapping.items(), key=lambda p: int(p[0])):
                if isinstance(val, dict):
                    trans = (val.get('translation') or '').strip()
                    orig = (val.get('original') or '').strip()
                else:
                    trans = (val or '').strip()
                    orig = ''
                self._log_console(f"{idx_str}. {trans}")
                self._add_history(trans, orig)

        ordered_lines = []
        if self.translation_overlay:
            ordered_lines = [itm['text'] for itm in self.translation_overlay.translated_items]
        else:
            for i in range(1, len(mapping) + 1):
                val = mapping.get(str(i), {})
                if isinstance(val, dict):
                    ordered_lines.append((val.get('translation') or '').strip())
                else:
                    ordered_lines.append((val or '').strip())

        self._start_tts(ordered_lines)

    @pyqtSlot()
    def on_run_all(self):
        print('[on_run_all] Starting combined process.')
        self.on_run_inference()
        self.on_translate()

    @pyqtSlot(int, str)
    def on_final_result(self, det_id, text):
        print(f'[on_final_result] det_id={det_id}, text={repr(text)}')
        clean = text.strip()
        self._log_console(f"{det_id + 1}. {clean}")
        self._add_history(clean)
        if self.translation_overlay:
            self.translation_overlay.update_text_for_id(det_id, clean)

        self.translation_threads = [t for t in self.translation_threads if t.isRunning()]
        if not self.translation_threads and self.translation_overlay:
            lines = [itm['text'] for itm in self.translation_overlay.translated_items]
            self._start_tts(lines)

    @pyqtSlot()
    def on_clear_history(self):
        self.translation_history.clear()
        self.console_lines.clear()
        if self.console_edit:
            self.console_edit.clear()
        self._log_console('[History cleared]')

    @pyqtSlot()
    def on_clear_overlays(self):
        self.close_overlays()

    def close_overlays(self):
        self._stop_translation_threads(force_kill=True)
        if self.detection_overlay:
            self.detection_overlay.close()
            self.detection_overlay = None
        if self.gradient_overlay:
            self.gradient_overlay.close()
            self.gradient_overlay = None
        if self.translation_overlay:
            self.translation_overlay.close()
            self.translation_overlay = None

    def periodic_check(self):
        pass

    @pyqtSlot(int)
    def on_tts_toggle(self, _):
        enabled = self.tts_check.isChecked()
        self.voice_id_edit.setEnabled(enabled)
        if enabled:
            if not self._check_11labs_key():
                self.tts_check.setChecked(False)

    @pyqtSlot(int)
    def on_gemini_tts_toggle(self, _):
        enabled = self.gemini_tts_check.isChecked()
        self.gemini_voice_combo.setEnabled(enabled)
        self.gemini_instruction_edit.setEnabled(enabled)
        self.gemini_model_combo.setEnabled(enabled)
        self.gemini_temp_label.setEnabled(enabled)
        self.gemini_temp_slider.setEnabled(enabled)
        if enabled:
            if not read_api_key_from_file():
                if not self._check_api_key():
                    self.gemini_tts_check.setChecked(False)

    def _check_api_key(self):
        key = read_api_key_from_file()
        if key:
            return True
        text, ok = QInputDialog.getText(
            self,
            'Enter API Key',
            'Gemini API key:'
        )
        if ok and text.strip():
            path = os.path.join(os.path.dirname(__file__), 'api_key.txt')
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text.strip())
                return True
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to save API key: {e}')
        return False

    def _check_11labs_key(self):
        key = self.eleven_api_key or read_11_labs_key_from_file()
        if key:
            self.eleven_api_key = key
            return True
        text, ok = QInputDialog.getText(
            self,
            'Enter 11Labs API Key',
            '11Labs API key:'
        )
        if ok and text.strip():
            path = os.path.join(os.path.dirname(__file__), '11_labs_api.txt')
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text.strip())
                self.eleven_api_key = text.strip()
                return True
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to save API key: {e}')
        return False

    def closeEvent(self, event):
        print('[MainWindow] closeEvent => stopping threads and overlays.')
        self.close_overlays()
        if self.region_select_overlay:
            self.region_select_overlay.close()
            self.region_select_overlay = None
        event.accept()


def main():
    print('[main] Starting final app: polygons w/ clipped + centered text.')
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

