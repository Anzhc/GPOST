import sys
import math
import os
import time
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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QPoint, QThread
from PyQt6.QtGui import QGuiApplication
from PIL import Image

from utils import (
    read_api_key_from_file,
    read_openai_key_from_file,
    read_11_labs_key_from_file,
    capture_screenshot,
    screenshot_to_array,
    parse_yolo_results,
    set_debug_logging,
    debug_log,
)
from model_helper import prepare_latest_model
from overlays import DetectionOverlay, RegionSelectOverlay, GradientOverlay
from overlays import AutoAreaOverlay
from translation_overlay import TranslationOverlay
from translation_threads import (
    GEMINI_AVAILABLE,
    OPENAI_AVAILABLE,
    GeminiTranslationThread,
    GeminiBatchTranslationThread,
    OpenAITranslationThread,
    OpenAIBatchTranslationThread,
    ElevenLabsTTSThread,
    GeminiTTSThread,
    ChatterboxTTSThread,
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


class AutoAreaWorker(QThread):
    area_rect_ready = pyqtSignal(dict)
    pre_capture = pyqtSignal()
    post_capture = pyqtSignal()

    def __init__(self, area_model, monitor_dict, parent=None):
        super().__init__(parent)
        self.area_model = area_model
        self.monitor = dict(monitor_dict) if monitor_dict else None
        self._running = True

    @pyqtSlot(dict)
    def update_monitor(self, mon):
        self.monitor = dict(mon) if mon else None

    def stop(self):
        self._running = False

    def run(self):
        if self.area_model is None:
            return
        import mss as _mss
        from utils import capture_screenshot, screenshot_to_array, parse_yolo_results
        while self._running:
            if not self.monitor:
                self.msleep(100)
                continue
            try:
                self.pre_capture.emit()
                # small pause to allow UI to repaint overlay suppressed
                self.msleep(5)
                with _mss.mss() as sct:
                    sshot = capture_screenshot(sct, self.monitor, quiet=True)
                img = screenshot_to_array(sshot)
                h, w = img.shape[0], img.shape[1]
                # suppress Ultralytics logs
                results = self.area_model(img, conf=0.25, iou=0.5, verbose=False)
                dets = parse_yolo_results(results, w, h, quiet=True)
                if dets:
                    # choose largest bbox (or bbox of polygon)
                    def det_bbox(det):
                        if det['type'] == 'bbox':
                            return det['coords']
                        xs = [p[0] for p in det['coords']]
                        ys = [p[1] for p in det['coords']]
                        return (min(xs), min(ys), max(xs), max(ys))
                    def area_of(b):
                        return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
                    best = max(dets, key=lambda d: area_of(det_bbox(d)))
                    x1, y1, x2, y2 = det_bbox(best)
                    rect_abs = {
                        'left': self.monitor['left'] + int(x1),
                        'top': self.monitor['top'] + int(y1),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                    }
                    self.area_rect_ready.emit(rect_abs)
            except Exception as e:
                print('[auto-area-worker] error:', e)
            finally:
                self.post_capture.emit()
                self.msleep(250)


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
        self.area_model = None

        self.detection_overlay = None
        self.translation_overlay = None
        self.gradient_overlay = None
        self.translation_threads = []
        self.tts_threads = []
        self.eleven_api_key = ""
        self.openai_api_key = ""
        self.region_select_overlay = None
        self.auto_area_overlay = None
        self.auto_area_device_rect = None  # last predicted area in device coords
        self.auto_area_monitor_index = 0

        self.current_keys_pressed = set()
        self.console_lines = []
        self.translation_history = []
        self.batch_result_cache = {}
        self.live_last_sig = None
        self.live_last_trigger_sig = None
        self.live_last_change_ts = 0.0
        self.live_debounce_sec = 2.0
        self.live_last_capture_ts = 0.0
        self.live_capture_interval_sec = 0.5
        self.live_diff_threshold = 0.05
        self.live_baseline_delay_ms = 150
        self.live_reinfer_threshold = 0.15
        self.live_last_change_diff = 0.0
        self.live_reinfer_next = False
        self.live_empty_ratio_threshold = 0.6
        self.live_trigger_active = False
        self.live_page_mode = True
        self.live_page_diff_threshold = 0.04
        self.live_page_sig_size = 32

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

        # Auto area background worker (started when enabled)
        self.auto_area_thread = None

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

        # Automatic Area controls
        auto_row = QHBoxLayout()
        self.auto_area_check = QCheckBox("Automatic Area")
        self.auto_area_check.stateChanged.connect(self.on_auto_area_toggle)
        auto_row.addWidget(self.auto_area_check)
        self.btn_load_area_model = QPushButton("Load Area YOLO Model")
        self.btn_load_area_model.clicked.connect(self.on_load_area_model)
        auto_row.addWidget(self.btn_load_area_model)
        left.addLayout(auto_row)

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
        gemini_models = [
            "gemini-3-flash-preview",
            "gemini-2.5-flash-preview-09-2025",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-lite-preview-09-2025",
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        for name in gemini_models:
            self.model_combo.addItem(name, ("gemini", name))
        self.model_combo.addItem("OpenAI GPT-4 Turbo", ("openai", "gpt-4-turbo"))
        self.model_combo.addItem("OpenAI GPT-4o", ("openai", "gpt-4o"))
        self.model_combo.addItem("OpenAI GPT-4o Mini", ("openai", "gpt-4o-mini"))
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
        self.batch_processing_check.stateChanged.connect(self.on_batch_toggle)
        mg.addWidget(self.batch_processing_check)
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch size (max):"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 40)
        self.batch_size_spin.setValue(12)
        self.batch_size_spin.setEnabled(self.batch_processing_check.isChecked())
        batch_row.addWidget(self.batch_size_spin)
        mg.addLayout(batch_row)
        self.fast_mode_check = QCheckBox("Fast mode (downscale + low detail)")
        self.fast_mode_check.stateChanged.connect(self.on_fast_mode_toggle)
        mg.addWidget(self.fast_mode_check)

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

        self.live_check = QCheckBox("Live updates (2s delay)")
        self.live_check.stateChanged.connect(self.on_live_toggle)
        right.addWidget(self.live_check)

        self.debug_btn = QPushButton("Debug: OFF")
        self.debug_btn.setCheckable(True)
        self.debug_btn.toggled.connect(self.on_debug_toggle)
        right.addWidget(self.debug_btn)

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

        self.chatterbox_tts_check = QCheckBox("Queue TTS (Chatterbox)")
        self.chatterbox_tts_check.stateChanged.connect(self.on_chatterbox_tts_toggle)
        right.addWidget(self.chatterbox_tts_check)

        self.chatterbox_voice_combo = QComboBox()
        self.chatterbox_voice_combo.setEnabled(False)
        self._populate_chatterbox_voices()
        right.addWidget(self.chatterbox_voice_combo)

        self.tts_check = QCheckBox("Queue TTS (11Labs)")
        self.tts_check.stateChanged.connect(self.on_tts_toggle)
        right.addWidget(self.tts_check)

        self.voice_id_edit = QLineEdit()
        self.voice_id_edit.setPlaceholderText("11Labs voice ID (optional)")
        self.voice_id_edit.setEnabled(False)
        right.addWidget(self.voice_id_edit)

        self.class_checkboxes = {}
        self.monitor_combo.currentIndexChanged.connect(self.on_monitor_changed)

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

    def _current_monitor(self):
        idx = self.monitor_combo.currentIndex()
        if idx < 0 or idx >= len(self.monitors):
            return None, -1
        return self.monitors[idx], idx

    def _monitor_logical_rect(self, mon):
        return self._make_logical_rect(mon)

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

    def _populate_chatterbox_voices(self):
        if not hasattr(self, 'chatterbox_voice_combo'):
            return
        current_data = None
        if self.chatterbox_voice_combo.count():
            current_data = self.chatterbox_voice_combo.currentData()
        self.chatterbox_voice_combo.blockSignals(True)
        self.chatterbox_voice_combo.clear()
        self.chatterbox_voice_combo.addItem('Model default voice', '')
        voice_dir = os.path.join(os.path.dirname(__file__), 'voicebank')
        if os.path.isdir(voice_dir):
            for name in sorted(os.listdir(voice_dir)):
                if name.lower().endswith('.wav'):
                    self.chatterbox_voice_combo.addItem(name, os.path.join(voice_dir, name))
        if current_data:
            idx = self.chatterbox_voice_combo.findData(current_data)
            if idx >= 0:
                self.chatterbox_voice_combo.setCurrentIndex(idx)
            else:
                self.chatterbox_voice_combo.setCurrentIndex(0)
        else:
            self.chatterbox_voice_combo.setCurrentIndex(0)
        self.chatterbox_voice_combo.blockSignals(False)

    def _start_tts(self, lines: list[str]):
        if not lines:
            return

        lines = [line.replace('-', '—') if isinstance(line, str) else line for line in lines]

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

        if getattr(self, 'chatterbox_tts_check', None) and self.chatterbox_tts_check.isChecked():
            voice_path = self.chatterbox_voice_combo.currentData()
            thr = ChatterboxTTSThread(lines, voice_path=voice_path or None)
            thr.start()
            self.tts_threads.append(thr)

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

    def _compute_batch_size(self, total_count: int, max_size: int) -> int:
        if total_count <= 0:
            return 0
        max_size = max(1, int(max_size))
        if total_count <= max_size:
            return total_count
        target_batches = 3
        size = math.ceil(total_count / target_batches)
        if size > max_size:
            target_batches = 4
            size = math.ceil(total_count / target_batches)
        if size > max_size:
            size = max_size
        if size < 2 and total_count > 1:
            size = 2
        return min(total_count, size)

    def _detection_bbox(self, det):
        if det['type'] == 'bbox':
            return det['coords']
        xs = [p[0] for p in det['coords']]
        ys = [p[1] for p in det['coords']]
        return (min(xs), min(ys), max(xs), max(ys))

    def _debug_preview(self, text, limit=160):
        if text is None:
            return ""
        cleaned = text.replace("\r", "\\r").replace("\n", "\\n")
        if len(cleaned) > limit:
            cleaned = cleaned[:limit] + "..."
        return cleaned

    def _debug_log_detections(self, ordered):
        if not ordered:
            debug_log("det_list empty")
            return
        debug_log(f"det_list total={len(ordered)}")
        for idx, det in enumerate(ordered, start=1):
            det_type = det.get("type", "")
            cls = det.get("class_name", "")
            if det_type == "bbox":
                x1, y1, x2, y2 = det.get("coords", (0, 0, 0, 0))
                debug_log(
                    f"det id={idx} type=bbox class={cls} "
                    f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                )
                continue
            pts = det.get("coords") or []
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                sample = ";".join([f"{p[0]:.0f},{p[1]:.0f}" for p in pts[:3]])
            else:
                x1 = y1 = x2 = y2 = 0.0
                sample = ""
            debug_log(
                f"det id={idx} type=poly class={cls} "
                f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) pts={sample}"
            )

    def _debug_log_translation(self, det_id, translation, original=""):
        preview = self._debug_preview(translation)
        orig_preview = self._debug_preview(original)
        debug_log(
            f"result id={det_id} len={len(translation)} text={preview} "
            f"orig_len={len(original)} orig={orig_preview}"
        )

    def _compute_page_signature(self, screenshot_rgb):
        if screenshot_rgb is None:
            return None
        size = max(8, int(self.live_page_sig_size))
        pil_img = Image.fromarray(screenshot_rgb, "RGB")
        small = pil_img.resize((size, size), Image.BILINEAR).convert("L")
        arr = np.asarray(small, dtype=np.uint8)
        return (arr // 16).astype(np.uint8)

    def _compute_live_signature(self, screenshot_rgb):
        if self.live_page_mode:
            return self._compute_page_signature(screenshot_rgb)
        return self._compute_text_signature(screenshot_rgb)

    def _maybe_flag_live_reinfer(self, lines):
        if not self.live_check.isChecked():
            return
        if not self.live_trigger_active:
            return
        if not lines:
            self.live_trigger_active = False
            return
        total = len(lines)
        empty = sum(1 for line in lines if not line.strip())
        ratio = empty / float(total) if total else 0.0
        debug_log(
            f"live_result_empty_ratio={ratio:.2f} empty={empty} total={total}"
        )
        if ratio >= self.live_empty_ratio_threshold:
            self.live_reinfer_next = True
            debug_log("live_reinfer_due_to_empty")
        self.live_trigger_active = False

    def _hide_overlays_for_capture(self):
        states = []
        overlays = [
            ("translation", self.translation_overlay),
            ("detection", self.detection_overlay),
            ("gradient", self.gradient_overlay),
        ]
        for name, overlay in overlays:
            if overlay is None:
                continue
            try:
                was_visible = overlay.isVisible()
                if was_visible:
                    overlay.setVisible(False)
                states.append((overlay, was_visible, name))
            except Exception:
                continue
        if states:
            QGuiApplication.processEvents()
            summary = ",".join([f"{name}:{int(vis)}" for _o, vis, name in states])
            debug_log(f"capture_hide_overlays {summary}")
        return states

    def _restore_overlays(self, states):
        if not states:
            return
        for overlay, was_visible, _name in states:
            if not was_visible:
                continue
            try:
                overlay.setVisible(True)
            except Exception:
                pass
        QGuiApplication.processEvents()
        summary = ",".join([f"{name}:{int(vis)}" for _o, vis, name in states])
        debug_log(f"capture_restore_overlays {summary}")

    def _compute_text_signature(self, screenshot_rgb):
        if screenshot_rgb is None or not self.last_detections:
            return None
        h, w = screenshot_rgb.shape[:2]
        boxes = []
        for det in self.last_detections:
            x1, y1, x2, y2 = self._detection_bbox(det)
            x1 = int(max(0, min(w, x1)))
            x2 = int(max(0, min(w, x2)))
            y1 = int(max(0, min(h, y1)))
            y2 = int(max(0, min(h, y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append((x1, y1, x2, y2))
        if not boxes:
            return None
        boxes.sort(key=lambda b: (b[1], b[0], b[3], b[2]))
        pil_img = Image.fromarray(screenshot_rgb, "RGB")
        sig_parts = []
        for box in boxes:
            crop = pil_img.crop(box).resize((16, 16), Image.BILINEAR).convert("L")
            arr = np.asarray(crop, dtype=np.uint8)
            q = (arr // 16).astype(np.uint8)
            sig_parts.append(q)
        if not sig_parts:
            return None
        return np.stack(sig_parts, axis=0)

    def _signature_diff(self, sig_a, sig_b):
        if sig_a is None or sig_b is None:
            return None
        if sig_a.shape != sig_b.shape:
            return None
        diff = np.abs(sig_a.astype(np.int16) - sig_b.astype(np.int16)).mean()
        return float(diff) / 15.0

    def _capture_signature_frame(self, hide_overlays=False):
        if not self.region_device:
            return None
        states = []
        if hide_overlays:
            states = self._hide_overlays_for_capture()
        try:
            with mss.mss() as sct:
                sshot = sct.grab(self.region_device)
            return screenshot_to_array(sshot)
        except Exception as e:
            debug_log(f"live_capture_error={e}")
            return None
        finally:
            if hide_overlays:
                self._restore_overlays(states)

    def _capture_text_signature(self):
        if not self.region_device:
            return None
        if not self.live_page_mode and not self.last_detections:
            return None
        hide_overlays = not self.live_page_mode
        img = self._capture_signature_frame(hide_overlays=hide_overlays)
        if img is None:
            return None
        return self._compute_live_signature(img)

    def _capture_current_screenshot(self, hide_overlays=False):
        if not self.region_device:
            return False
        states = []
        if hide_overlays:
            states = self._hide_overlays_for_capture()
        try:
            with mss.mss() as sct:
                sshot = capture_screenshot(sct, self.region_device)
            self.screenshot_rgb = screenshot_to_array(sshot)
            return True
        except Exception as e:
            debug_log(f"live_capture_error={e}")
            return False
        finally:
            if hide_overlays:
                self._restore_overlays(states)

    def _live_translate(self):
        overlay_visible = bool(self.translation_overlay and self.translation_overlay.isVisible())
        debug_log(f"live_translate start overlay_visible={int(overlay_visible)}")
        self.live_trigger_active = True
        if not self._capture_current_screenshot(hide_overlays=True):
            self.live_trigger_active = False
            return
        if self.bypass_yolo_check.isChecked() and not self.last_detections:
            h, w = self.screenshot_rgb.shape[:2]
            self.last_detections = [{
                'type': 'bbox',
                'coords': (0, 0, w, h),
                'confidence': 1.0,
                'class_id': -1,
                'class_name': 'full_image',
            }]
        self.on_translate()

    def _live_run_inference_and_translate(self):
        if not self.region_device:
            return
        debug_log("live_infer start")
        self.live_trigger_active = True
        t_start = time.perf_counter()
        if not self._capture_current_screenshot(hide_overlays=True):
            self.live_trigger_active = False
            return
        t_capture = time.perf_counter()
        debug_log(f"live_infer_capture_ms={int((t_capture - t_start) * 1000)}")
        w = self.screenshot_rgb.shape[1]
        h = self.screenshot_rgb.shape[0]

        if self.bypass_yolo_check.isChecked():
            self.last_detections = [{
                'type': 'bbox',
                'coords': (0, 0, w, h),
                'confidence': 1.0,
                'class_id': -1,
                'class_name': 'full_image',
            }]
            debug_log("live_infer_bypass_yolo=true")
            self.on_translate()
            return

        if not self.model:
            debug_log("live_infer_no_model")
            self.live_trigger_active = False
            return

        t_yolo_start = time.perf_counter()
        results = self.model(self.screenshot_rgb, conf=0.25, iou=0.5)
        t_yolo_end = time.perf_counter()
        debug_log(f"live_infer_yolo_ms={int((t_yolo_end - t_yolo_start) * 1000)}")

        t_parse_start = time.perf_counter()
        self.last_detections = parse_yolo_results(results, w, h)
        t_parse_end = time.perf_counter()
        debug_log(
            f"live_infer_parse_ms={int((t_parse_end - t_parse_start) * 1000)} "
            f"detections={len(self.last_detections)}"
        )
        enabled_classes = self.get_enabled_classes()
        if enabled_classes is not None:
            before = len(self.last_detections)
            self.last_detections = [
                d for d in self.last_detections if d['class_name'] in enabled_classes
            ]
            debug_log(
                f"live_infer_class_filter kept={len(self.last_detections)} "
                f"dropped={before - len(self.last_detections)}"
            )
        if self.merge_overlap_check.isChecked():
            is_alt = self.alt_overlap.isChecked()
            thr = self.merge_iou_slider.value() / 100.0
            t_merge_start = time.perf_counter()
            self.last_detections = self.postProcessMergeByArea(
                self.last_detections, overlap_threshold=thr, use_alt_merge=is_alt
            )
            t_merge_end = time.perf_counter()
            debug_log(f"live_infer_merge_ms={int((t_merge_end - t_merge_start) * 1000)}")
        debug_log(f"live_infer_total_ms={int((time.perf_counter() - t_start) * 1000)}")
        self.on_translate()

    def _update_live_signature_baseline(self):
        if not self.live_check.isChecked():
            return
        sig = self._capture_text_signature()
        if sig is None:
            return
        now = time.perf_counter()
        self.live_last_sig = sig
        self.live_last_trigger_sig = sig
        self.live_last_change_ts = now
        self.live_last_change_diff = 0.0
        self.live_last_capture_ts = now
        mode = "page" if self.live_page_mode else "text"
        debug_log(f"live_baseline_updated shape={sig.shape} mode={mode}")

    def _schedule_live_signature_baseline(self):
        if not self.live_check.isChecked():
            return
        delay = int(self.live_baseline_delay_ms)
        if delay <= 0:
            self._update_live_signature_baseline()
        else:
            QTimer.singleShot(delay, self._update_live_signature_baseline)

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

    def _ensure_auto_area_overlay(self):
        if not self.auto_area_check.isChecked():
            return
        mon, idx = self._current_monitor()
        if not mon:
            return
        mon_logical = self._monitor_logical_rect(mon)
        # Recreate if monitor changed or overlay missing
        if (self.auto_area_overlay is None) or (self.auto_area_monitor_index != idx):
            if self.auto_area_overlay:
                try:
                    self.auto_area_overlay.close()
                except Exception:
                    pass
                self.auto_area_overlay = None
            self.auto_area_overlay = AutoAreaOverlay(mon_logical)
            self.auto_area_monitor_index = idx
        # Push the latest rect
        if self.auto_area_device_rect:
            # Convert absolute device rect to monitor-relative
            rel_left = self.auto_area_device_rect['left'] - mon['left']
            rel_top = self.auto_area_device_rect['top'] - mon['top']
            rect_rel = {
                'left': rel_left,
                'top': rel_top,
                'width': self.auto_area_device_rect['width'],
                'height': self.auto_area_device_rect['height'],
            }
            self.auto_area_overlay.set_rect(rect_rel)

    def _start_auto_area_worker(self):
        if not (self.auto_area_check.isChecked() and self.area_model is not None):
            return
        mon, _ = self._current_monitor()
        if not mon:
            return
        if self.auto_area_thread and self.auto_area_thread.isRunning():
            # update only monitor
            self.auto_area_thread.update_monitor(mon)
            return
        self.auto_area_thread = AutoAreaWorker(self.area_model, mon)
        self.auto_area_thread.area_rect_ready.connect(self._on_auto_area_result)
        self.auto_area_thread.start()

    def _stop_auto_area_worker(self):
        if self.auto_area_thread:
            try:
                self.auto_area_thread.stop()
                self.auto_area_thread.wait(300)
            except Exception:
                pass
            self.auto_area_thread = None

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
    def on_load_area_model(self):
        dlg = QFileDialog(self, 'Select Area YOLO Model')
        dlg.setNameFilter('YOLO model files (*.pt *.onnx *.pth)')
        if dlg.exec():
            chosen = dlg.selectedFiles()
            if chosen:
                mp = chosen[0]
                print('[on_load_area_model] Loading Area YOLO =>', mp)
                try:
                    self.area_model = YOLO(mp)
                    print('[on_load_area_model] Area model loaded.')
                    if self.auto_area_check.isChecked():
                        self._start_auto_area_worker()
                except Exception as e:
                    print('[on_load_area_model] Failed to load area model:', e)

    @pyqtSlot(int)
    def on_auto_area_toggle(self, _):
        if not self.auto_area_check.isChecked():
            if self.auto_area_overlay:
                try:
                    self.auto_area_overlay.close()
                except Exception:
                    pass
                self.auto_area_overlay = None
            self._stop_auto_area_worker()
            return
        # Ensure overlay shows up immediately
        self._ensure_auto_area_overlay()
        self._start_auto_area_worker()

    @pyqtSlot(int)
    def on_monitor_changed(self, _):
        # Rebuild overlay to new monitor if needed
        if self.auto_area_check.isChecked():
            self._ensure_auto_area_overlay()
            # inform worker
            if self.auto_area_thread and self.auto_area_thread.isRunning():
                mon, _ = self._current_monitor()
                if mon:
                    self.auto_area_thread.update_monitor(mon)

    @pyqtSlot()
    def on_run_inference(self):
        # If auto-area is enabled, use the latest/next predicted area as region
        if self.auto_area_check.isChecked() and self.area_model is not None:
            if not self.auto_area_device_rect:
                # Run a quick one-shot area inference to seed the region
                self._run_area_inference_once()
            if self.auto_area_device_rect:
                self.region_device = dict(self.auto_area_device_rect)
                self.region_logical = self._make_logical_rect(self.region_device)
        if not self.region_device:
            QMessageBox.warning(self, 'No region', 'Select a monitor/sub-area or enable Automatic Area.')
            return
        t_start = time.perf_counter()
        debug_log(f"inference_start region={self.region_device} bypass={self.bypass_yolo_check.isChecked()}")
        self.close_overlays()
        with mss.mss() as sct:
            sshot = capture_screenshot(sct, self.region_device)
        self.screenshot_rgb = screenshot_to_array(sshot)
        t_capture = time.perf_counter()
        debug_log(f"inference_capture_ms={int((t_capture - t_start) * 1000)}")
        w = self.screenshot_rgb.shape[1]
        h = self.screenshot_rgb.shape[0]

        if self.bypass_yolo_check.isChecked():
            self.last_detections = [{'type': 'bbox', 'coords': (0, 0, w, h), 'confidence': 1.0, 'class_id': -1, 'class_name': 'full_image'}]
            print('[on_run_inference] Bypassing YOLO, using full image.')
            debug_log('inference_bypass_yolo=true')
            return

        if not self.model:
            QMessageBox.warning(self, 'No model', 'Load a YOLO model first.')
            return
        print('[on_run_inference] YOLO inference...')
        t_yolo_start = time.perf_counter()
        results = self.model(self.screenshot_rgb, conf=0.25, iou=0.5)
        t_yolo_end = time.perf_counter()
        debug_log(f"inference_yolo_ms={int((t_yolo_end - t_yolo_start) * 1000)}")
        print('[on_run_inference] done.')
        t_parse_start = time.perf_counter()
        self.last_detections = parse_yolo_results(results, w, h)
        t_parse_end = time.perf_counter()
        debug_log(
            f"inference_parse_ms={int((t_parse_end - t_parse_start) * 1000)} "
            f"detections={len(self.last_detections)}"
        )
        print('[on_run_inference] # detections =>', len(self.last_detections))
        enabled_classes = self.get_enabled_classes()
        if enabled_classes is not None:
            before = len(self.last_detections)
            self.last_detections = [d for d in self.last_detections if d['class_name'] in enabled_classes]
            print(f'[on_run_inference] after class-filter ⇒ {len(self.last_detections)} (dropped {before - len(self.last_detections)})')
            debug_log(f"inference_class_filter kept={len(self.last_detections)} dropped={before - len(self.last_detections)}")
        if self.merge_overlap_check.isChecked():
            print('Running iou merger')
            is_alt = self.alt_overlap.isChecked()
            thr = self.merge_iou_slider.value() / 100.0
            t_merge_start = time.perf_counter()
            self.last_detections = self.postProcessMergeByArea(self.last_detections, overlap_threshold=thr, use_alt_merge=is_alt)
            t_merge_end = time.perf_counter()
            debug_log(f"inference_merge_ms={int((t_merge_end - t_merge_start) * 1000)}")

        if self.gradient_overlay_check.isChecked():
            self.gradient_overlay = GradientOverlay(self.region_logical, w, h, self.order_combo.currentData())
        else:
            self.gradient_overlay = None
        self.detection_overlay = DetectionOverlay(self.region_logical, w, h)
        self.detection_overlay.detections = self.last_detections
        self.detection_overlay.update()
        debug_log(f"inference_total_ms={int((time.perf_counter() - t_start) * 1000)}")

    def _run_area_inference_once(self):
        mon, _ = self._current_monitor()
        if not mon:
            return
        with mss.mss() as sct:
            sshot = capture_screenshot(sct, mon, quiet=True)
        img = screenshot_to_array(sshot)
        try:
            results = self.area_model(img, conf=0.25, iou=0.5, verbose=False)
        except Exception as e:
            print('[auto-area] inference error:', e)
            return
        w, h = img.shape[1], img.shape[0]
        dets = parse_yolo_results(results, w, h, quiet=True)
        if not dets:
            return
        # Choose the largest bbox (or bbox of polygon)
        def det_bbox(det):
            if det['type'] == 'bbox':
                return det['coords']
            xs = [p[0] for p in det['coords']]
            ys = [p[1] for p in det['coords']]
            return (min(xs), min(ys), max(xs), max(ys))
        def area_of(b):
            return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        best = max(dets, key=lambda d: area_of(det_bbox(d)))
        x1, y1, x2, y2 = det_bbox(best)
        # Save absolute device rect
        self.auto_area_device_rect = {
            'left': mon['left'] + int(x1),
            'top': mon['top'] + int(y1),
            'width': int(x2 - x1),
            'height': int(y2 - y1),
        }
        self._ensure_auto_area_overlay()

    @pyqtSlot()
    def _on_auto_pre_capture(self):
        if self.auto_area_overlay:
            try:
                self.auto_area_overlay.set_suppressed(True)
                self.auto_area_overlay.repaint()
            except Exception:
                pass

    @pyqtSlot()
    def _on_auto_post_capture(self):
        if self.auto_area_overlay:
            try:
                self.auto_area_overlay.set_suppressed(False)
                self.auto_area_overlay.repaint()
            except Exception:
                pass

    @pyqtSlot(dict)
    def _on_auto_area_result(self, rect_abs):
        self.auto_area_device_rect = dict(rect_abs)
        self._ensure_auto_area_overlay()

    @pyqtSlot()
    def on_translate(self):
        if not self.last_detections:
            print('No detections'); return
        if self.screenshot_rgb is None:
            QMessageBox.information(self, 'No screenshot', 'No screenshot to translate from!'); return
        if self.translation_threads:
            print('[on_translate] Translation in progress – ignoring duplicate call.'); return

        model_entry = self.model_combo.currentData()
        if isinstance(model_entry, tuple) and len(model_entry) == 2:
            provider, model_name = model_entry
        else:
            model_name = (self.model_combo.currentText() or '').strip()
            provider = 'gemini' if model_name.startswith('gemini-') else 'openai'

        if provider == 'gemini':
            if not GEMINI_AVAILABLE:
                QMessageBox.warning(self, 'Gemini missing', 'google.genai not installed!'); return
            if not (read_api_key_from_file() or self._check_api_key()):
                return
        elif provider == 'openai':
            if not OPENAI_AVAILABLE:
                QMessageBox.warning(self, 'OpenAI missing', 'openai not installed!'); return
            if not self._check_openai_key():
                return
        else:
            QMessageBox.warning(self, 'Model error', 'Unknown model provider.'); return

        self._stop_translation_threads()
        self._log_console('-' * 40)
        ordered = self._ordered_detections()
        fast_mode = self.fast_mode_check.isChecked()
        effective_batch = self.batch_processing_check.isChecked()
        batch_size_limit = self.batch_size_spin.value()
        batch_size_used = self._compute_batch_size(len(ordered), batch_size_limit) if effective_batch else 0
        self._debug_log_detections(ordered)
        debug_log(
            f"translate_start provider={provider} model={model_name} "
            f"detections={len(ordered)} batch={effective_batch} batch_size={batch_size_used} "
            f"fast_mode={fast_mode} bypass={self.bypass_yolo_check.isChecked()}"
        )
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
        provider_label = 'Gemini' if provider == 'gemini' else 'OpenAI'

        ctx_enabled = self.context_check.isChecked()
        ctx_num = self.context_spin.value()
        ctx_relevant = self.context_relevant_check.isChecked()
        prev_ctx = "\n".join(self.translation_history[-ctx_num:]) if ctx_enabled and ctx_num > 0 else ''
        debug_log(
            f"translate_context enabled={ctx_enabled} size={ctx_num} relevant={ctx_relevant}"
        )
        if ctx_enabled and ctx_num > 0:
            print('[on_translate] Sending context:\n' + prev_ctx)

        self.translation_threads = []
        if effective_batch:
            self.batch_result_cache = {}
            batch_size = max(1, batch_size_used)
            total_batches = (len(ordered) + batch_size - 1) // batch_size if ordered else 0
            for start in range(0, len(ordered), batch_size):
                chunk = ordered[start:start + batch_size]
                if provider == 'gemini':
                    thr = GeminiBatchTranslationThread(
                        chunk,
                        self.screenshot_rgb,
                        tgt_lang,
                        model_name,
                        previous_context=prev_ctx,
                        use_context=ctx_enabled,
                        context_relevant=ctx_relevant,
                        fast_mode=fast_mode,
                        start_index=start,
                    )
                else:
                    thr = OpenAIBatchTranslationThread(
                        chunk,
                        self.screenshot_rgb,
                        tgt_lang,
                        model_name,
                        previous_context=prev_ctx,
                        use_context=ctx_enabled,
                        context_relevant=ctx_relevant,
                        fast_mode=fast_mode,
                        start_index=start,
                    )
                thr.finished_signal.connect(self.on_batch_results)
                thr.start()
                self.translation_threads.append(thr)
            print(f'[on_translate] Batch {provider_label} request launched ({total_batches} batches).')
            debug_log(
                f"translate_batch_started provider={provider_label} count={len(ordered)} "
                f"batches={total_batches} batch_size={batch_size} max_batch={batch_size_limit}"
            )
        else:
            for idx, det in enumerate(ordered):
                if provider == 'gemini':
                    thr = GeminiTranslationThread(
                        idx,
                        det,
                        self.screenshot_rgb,
                        tgt_lang,
                        model_name,
                        previous_context=prev_ctx,
                        use_context=ctx_enabled,
                        context_relevant=ctx_relevant,
                        fast_mode=fast_mode,
                    )
                else:
                    thr = OpenAITranslationThread(
                        idx,
                        det,
                        self.screenshot_rgb,
                        tgt_lang,
                        model_name,
                        previous_context=prev_ctx,
                        use_context=ctx_enabled,
                        context_relevant=ctx_relevant,
                        fast_mode=fast_mode,
                    )
                thr.finished_signal.connect(self.on_final_result)
                thr.start()
                self.translation_threads.append(thr)
            print(f'[on_translate] Spawned {len(self.translation_threads)} per-crop threads ({provider_label}).')
            debug_log(f"translate_per_crop_started provider={provider_label} count={len(self.translation_threads)}")

    @pyqtSlot(dict, str)
    def on_batch_results(self, mapping, error_msg):
        if error_msg:
            if self.live_trigger_active:
                self.live_trigger_active = False
            self._log_console(error_msg)
            return
        if mapping is None:
            mapping = {}
        debug_log(f"batch_results received={len(mapping)}")
        if not hasattr(self, 'batch_result_cache') or self.batch_result_cache is None:
            self.batch_result_cache = {}

        if self.translation_overlay:
            for idx_str, entry in mapping.items():
                try:
                    idx = int(idx_str)
                except Exception:
                    continue
                if idx < 1 or idx > len(self.translation_overlay.translated_items):
                    continue
                itm = self.translation_overlay.translated_items[idx - 1]
                if isinstance(entry, dict):
                    trans = (entry.get('translation') or '').strip()
                    orig = (entry.get('original') or '').strip()
                else:
                    trans = (entry or '').strip()
                    orig = ''
                self._debug_log_translation(idx, trans, orig)
                itm['text'] = trans
                self._log_console(f"{idx}. {trans}")
                self._add_history(trans, orig)
                self.batch_result_cache[idx] = trans
            self.translation_overlay.update()
        else:
            for idx_str, val in sorted(mapping.items(), key=lambda p: int(p[0])):
                if isinstance(val, dict):
                    trans = (val.get('translation') or '').strip()
                    orig = (val.get('original') or '').strip()
                else:
                    trans = (val or '').strip()
                    orig = ''
                self._debug_log_translation(int(idx_str), trans, orig)
                self._log_console(f"{idx_str}. {trans}")
                self._add_history(trans, orig)
                try:
                    self.batch_result_cache[int(idx_str)] = trans
                except Exception:
                    pass

        self.translation_threads = [t for t in self.translation_threads if t.isRunning()]
        if not self.translation_threads:
            if self.translation_overlay:
                ordered_lines = [itm['text'] for itm in self.translation_overlay.translated_items]
            else:
                ordered_lines = [self.batch_result_cache[k] for k in sorted(self.batch_result_cache)]
            self._start_tts(ordered_lines)
            self._maybe_flag_live_reinfer(ordered_lines)
            self._schedule_live_signature_baseline()

    @pyqtSlot()
    def on_run_all(self):
        print('[on_run_all] Starting combined process.')
        self.on_run_inference()
        self.on_translate()

    @pyqtSlot(int, str)
    def on_final_result(self, det_id, text):
        print(f'[on_final_result] det_id={det_id}, text={repr(text)}')
        clean = text.strip()
        self._debug_log_translation(det_id + 1, clean, "")
        self._log_console(f"{det_id + 1}. {clean}")
        self._add_history(clean)
        if self.translation_overlay:
            self.translation_overlay.update_text_for_id(det_id, clean)

        self.translation_threads = [t for t in self.translation_threads if t.isRunning()]
        if not self.translation_threads and self.translation_overlay:
            lines = [itm['text'] for itm in self.translation_overlay.translated_items]
            self._start_tts(lines)
            self._maybe_flag_live_reinfer(lines)
            self._schedule_live_signature_baseline()

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
        # Intentionally do NOT close auto_area_overlay here. It is independent.

    def periodic_check(self):
        if not self.live_check.isChecked():
            return
        if not self.region_device:
            return
        if not self.live_page_mode and not self.last_detections:
            return
        if self.translation_threads:
            return
        now = time.perf_counter()
        if now - self.live_last_capture_ts < self.live_capture_interval_sec:
            return
        self.live_last_capture_ts = now
        sig = self._capture_text_signature()
        if sig is None:
            return
        diff_threshold = (
            self.live_page_diff_threshold if self.live_page_mode else self.live_diff_threshold
        )
        mode = "page" if self.live_page_mode else "text"
        diff = self._signature_diff(sig, self.live_last_sig)
        if diff is None:
            prev_shape = getattr(self.live_last_sig, "shape", None)
            debug_log(
                f"live_sig_reset shape_cur={sig.shape} shape_prev={prev_shape} mode={mode}"
            )
            self.live_last_sig = sig
            self.live_last_trigger_sig = sig
            self.live_last_change_ts = now
            self.live_last_change_diff = 1.0
            self.live_reinfer_next = True
            return
        if diff > diff_threshold:
            self.live_last_sig = sig
            self.live_last_change_ts = now
            self.live_last_change_diff = diff
            self.live_reinfer_next = self.live_page_mode or diff >= self.live_reinfer_threshold
            debug_log(f"live_change_detected diff={diff:.3f} mode={mode}")
            return
        self.live_last_sig = sig
        if now - self.live_last_change_ts < self.live_debounce_sec:
            return
        if self.translation_threads:
            return
        trigger_diff = self._signature_diff(sig, self.live_last_trigger_sig)
        if trigger_diff is None:
            prev_shape = getattr(self.live_last_trigger_sig, "shape", None)
            debug_log(
                f"live_trigger_reset shape_cur={sig.shape} shape_prev={prev_shape} mode={mode}"
            )
            self.live_last_trigger_sig = sig
            return
        if trigger_diff <= diff_threshold:
            return
        self.live_last_trigger_sig = sig
        reinfer = self.live_reinfer_next
        debug_log(
            f"live_trigger diff={self.live_last_change_diff:.3f} reinfer={int(reinfer)} mode={mode}"
        )
        self.live_reinfer_next = False
        if reinfer:
            self._live_run_inference_and_translate()
        else:
            self._live_translate()

    # removed: timer-based auto area tick (replaced by worker thread)

    @pyqtSlot(bool)
    def on_debug_toggle(self, enabled):
        if enabled:
            set_debug_logging(True)
            self.debug_btn.setText("Debug: ON")
            debug_log("debug_enabled")
        else:
            debug_log("debug_disabled")
            set_debug_logging(False)
            self.debug_btn.setText("Debug: OFF")

    @pyqtSlot(int)
    def on_live_toggle(self, _):
        enabled = self.live_check.isChecked()
        self.live_last_sig = None
        self.live_last_trigger_sig = None
        self.live_last_change_ts = time.perf_counter()
        self.live_last_capture_ts = 0.0
        self.live_last_change_diff = 0.0
        self.live_reinfer_next = False
        self.live_trigger_active = False
        mode = "page" if self.live_page_mode else "text"
        debug_log(f"live_enabled={enabled} mode={mode}")
        if enabled and not self.translation_threads:
            self._schedule_live_signature_baseline()

    @pyqtSlot(int)
    def on_fast_mode_toggle(self, _):
        enabled = self.fast_mode_check.isChecked()
        debug_log(f"fast_mode enabled={enabled}")

    @pyqtSlot(int)
    def on_batch_toggle(self, _):
        enabled = self.batch_processing_check.isChecked()
        self.batch_size_spin.setEnabled(enabled)
        debug_log(f"batch_mode enabled={enabled} max_size={self.batch_size_spin.value()}")

    @pyqtSlot(int)
    def on_chatterbox_tts_toggle(self, _):
        enabled = self.chatterbox_tts_check.isChecked()
        self.chatterbox_voice_combo.setEnabled(enabled)
        if enabled:
            self._populate_chatterbox_voices()

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

    def _check_openai_key(self):
        key = self.openai_api_key or read_openai_key_from_file()
        if key:
            self.openai_api_key = key
            return True
        text, ok = QInputDialog.getText(
            self,
            'Enter API Key',
            'OpenAI API key:'
        )
        if ok and text.strip():
            path = os.path.join(os.path.dirname(__file__), 'openai_api_key.txt')
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text.strip())
                self.openai_api_key = text.strip()
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
        self._stop_auto_area_worker()
        if self.region_select_overlay:
            self.region_select_overlay.close()
            self.region_select_overlay = None
        if self.auto_area_overlay:
            try:
                self.auto_area_overlay.close()
            except Exception:
                pass
            self.auto_area_overlay = None
        event.accept()


def main():
    print('[main] Starting final app: polygons w/ clipped + centered text.')
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

