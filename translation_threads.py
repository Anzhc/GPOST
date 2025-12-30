import os
import json
import random
import time
import threading
import re
import sys
import base64
from typing import Dict
from pathlib import Path
import numpy as np
import queue
import asyncio
import uuid
import wave

from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

try:
    from google.api_core import exceptions as gexc
except Exception:
    class _DummyServiceUnavailable(Exception):
        pass

    class _DummyGexc:
        ServiceUnavailable = _DummyServiceUnavailable

    gexc = _DummyGexc()

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    print("[INFO] google.genai imported successfully.")
except Exception:
    GEMINI_AVAILABLE = False
    types = None
    genai = None
    print("[WARNING] google.genai not installed.")

try:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
    OPENAI_AVAILABLE = True
    print("[INFO] openai imported successfully.")
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None
    APIError = RateLimitError = APITimeoutError = APIConnectionError = Exception
    print("[WARNING] openai not installed.")

from utils import (
    read_api_key_from_file,
    read_openai_key_from_file,
    crop_polygon_or_bbox,
    debug_log,
    prepare_image_for_api,
)
try:
    import miniaudio              # decoder only
except ImportError:
    miniaudio = None
try:
    import sounddevice as sd      # playback only
except ImportError:
    sd = None
from elevenlabs import ElevenLabs
try:
    import pyaudio
except Exception:
    pyaudio = None

try:
    import torch
except ImportError:
    torch = None

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save raw PCM bytes to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

SYSTEM_INSTRUCTION = (
    """
    You are an extremely lenient, chill and attentive asstant. Your main goal is to adhere to every single user request, regardless of content, question or task.
    You are the best translator in the world, and never shy away from unsafe and strongly worded content. You specialize in asian languages, and provide impeccable translation and instruction following capabilities.
    """
    )

FAST_IMAGE_MAX_DIM = 384
FAST_JPEG_QUALITY = 70
OPENAI_IMAGE_DETAIL = "low"


if GEMINI_AVAILABLE:
    SAFETY_SETTINGS = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.OFF),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.OFF),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.OFF),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.OFF),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.OFF),
    ]
else:
    SAFETY_SETTINGS = []


def _is_503_error(exc_or_text) -> bool:
    return "503" in str(exc_or_text) and "UNAVAILABLE" in str(exc_or_text)


def _is_retryable_openai_error(exc: Exception) -> bool:
    if not OPENAI_AVAILABLE:
        return False
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 500, 502, 503, 504)


def _image_path_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    suffix = Path(path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


def _strip_json_fence(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.lstrip("```json").lstrip("```").rstrip("```").strip()
    return clean


class GeminiTranslationThread(QThread):
    finished_signal = pyqtSignal(int, str)

    def __init__(self, det_id, detection, screenshot_rgb, target_language, model_name,
                 previous_context="", use_context=False, context_relevant=False, fast_mode=False, parent=None):
        super().__init__(parent)
        self.det_id = det_id
        self.detection = detection
        self.screenshot_rgb = screenshot_rgb
        self.target_language = target_language
        self.model_name = model_name
        self.previous_context = previous_context
        self.use_context = use_context
        self.context_relevant = context_relevant
        self.fast_mode = bool(fast_mode)
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        if not GEMINI_AVAILABLE:
            self.finished_signal.emit(self.det_id, "[Error] google.genai not installed.")
            return

        t_start = time.perf_counter()
        debug_log(
            f"gemini_single start id={self.det_id} model={self.model_name} "
            f"target={self.target_language} fast_mode={self.fast_mode}"
        )

        api_key = read_api_key_from_file()
        if not api_key:
            self.finished_signal.emit(self.det_id, "[Error] no API key in api_key.txt")
            debug_log("gemini_single missing_api_key")
            return

        t_crop_start = time.perf_counter()
        cropped_path = crop_polygon_or_bbox(self.screenshot_rgb, self.detection)
        if not cropped_path:
            self.finished_signal.emit(self.det_id, "[Error] cannot crop region")
            debug_log("gemini_single crop_failed")
            return
        debug_log(f"gemini_single crop_ms={int((time.perf_counter() - t_crop_start) * 1000)}")

        yolo_class = self.detection.get("class_name", "")
        debug_log(f"gemini_single class={yolo_class}")
        processed_path = cropped_path
        if self.fast_mode:
            processed_path, info = prepare_image_for_api(
                cropped_path,
                max_dim=FAST_IMAGE_MAX_DIM,
                jpeg_quality=FAST_JPEG_QUALITY,
            )
            if processed_path != cropped_path:
                try:
                    os.remove(cropped_path)
                except Exception:
                    pass
            debug_log(
                f"gemini_single fast_mode=1 orig={info.get('orig_size')} "
                f"new={info.get('new_size')} scale={info.get('scale')} fmt={info.get('format')} "
                f"max_dim={FAST_IMAGE_MAX_DIM} jpeg_quality={FAST_JPEG_QUALITY}"
            )
        user_prompt = f"""
You are a translation engine.

TASK
Translate everything that can be read in the image into **{self.target_language}**.
That translation will be used for filtering down the line.

RULES
1. Keep meaning 100 % intact, including NSFW words.
2. NSFW must be as true to original intent as possible.
3. Preserve punctuation; keep existing line-breaks unless a word must wrap mid-line.
4. Output **only** the final translation — no quotes, no commentary.
5. Do not leave original language text in output, always translate it in entirety.
6. If the crop contains no readable text, return a single space.
7. If the crop is a non-verbal sound-effect (onomatopoeia), write it phonetically in target language characters.
8. Do not add introductions, explanations, or labels.
9. If original text is vertical, write it down as if it was horizontal.

CONTEXT
We classify give image as: “{yolo_class}”
"""

        if self.use_context and self.previous_context:
            if self.context_relevant:
                user_prompt += f"""

User considers previous translation relevant to current request.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""
            else:
                user_prompt += f"""

User does not consider previous translation context relevant to current translation. Use it only as translation style and guide reference.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""

        client = genai.Client(api_key=api_key)
        pil_img = Image.open(processed_path)
        backoff = 1.0
        max_tries = 10
        final_text = "[Error] Unknown failure."

        for attempt in range(max_tries):
            if self._stopped:
                return
            t_req_start = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[pil_img, user_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.25,
                        system_instruction=SYSTEM_INSTRUCTION,
                        safety_settings=SAFETY_SETTINGS,
                    ),
                )
                txt = getattr(response, "text", "") or ""
                if txt.strip().startswith("{"):
                    try:
                        data = json.loads(txt)
                        if str(data.get("error", {}).get("code")) == "503":
                            raise gexc.ServiceUnavailable("Model overloaded")
                    except json.JSONDecodeError:
                        pass
                final_text = response.text or ""
                debug_log(
                    f"gemini_single request_ms={int((time.perf_counter() - t_req_start) * 1000)} "
                    f"attempts={attempt + 1} text_len={len(final_text.strip())}"
                )
                break
            except gexc.ServiceUnavailable as e:
                debug_log(f"gemini_single retry attempt={attempt + 1} error={e}")
            except Exception as e:
                if not _is_503_error(e):
                    final_text = f"[GeminiTranslationThread error => {e}]"
                    debug_log(f"gemini_single error attempt={attempt + 1} error={e}")
                    break
                debug_log(f"gemini_single retry attempt={attempt + 1} error={e}")
            if attempt == max_tries - 1:
                final_text = "[Error] Gemini model overloaded after retries.]"
                debug_log("gemini_single error attempt=max retries")
                break
            sleep_for = backoff + random.uniform(0, 0.5)
            time.sleep(sleep_for)
            backoff *= 2

        try:
            os.remove(processed_path)
        except Exception:
            pass

        debug_log(f"gemini_single total_ms={int((time.perf_counter() - t_start) * 1000)} text_len={len(final_text.strip())}")
        self.finished_signal.emit(self.det_id, final_text)


class GeminiBatchTranslationThread(QThread):
    finished_signal = pyqtSignal(dict, str)

    def __init__(self, detections, screenshot_rgb, target_language, model_name,
                 previous_context="", use_context=False, context_relevant=False,
                 fast_mode=False, start_index=0, parent=None):
        super().__init__(parent)
        self.detections = detections
        self.screenshot_rgb = screenshot_rgb
        self.target_language = target_language
        self.model_name = model_name
        self.previous_context = previous_context
        self.use_context = use_context
        self.context_relevant = context_relevant
        self.fast_mode = bool(fast_mode)
        self.start_index = int(start_index)
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        if not GEMINI_AVAILABLE or self._stopped:
            self.finished_signal.emit({}, "")
            return
        t_start = time.perf_counter()
        debug_log(
            f"gemini_batch start model={self.model_name} target={self.target_language} "
            f"count={len(self.detections)} fast_mode={self.fast_mode} start_index={self.start_index}"
        )
        api_key = read_api_key_from_file()
        if not api_key:
            self.finished_signal.emit({}, "")
            debug_log("gemini_batch missing_api_key")
            return

        crop_paths, crop_imgs, idx_map = [], [], []
        fast_scales = []
        t_crop_start = time.perf_counter()
        for i, det in enumerate(self.detections, start=1):
            if self._stopped:
                self.finished_signal.emit({}, "")
                return
            pth = crop_polygon_or_bbox(self.screenshot_rgb, det)
            if pth:
                use_path = pth
                if self.fast_mode:
                    use_path, info = prepare_image_for_api(
                        pth,
                        max_dim=FAST_IMAGE_MAX_DIM,
                        jpeg_quality=FAST_JPEG_QUALITY,
                    )
                    if use_path != pth:
                        try:
                            os.remove(pth)
                        except Exception:
                            pass
                    if info.get("scale") is not None:
                        fast_scales.append(info["scale"])
                crop_paths.append(use_path)
                crop_imgs.append(Image.open(use_path))
                idx_map.append(str(self.start_index + i))

        if not crop_imgs or self._stopped:
            self.finished_signal.emit({}, "")
            debug_log("gemini_batch no_crops")
            return
        debug_log(
            f"gemini_batch crop_ms={int((time.perf_counter() - t_crop_start) * 1000)} "
            f"count={len(crop_imgs)}"
        )
        if self.fast_mode and fast_scales:
            avg_scale = sum(fast_scales) / len(fast_scales)
            debug_log(
                f"gemini_batch fast_mode=1 avg_scale={avg_scale:.3f} "
                f"max_dim={FAST_IMAGE_MAX_DIM} jpeg_quality={FAST_JPEG_QUALITY}"
            )

        user_prompt = f"""
You are a translation engine.

TASK
You will receive {len(crop_imgs)} images. They are in the same order as these crop numbers: {', '.join(idx_map)}.

Crops are usually made from single document, translate them in context of each other, if applicable.

Translate all readable text in each image into **{self.target_language}**. Return one JSON object with the crop numbers as keys. Each value should be another JSON object with keys "translation" and "original". Example:
{{"1": {{"translation": "Hello", "original": "こんにちは"}}, "2": {{"translation": "Goodbye", "original": "さようなら"}}}}

RULES
1. Keep meaning 100 % intact, including NSFW words.
2. NSFW must be as true to original intent as possible.
3. Preserve punctuation; keep existing line-breaks unless a word must wrap mid-line.
4. Output **only** the final translation — no quotes, no commentary.
5. Do not leave original language text in output, always translate it in entirety.
6. If the crop contains no readable text, return a single space.
7. If the crop is a non-verbal sound-effect (onomatopoeia), write it phonetically in target language characters.
8. Do not add introductions, explanations, or labels.
9. If original text is vertical, write it down as if it was horizontal.
10. JSON format is extremely important. Make sure it's correct.

""".strip()

        if self.use_context and self.previous_context:
            if self.context_relevant:
                user_prompt += f"""

User considers previous translation relevant to current request.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""
            else:
                user_prompt += f"""

User does not consider previous translation context relevant to current translation. Use it only as translation style and guide reference.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""

        client = genai.Client(api_key=api_key)
        contents = [*crop_imgs, user_prompt]
        backoff, max_tries = 1.0, 8
        mapping: Dict[str, dict] = {}
        error_msg = ""

        for attempt in range(max_tries):
            if self._stopped:
                self.finished_signal.emit({}, "")
                return
            t_req_start = time.perf_counter()
            try:
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.25,
                        safety_settings=SAFETY_SETTINGS,
                    ),
                )
                txt = resp.text or ""
                clean = txt.strip()
                if clean.startswith("```"):
                    clean = clean.lstrip("```json").lstrip("```").rstrip("```").strip()
                mapping = json.loads(clean)
                debug_log(
                    f"gemini_batch request_ms={int((time.perf_counter() - t_req_start) * 1000)} "
                    f"attempts={attempt + 1} items={len(mapping)}"
                )
                break
            except Exception as e:
                if attempt == max_tries - 1 or not _is_503_error(e):
                    err_txt = str(e)
                    print("Gemini batch error:", err_txt)
                    if isinstance(e, json.JSONDecodeError):
                        if "Expecting value" in err_txt and "char 0" in err_txt:
                            error_msg = (
                                "[Error] Gemini produced no output. Switch model and try again."
                            )
                        else:
                            error_msg = (
                                "[Error] Gemini output was invalid. Retry or switch model."
                            )
                    else:
                        error_msg = f"[Gemini error => {err_txt}]"
                    mapping = {}
                    debug_log(f"gemini_batch error attempt={attempt + 1} error={err_txt}")
                    break
                debug_log(f"gemini_batch retry attempt={attempt + 1} error={e}")
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2

        debug_log(f"gemini_batch total_ms={int((time.perf_counter() - t_start) * 1000)} items={len(mapping)}")

        for p in crop_paths:
            try:
                os.remove(p)
            except Exception:
                pass

        self.finished_signal.emit(mapping, error_msg)


class OpenAITranslationThread(QThread):
    finished_signal = pyqtSignal(int, str)

    def __init__(self, det_id, detection, screenshot_rgb, target_language, model_name,
                 previous_context="", use_context=False, context_relevant=False, fast_mode=False, parent=None):
        super().__init__(parent)
        self.det_id = det_id
        self.detection = detection
        self.screenshot_rgb = screenshot_rgb
        self.target_language = target_language
        self.model_name = model_name
        self.previous_context = previous_context
        self.use_context = use_context
        self.context_relevant = context_relevant
        self.fast_mode = bool(fast_mode)
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        if not OPENAI_AVAILABLE:
            self.finished_signal.emit(self.det_id, "[Error] openai not installed.")
            debug_log("openai_single missing_openai_module")
            return

        t_start = time.perf_counter()
        debug_log(
            f"openai_single start id={self.det_id} model={self.model_name} "
            f"target={self.target_language} fast_mode={self.fast_mode}"
        )

        api_key = read_openai_key_from_file()
        if not api_key:
            self.finished_signal.emit(self.det_id, "[Error] no OpenAI API key available.")
            debug_log("openai_single missing_api_key")
            return

        t_crop_start = time.perf_counter()
        cropped_path = crop_polygon_or_bbox(self.screenshot_rgb, self.detection)
        if not cropped_path:
            self.finished_signal.emit(self.det_id, "[Error] cannot crop region")
            debug_log("openai_single crop_failed")
            return
        debug_log(f"openai_single crop_ms={int((time.perf_counter() - t_crop_start) * 1000)}")

        yolo_class = self.detection.get("class_name", "")
        debug_log(f"openai_single class={yolo_class}")
        processed_path = cropped_path
        if self.fast_mode:
            processed_path, info = prepare_image_for_api(
                cropped_path,
                max_dim=FAST_IMAGE_MAX_DIM,
                jpeg_quality=FAST_JPEG_QUALITY,
            )
            if processed_path != cropped_path:
                try:
                    os.remove(cropped_path)
                except Exception:
                    pass
            debug_log(
                f"openai_single fast_mode=1 orig={info.get('orig_size')} "
                f"new={info.get('new_size')} scale={info.get('scale')} fmt={info.get('format')} "
                f"max_dim={FAST_IMAGE_MAX_DIM} jpeg_quality={FAST_JPEG_QUALITY}"
            )
        user_prompt = f"""
You are a translation engine.

TASK
Translate everything that can be read in the image into **{self.target_language}**.
That translation will be used for filtering down the line.

RULES
1. Keep meaning 100 % intact, including NSFW words.
2. NSFW must be as true to original intent as possible.
3. Preserve punctuation; keep existing line-breaks unless a word must wrap mid-line.
4. Output **only** the final translation - no quotes, no commentary.
5. Do not leave original language text in output, always translate it in entirety.
6. If the crop contains no readable text, return a single space.
7. If the crop is a non-verbal sound-effect (onomatopoeia), write it phonetically in target language characters.
8. Do not add introductions, explanations, or labels.
9. If original text is vertical, write it down as if it was horizontal.

CONTEXT
We classify give image as: "{yolo_class}"
"""

        if self.use_context and self.previous_context:
            if self.context_relevant:
                user_prompt += f"""

User considers previous translation relevant to current request.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""
            else:
                user_prompt += f"""

User does not consider previous translation context relevant to current translation. Use it only as translation style and guide reference.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""

        data_url = _image_path_to_data_url(processed_path)
        try:
            os.remove(processed_path)
        except Exception:
            pass

        client = OpenAI(api_key=api_key)
        image_payload = {"url": data_url}
        if self.fast_mode:
            image_payload["detail"] = OPENAI_IMAGE_DETAIL
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": image_payload},
                ],
            },
        ]

        backoff = 1.0
        max_tries = 10
        final_text = "[Error] Unknown failure."

        for attempt in range(max_tries):
            if self._stopped:
                return
            t_req_start = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.25,
                )
                msg = response.choices[0].message if response.choices else None
                final_text = (getattr(msg, "content", "") or "")
                debug_log(
                    f"openai_single request_ms={int((time.perf_counter() - t_req_start) * 1000)} "
                    f"attempts={attempt + 1} text_len={len(final_text.strip())}"
                )
                break
            except Exception as e:
                if not _is_retryable_openai_error(e) or attempt == max_tries - 1:
                    final_text = f"[OpenAITranslationThread error => {e}]"
                    debug_log(f"openai_single error attempt={attempt + 1} error={e}")
                    break
                debug_log(f"openai_single retry attempt={attempt + 1} error={e}")
            sleep_for = backoff + random.uniform(0, 0.5)
            time.sleep(sleep_for)
            backoff *= 2

        debug_log(f"openai_single total_ms={int((time.perf_counter() - t_start) * 1000)} text_len={len(final_text.strip())}")
        self.finished_signal.emit(self.det_id, final_text)


class OpenAIBatchTranslationThread(QThread):
    finished_signal = pyqtSignal(dict, str)

    def __init__(self, detections, screenshot_rgb, target_language, model_name,
                 previous_context="", use_context=False, context_relevant=False,
                 fast_mode=False, start_index=0, parent=None):
        super().__init__(parent)
        self.detections = detections
        self.screenshot_rgb = screenshot_rgb
        self.target_language = target_language
        self.model_name = model_name
        self.previous_context = previous_context
        self.use_context = use_context
        self.context_relevant = context_relevant
        self.fast_mode = bool(fast_mode)
        self.start_index = int(start_index)
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        if not OPENAI_AVAILABLE or self._stopped:
            self.finished_signal.emit({}, "")
            return
        t_start = time.perf_counter()
        debug_log(
            f"openai_batch start model={self.model_name} target={self.target_language} "
            f"count={len(self.detections)} fast_mode={self.fast_mode} start_index={self.start_index}"
        )
        api_key = read_openai_key_from_file()
        if not api_key:
            self.finished_signal.emit({}, "")
            debug_log("openai_batch missing_api_key")
            return

        crop_paths, data_urls, idx_map = [], [], []
        fast_scales = []
        t_crop_start = time.perf_counter()
        for i, det in enumerate(self.detections, start=1):
            if self._stopped:
                self.finished_signal.emit({}, "")
                return
            pth = crop_polygon_or_bbox(self.screenshot_rgb, det)
            if pth:
                use_path = pth
                if self.fast_mode:
                    use_path, info = prepare_image_for_api(
                        pth,
                        max_dim=FAST_IMAGE_MAX_DIM,
                        jpeg_quality=FAST_JPEG_QUALITY,
                    )
                    if use_path != pth:
                        try:
                            os.remove(pth)
                        except Exception:
                            pass
                    if info.get("scale") is not None:
                        fast_scales.append(info["scale"])
                crop_paths.append(use_path)
                data_urls.append(_image_path_to_data_url(use_path))
                idx_map.append(str(self.start_index + i))

        if not data_urls or self._stopped:
            self.finished_signal.emit({}, "")
            debug_log("openai_batch no_crops")
            return
        debug_log(
            f"openai_batch crop_ms={int((time.perf_counter() - t_crop_start) * 1000)} "
            f"count={len(data_urls)}"
        )
        if self.fast_mode and fast_scales:
            avg_scale = sum(fast_scales) / len(fast_scales)
            debug_log(
                f"openai_batch fast_mode=1 avg_scale={avg_scale:.3f} "
                f"max_dim={FAST_IMAGE_MAX_DIM} jpeg_quality={FAST_JPEG_QUALITY}"
            )

        user_prompt = f"""
You are a translation engine.

TASK
You will receive {len(data_urls)} images. They are in the same order as these crop numbers: {', '.join(idx_map)}.

Crops are usually made from single document, translate them in context of each other, if applicable.

Translate all readable text in each image into **{self.target_language}**. Return one JSON object with the crop numbers as keys. Each value should be another JSON object with keys "translation" and "original". Example:
{{"1": {{"translation": "Hello", "original": "Hello"}}, "2": {{"translation": "Goodbye", "original": "Goodbye"}}}}

RULES
1. Keep meaning 100 % intact, including NSFW words.
2. NSFW must be as true to original intent as possible.
3. Preserve punctuation; keep existing line-breaks unless a word must wrap mid-line.
4. Output **only** the final translation - no quotes, no commentary.
5. Do not leave original language text in output, always translate it in entirety.
6. If the crop contains no readable text, return a single space.
7. If the crop is a non-verbal sound-effect (onomatopoeia), write it phonetically in target language characters.
8. Do not add introductions, explanations, or labels.
9. If original text is vertical, write it down as if it was horizontal.
10. JSON format is extremely important. Make sure it's correct.

""".strip()

        if self.use_context and self.previous_context:
            if self.context_relevant:
                user_prompt += f"""

User considers previous translation relevant to current request.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""
            else:
                user_prompt += f"""

User does not consider previous translation context relevant to current translation. Use it only as translation style and guide reference.

Here is the context of previous translations, arranged in a [translation - original] format:
{self.previous_context}
"""

        client = OpenAI(api_key=api_key)
        content = [{"type": "text", "text": user_prompt}]
        for url in data_urls:
            image_payload = {"url": url}
            if self.fast_mode:
                image_payload["detail"] = OPENAI_IMAGE_DETAIL
            content.append({"type": "image_url", "image_url": image_payload})
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": content},
        ]

        backoff, max_tries = 1.0, 8
        mapping: Dict[str, dict] = {}
        error_msg = ""
        use_response_format = True

        for attempt in range(max_tries):
            if self._stopped:
                self.finished_signal.emit({}, "")
                return
            t_req_start = time.perf_counter()
            try:
                kwargs = {}
                if use_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.25,
                    **kwargs,
                )
                txt = resp.choices[0].message.content if resp.choices else ""
                clean = _strip_json_fence(txt or "")
                mapping = json.loads(clean)
                debug_log(
                    f"openai_batch request_ms={int((time.perf_counter() - t_req_start) * 1000)} "
                    f"attempts={attempt + 1} items={len(mapping)} response_format={use_response_format}"
                )
                break
            except json.JSONDecodeError as e:
                err_txt = str(e)
                if "Expecting value" in err_txt and "char 0" in err_txt:
                    error_msg = (
                        "[Error] OpenAI produced no output. Switch model and try again."
                    )
                else:
                    error_msg = (
                        "[Error] OpenAI output was invalid. Retry or switch model."
                    )
                mapping = {}
                debug_log(f"openai_batch error attempt={attempt + 1} error={err_txt}")
                break
            except Exception as e:
                err_txt = str(e)
                if use_response_format and "response_format" in err_txt:
                    use_response_format = False
                    debug_log("openai_batch response_format_unsupported")
                    continue
                if attempt == max_tries - 1 or not _is_retryable_openai_error(e):
                    print("OpenAI batch error:", err_txt)
                    error_msg = f"[OpenAI error => {err_txt}]"
                    mapping = {}
                    debug_log(f"openai_batch error attempt={attempt + 1} error={err_txt}")
                    break
                debug_log(f"openai_batch retry attempt={attempt + 1} error={err_txt}")
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2

        debug_log(f"openai_batch total_ms={int((time.perf_counter() - t_start) * 1000)} items={len(mapping)}")

        for p in crop_paths:
            try:
                os.remove(p)
            except Exception:
                pass

        self.finished_signal.emit(mapping, error_msg)


class ElevenLabsTTSThread(QThread):
    """Generate speech from lines using ElevenLabs API and play them."""

    finished_signal = pyqtSignal()

    def __init__(self, lines: list[str], api_key: str, voice_id: str | None = None, parent=None):
        super().__init__(parent)
        self.lines = lines
        self.api_key = api_key
        self.voice_id = voice_id or ""
        self._stopped = False

        # ── NEW: serialise playback via a queue + 1 worker
        self._play_q: queue.Queue[
            tuple[bytes, int, int] | None
        ] = queue.Queue()  # (pcm_bytes, nch, sr)  or  None sentinel
        self._play_worker = threading.Thread(
            target=self._playback_loop, daemon=True
        )

    # ──────────────────────────────────────────────────────
    #  Playback loop  (runs in a daemon thread)
    # ──────────────────────────────────────────────────────
    def _playback_loop(self):
        if sd is None:
            print("[ElevenLabsTTSThread] sounddevice not available")
            return
        while True:
            item = self._play_q.get()
            if item is None:            # sentinel → tidy shutdown
                break
            pcm_bytes, nch, sr = item
            try:
                audio = np.frombuffer(pcm_bytes, dtype=np.int16)
                if nch > 1:
                    audio = audio.reshape(-1, nch)
                sd.play(audio, sr)
                sd.wait()               # sequential: block *inside* worker
            except Exception as e:
                print("[ElevenLabsTTSThread] playback error:", e)

    # ──────────────────────────────────────────────────────
    #  Queue helper
    # ──────────────────────────────────────────────────────
    def _enqueue_play(self, pcm_samples: bytes, nch: int, sr: int):
        """Add one buffer to the playback queue and start worker if idle."""
        if not self._play_worker.is_alive():
            self._play_worker.start()
        self._play_q.put((pcm_samples, nch, sr))

    # ──────────────────────────────────────────────────────
    #  Public controls
    # ──────────────────────────────────────────────────────
    def stop(self):
        self._stopped = True
        if sd:
            sd.stop()
        # unblock and terminate playback thread
        self._play_q.put(None)

    # ──────────────────────────────────────────────────────
    #  Worker body
    # ──────────────────────────────────────────────────────
    def run(self):
        if not self.api_key or self._stopped:
            self.finished_signal.emit()
            return

        client = ElevenLabs(api_key=self.api_key)
        vid = self.voice_id or "pjcYQlDFKMbcOUp6F5GD"

        for line in self.lines:
            if self._stopped:
                break

            # ──────────────────────────────────────────────
            # Skip empty / whitespace-only inputs
            # ──────────────────────────────────────────────
            if not line or not line.strip():
                print("[ElevenLabsTTSThread] blank input; skipping")
                continue

            try:
                # 1️⃣ Ask for MP3 (free tier: 44.1 kHz / 128 kbps)
                audio = client.text_to_speech.convert(
                    text=line,
                    voice_id=vid,
                    model_id="eleven_flash_v2_5",
                    output_format="mp3_44100_128",
                )
                mp3_bytes = (
                    b"".join(audio) if hasattr(audio, "__iter__") else audio or b""
                )

                # 2️⃣ Decode MP3 → 16-bit PCM with miniaudio
                decoded = miniaudio.mp3_read_s16(mp3_bytes)
                pcm_bytes = decoded.samples.tobytes()
                nch, sr = decoded.nchannels, decoded.sample_rate  # 1, 44100

                # 3️⃣ Queue for playback (non-blocking for run())
                if not self._stopped:
                    self._enqueue_play(pcm_bytes, nch, sr)

            except Exception as e:
                print("[ElevenLabsTTSThread] error:", e)

        # tell playback loop to exit after queue drains
        self._play_q.put(None)
        self.finished_signal.emit()




class ChatterboxTTSThread(QThread):
    """Generate speech batches via Chatterbox and stream playback."""

    finished_signal = pyqtSignal()

    _model_lock = threading.Lock()
    _inference_lock = threading.Lock()
    _model_cache = {}

    def __init__(self, lines: list[str], voice_path: str | None = None, exaggeration: float = 0.5, parent=None):
        super().__init__(parent)
        self.lines = lines
        self.voice_path = voice_path or ''
        self.exaggeration = exaggeration
        self._stopped = False
        self._sample_rate = 24000
        self._output_path = ''
        self._segments = []
        self._play_q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._play_worker = threading.Thread(target=self._playback_loop, daemon=True)

    @classmethod
    def _ensure_model(cls):
        if torch is None:
            print('[ChatterboxTTSThread] torch not available; cannot run Chatterbox TTS.')
            return None
        device = 'cpu'
        try:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
        except Exception:
            device = 'cpu'
        with cls._model_lock:
            cached = cls._model_cache.get(device)
            if cached is not None:
                return cached
            cb_src = Path(__file__).resolve().parent / 'chatterbox' / 'src'
            if cb_src.exists():
                cb_src_str = str(cb_src)
                if cb_src_str not in sys.path:
                    sys.path.insert(0, cb_src_str)
            try:
                from chatterbox.tts import ChatterboxTTS  # type: ignore
            except Exception as exc:
                print(f'[ChatterboxTTSThread] import error: {exc}')
                cls._model_cache[device] = None
                return None
            try:
                model = ChatterboxTTS.from_pretrained(device=device)
                cls._model_cache[device] = model
                print(f'[ChatterboxTTSThread] loaded Chatterbox on {device}')
            except Exception as exc:
                print(f'[ChatterboxTTSThread] load error: {exc}')
                cls._model_cache[device] = None
                model = None
            return model

    @staticmethod
    def _prepare_batches(lines: list[str]) -> list[str]:
        joined = ' '.join(seg.strip() for seg in lines if seg and seg.strip())
        if not joined:
            return []
        clean = re.sub(r'\s+', ' ', joined).strip()
        if not clean:
            return []
        raw = re.split(r'(?<=[\.\!\?。！？])\s+|[\r\n]+', clean)

        batches: list[str] = []
        pending = ''
        pending_words = 0
        for segment in raw:
            seg = segment.strip()
            if not seg:
                continue
            words = seg.split()
            if len(words) >= 10 and pending_words == 0:
                batches.append(seg)
                continue
            if not pending:
                pending = seg
                pending_words = len(words)
            else:
                pending = f"{pending} {seg}".strip()
                pending_words += len(words)
            if pending_words >= 10:
                batches.append(pending)
                pending = ''
                pending_words = 0
        if pending:
            batches.append(pending)
        return batches

    def _ensure_worker(self):
        if sd is None:
            return False
        if not self._play_worker.is_alive():
            self._play_worker.start()
        return True

    def _enqueue_play(self, audio: np.ndarray):
        if sd is None:
            return
        if self._ensure_worker():
            self._play_q.put(audio.astype(np.float32))

    def _playback_loop(self):
        if sd is None:
            return
        while True:
            item = self._play_q.get()
            if item is None:
                break
            try:
                sd.play(item, self._sample_rate)
                sd.wait()
            except Exception as exc:
                print(f'[ChatterboxTTSThread] playback error: {exc}')

    def stop(self):
        self._stopped = True
        if sd:
            try:
                sd.stop()
            except Exception:
                pass
        try:
            self._play_q.put_nowait(None)
        except Exception:
            pass

    def _compose_and_save(self):
        if not self._segments:
            return
        combined = np.concatenate(self._segments)
        pcm = np.clip(combined, -1.0, 1.0)
        pcm_i16 = (pcm * 32767).astype(np.int16)
        tts_dir = Path(__file__).resolve().parent / 'TTS'
        tts_dir.mkdir(exist_ok=True)
        out_path = tts_dir / f'chatterbox_{uuid.uuid4().hex}.wav'
        try:
            wave_file(str(out_path), pcm_i16.tobytes(), channels=1, rate=self._sample_rate)
            self._output_path = str(out_path)
            print(f'[ChatterboxTTSThread] saved audio to {out_path}')
        except Exception as exc:
            print(f'[ChatterboxTTSThread] save error: {exc}')

    def run(self):
        model = self._ensure_model()
        if model is None or self._stopped:
            self.finished_signal.emit()
            return
        self._sample_rate = getattr(model, 'sr', 24000)
        batches = self._prepare_batches(self.lines)
        if not batches:
            self.finished_signal.emit()
            return

        print(f'[ChatterboxTTSThread] generating {len(batches)} batches')
        first = True
        try:
            with self.__class__._inference_lock:
                for text in batches:
                    if self._stopped:
                        break
                    try:
                        wav = model.generate(
                            text,
                            audio_prompt_path=self.voice_path if first and self.voice_path else None,
                            exaggeration=self.exaggeration,
                        )
                        first = False
                    except Exception as exc:
                        print(f'[ChatterboxTTSThread] generation error: {exc}')
                        break
                    if torch is not None and hasattr(wav, 'detach'):
                        audio_np = wav.detach().cpu().numpy()
                    else:
                        audio_np = np.asarray(wav)
                    audio = np.asarray(audio_np, dtype=np.float32).reshape(-1)
                    if audio.size == 0:
                        continue
                    self._segments.append(audio)
                    if not self._stopped:
                        self._enqueue_play(audio)
        finally:
            if self._ensure_worker():
                self._play_q.put(None)
            self._compose_and_save()
            self.finished_signal.emit()

class GeminiTTSThread(QThread):
    """Generate speech using Gemini text-to-speech with streaming."""

    finished_signal = pyqtSignal()

    def __init__(
        self,
        lines: list[str],
        api_key: str,
        voice_name: str = "Leda",
        instruction: str = "",
        model_name: str = "gemini-2.5-flash-preview-tts",
        temperature: float = 1.5,
        parent=None,
    ):
        super().__init__(parent)
        self.lines = lines
        self.api_key = api_key
        self.voice_name = voice_name or "Leda"
        self.instruction = instruction or ""
        self.model_name = model_name or "gemini-2.5-flash-preview-tts"
        self.temperature = float(temperature)
        self._stopped = False

        self._pya = None
        self._stream = None

    def _cleanup(self):
        """Close audio resources safely."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pya:
            try:
                self._pya.terminate()
            except Exception:
                pass
            self._pya = None

    def stop(self):
        self._stopped = True

    def run(self):
        if not self.api_key or not GEMINI_AVAILABLE or pyaudio is None or self._stopped:
            self.finished_signal.emit()
            return

        print(f"[GeminiTTSThread] starting with voice={self.voice_name}")
        try:
            self._pya = pyaudio.PyAudio()
            self._stream = self._pya.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
            )
        except Exception as e:
            print("[GeminiTTSThread] audio init error:", e)
            self._cleanup()
            self.finished_signal.emit()
            return

        prompt = [
            f"Below are a series of instruction:message, please follow them and read them all. User wants you to act with according emotions: {self.instruction}"
        ]
        for line in self.lines:
            if line and line.strip():
                # Add double newline after each sentence to encourage streaming
                prompt.append(
                    f"Read or act out the following text with emotions and acting appropriate to it:\n{line.strip()}\n"
                )
        full_text = "\n\n".join(prompt)


        client = genai.Client(api_key=self.api_key)
        print("[GeminiTTSThread] opened client")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tts_dir = os.path.join(os.path.dirname(__file__), "TTS")
        os.makedirs(tts_dir, exist_ok=True)
        out_path = os.path.join(tts_dir, f"{uuid.uuid4().hex}.wav")

        async def async_task():
            client = genai.Client(api_key=self.api_key)
            collected = bytearray()
            try:
                print("[GeminiTTSThread] requesting audio stream...")
                responses = await client.aio.models.generate_content_stream(
                    model=self.model_name,
                    contents=full_text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        temperature=self.temperature,
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=self.voice_name
                                )
                            )
                        ),
                    ),
                )

                print("[GeminiTTSThread] streaming audio chunks")

                async for chunk in responses:
                    if self._stopped:
                        break

                    # ---- SAFE AUDIO EXTRACTION ----
                    pcm_bytes = None
                    mime_type = None

                    for cand in chunk.candidates:
                        content = getattr(cand, "content", None)
                        if not content or not content.parts:
                            continue

                        for part in content.parts:
                            inline = getattr(part, "inline_data", None)
                            if inline and inline.data:
                                pcm_bytes = inline.data          # raw 16-bit PCM
                                mime_type = inline.mime_type
                                break
                        if pcm_bytes:
                            break     # exit outer ‘for cand’ loop as soon as we found data

                    # If this chunk carried no audio, just ignore it.
                    if not pcm_bytes:
                        continue

                    print(f"[GeminiTTSThread] chunk bytes={len(pcm_bytes)} mime={mime_type}")

                    try:
                        self._stream.write(pcm_bytes)
                        collected.extend(pcm_bytes)
                    except Exception as e:
                        print("[GeminiTTSThread] write error:", e)
                        break
            finally:
                self._cleanup()
                try:
                    wave_file(out_path, collected)
                    print(f"[GeminiTTSThread] saved audio to {out_path}")
                except Exception as e:
                    print("[GeminiTTSThread] save error:", e)
                self.finished_signal.emit()

        loop.run_until_complete(async_task())
        loop.close()
