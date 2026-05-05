from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from loguru import logger

from lsparabic.data.interfaces import BaseROICropper

# FaceMesh 478-landmark indices for outer + inner lips
MOUTH_LANDMARK_INDICES: list[int] = [
    61, 62, 63, 64, 65, 66, 67, 68,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 95, 96,
    146, 178, 179, 180, 181, 183, 184, 185, 191,
    308, 310, 311, 312, 313, 314, 317, 318, 319, 320, 321, 324, 375,
    402, 403, 404, 405, 407, 408, 409, 415,
]

_DEFAULT_MODEL = str(Path(__file__).parent.parent.parent.parent / "data/models/face_landmarker.task")
_HAAR_CASCADE = str(Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml")


class ROICropError(RuntimeError):
    pass


class MediaPipeMouthCropper(BaseROICropper):
    def __init__(
        self,
        output_size: tuple[int, int] = (96, 96),
        crop_factor: float = 1.5,
        temporal_smooth_alpha: float = 0.7,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        self.output_size = output_size
        self.crop_factor = crop_factor
        self.alpha = temporal_smooth_alpha
        self._static_image_mode = static_image_mode
        self._max_num_faces = max_num_faces
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_path = model_path or _DEFAULT_MODEL
        self._landmarker = None
        self._use_fallback = False  # set True if MediaPipe fails to load

    def _get_landmarker(self):
        if self._use_fallback:
            return None
        if self._landmarker is None:
            try:
                import mediapipe as mp
                running_mode = mp.tasks.vision.RunningMode.IMAGE if self._static_image_mode \
                    else mp.tasks.vision.RunningMode.VIDEO
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
                    running_mode=running_mode,
                    num_faces=self._max_num_faces,
                    min_face_detection_confidence=self._min_detection_confidence,
                    min_tracking_confidence=self._min_tracking_confidence,
                )
                self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            except OSError as e:
                logger.warning(
                    f"MediaPipe Tasks failed to load ({e}). "
                    "Falling back to OpenCV Haar cascade mouth estimation. "
                    "Fix: sudo apt-get install libgles2"
                )
                self._use_fallback = True
        return self._landmarker

    def crop(self, frames: list[np.ndarray]) -> np.ndarray:
        """Crop mouth ROI from each frame; returns (T, H, W, 3) uint8."""
        landmarker = self._get_landmarker()

        if self._use_fallback:
            raw_bboxes = self._detect_bboxes_haar(frames)
        else:
            raw_bboxes = self._detect_bboxes_mediapipe(frames, landmarker)

        smoothed_bboxes = self._stabilize_bbox(raw_bboxes)
        crops: list[np.ndarray] = []

        for frame, bbox in zip(frames, smoothed_bboxes):
            if bbox is None:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                size = min(h, w) // 4
                bbox = (cx - size, cy - size, cx + size, cy + size)
            crops.append(self._crop_and_resize(frame, bbox))

        result_arr = np.stack(crops, axis=0)  # (T, H, W, 3)
        logger.debug(f"Cropped {len(crops)} mouth ROIs at {self.output_size}")
        return result_arr

    def _detect_bboxes_mediapipe(self, frames, landmarker) -> list:
        import mediapipe as mp
        raw_bboxes = []
        for frame_idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if self._static_image_mode:
                result = landmarker.detect(mp_image)
            else:
                result = landmarker.detect_for_video(mp_image, frame_idx * 40)
            if result.face_landmarks:
                bbox = self._get_mouth_bbox(result.face_landmarks[0], h, w)
            else:
                bbox = None
            raw_bboxes.append(bbox)
        return raw_bboxes

    def _detect_bboxes_haar(self, frames: list[np.ndarray]) -> list:
        """Estimate mouth bbox via Haar face detection + geometric approximation."""
        cascade = cv2.CascadeClassifier(_HAAR_CASCADE)
        raw_bboxes = []
        for frame in frames:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                # Mouth is roughly in the bottom third of the face
                cx = fx + fw // 2
                cy = fy + int(fh * 0.75)
                half = int(fw * 0.3 * self.crop_factor)
                bbox = (
                    max(0, cx - half), max(0, cy - half),
                    min(w, cx + half), min(h, cy + half),
                )
            else:
                bbox = None
            raw_bboxes.append(bbox)
        return raw_bboxes

    def _get_mouth_bbox(
        self,
        landmarks,
        frame_h: int,
        frame_w: int,
    ) -> tuple[int, int, int, int]:
        xs = [landmarks[i].x * frame_w for i in MOUTH_LANDMARK_INDICES]
        ys = [landmarks[i].y * frame_h for i in MOUTH_LANDMARK_INDICES]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        half_w = (xmax - xmin) / 2 * self.crop_factor
        half_h = (ymax - ymin) / 2 * self.crop_factor

        half = max(half_w, half_h)
        x0 = max(0, int(cx - half))
        y0 = max(0, int(cy - half))
        x1 = min(frame_w, int(cx + half))
        y1 = min(frame_h, int(cy + half))
        return x0, y0, x1, y1

    def _stabilize_bbox(
        self,
        bboxes: list[tuple[int, int, int, int] | None],
    ) -> list[tuple[int, int, int, int] | None]:
        smoothed: list[tuple[int, int, int, int] | None] = []
        ema: tuple[float, float, float, float] | None = None

        for bbox in bboxes:
            if bbox is None:
                smoothed.append(tuple(int(v) for v in ema) if ema else None)  # type: ignore[arg-type]
                continue
            b = tuple(float(v) for v in bbox)
            ema = b if ema is None else tuple(self.alpha * e + (1 - self.alpha) * x for e, x in zip(ema, b))  # type: ignore[assignment]
            smoothed.append(tuple(int(v) for v in ema))  # type: ignore[arg-type]

        return smoothed

    def _crop_and_resize(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            crop = frame
        return cv2.resize(crop, self.output_size, interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __del__(self) -> None:
        self.close()
