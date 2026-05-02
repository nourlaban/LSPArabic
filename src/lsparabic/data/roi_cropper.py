from __future__ import annotations

import numpy as np
import cv2
from loguru import logger

from lsparabic.data.interfaces import BaseROICropper

# MediaPipe FaceMesh lip landmark indices (outer + inner lips)
MOUTH_LANDMARK_INDICES: list[int] = [
    61, 62, 63, 64, 65, 66, 67, 68,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 95, 96,
    146, 178, 179, 180, 181, 183, 184, 185, 191,
    308, 310, 311, 312, 313, 314, 317, 318, 319, 320, 321, 324, 375,
    402, 403, 404, 405, 407, 408, 409, 415,
]


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
    ) -> None:
        self.output_size = output_size
        self.crop_factor = crop_factor
        self.alpha = temporal_smooth_alpha
        self._static_image_mode = static_image_mode
        self._max_num_faces = max_num_faces
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._face_mesh = None  # lazy init to avoid import at module level

    def _get_face_mesh(self):
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=self._static_image_mode,
                max_num_faces=self._max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
        return self._face_mesh

    def crop(self, frames: list[np.ndarray]) -> np.ndarray:
        """Crop mouth ROI from each frame; returns (T, H, W, 3) uint8."""
        face_mesh = self._get_face_mesh()
        raw_bboxes: list[tuple[int, int, int, int] | None] = []

        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                bbox = self._get_mouth_bbox(results.multi_face_landmarks[0].landmark, h, w)
            else:
                bbox = None
            raw_bboxes.append(bbox)

        smoothed_bboxes = self._stabilize_bbox(raw_bboxes)
        crops: list[np.ndarray] = []

        for frame, bbox in zip(frames, smoothed_bboxes):
            if bbox is None:
                # fallback: centre crop
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                size = min(h, w) // 4
                bbox = (cx - size, cy - size, cx + size, cy + size)
            crops.append(self._crop_and_resize(frame, bbox))

        result = np.stack(crops, axis=0)  # (T, H, W, 3)
        logger.debug(f"Cropped {len(crops)} mouth ROIs at {self.output_size}")
        return result

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
        """Apply EMA smoothing across the sequence to reduce jitter."""
        smoothed: list[tuple[int, int, int, int] | None] = []
        ema: tuple[float, float, float, float] | None = None

        for bbox in bboxes:
            if bbox is None:
                smoothed.append(ema and tuple(int(v) for v in ema))  # type: ignore[arg-type]
                continue
            b = tuple(float(v) for v in bbox)
            if ema is None:
                ema = b  # type: ignore[assignment]
            else:
                ema = tuple(self.alpha * e + (1 - self.alpha) * x for e, x in zip(ema, b))  # type: ignore[assignment]
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

    def __del__(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
