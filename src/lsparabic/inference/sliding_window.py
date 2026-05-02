from __future__ import annotations

import numpy as np


class SlidingWindowProcessor:
    """Split a long ROI sequence into overlapping windows and merge predictions."""

    def __init__(
        self,
        window_size: int = 75,
        stride: int = 25,
        overlap_strategy: str = "max_pool",
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.overlap_strategy = overlap_strategy

    def process(self, roi_sequence: np.ndarray) -> list[np.ndarray]:
        """roi_sequence: (T, H, W, C) -> list of (window_size, H, W, C) chunks."""
        t = roi_sequence.shape[0]
        windows: list[np.ndarray] = []

        if t <= self.window_size:
            # Pad single window to window_size
            pad = self.window_size - t
            padded = np.concatenate(
                [roi_sequence, np.zeros((pad, *roi_sequence.shape[1:]), dtype=roi_sequence.dtype)],
                axis=0,
            )
            return [padded]

        start = 0
        while start < t:
            end = start + self.window_size
            if end > t:
                # Last window: take the final window_size frames
                chunk = roi_sequence[t - self.window_size: t]
            else:
                chunk = roi_sequence[start:end]
            windows.append(chunk)
            if end >= t:
                break
            start += self.stride

        return windows

    def merge_predictions(self, window_preds: list[str]) -> str:
        """Merge per-window text predictions into a single output string.

        Uses simple deduplication: remove tokens at window boundaries that
        appear to be cut-off repetitions from the previous window.
        """
        if not window_preds:
            return ""
        if len(window_preds) == 1:
            return window_preds[0].strip()

        merged_words: list[str] = []
        for pred in window_preds:
            words = pred.strip().split()
            if not words:
                continue
            if not merged_words:
                merged_words.extend(words)
                continue
            # Deduplicate: if the first word of this window == last word of merged, skip it
            start_idx = 1 if (words[0] == merged_words[-1]) else 0
            merged_words.extend(words[start_idx:])

        return " ".join(merged_words)
