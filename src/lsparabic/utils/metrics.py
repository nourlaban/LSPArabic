from __future__ import annotations

from lsparabic.data.interfaces import BaseTokenizer
from lsparabic.utils.arabic_text import compute_cer, compute_wer, normalize_arabic


class MetricsTracker:
    """Accumulates WER and CER across validation/test batches."""

    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self.tokenizer = tokenizer
        self._hyps: list[str] = []
        self._refs: list[str] = []

    def update(self, predictions: list[str], references: list[str]) -> None:
        assert len(predictions) == len(references)
        self._hyps.extend(normalize_arabic(h) for h in predictions)
        self._refs.extend(normalize_arabic(r) for r in references)

    def compute(self) -> dict[str, float]:
        if not self._refs:
            return {"wer": 0.0, "cer": 0.0}
        all_hyp = " ".join(self._hyps)
        all_ref = " ".join(self._refs)
        return {
            "wer": compute_wer(all_hyp, all_ref),
            "cer": compute_cer(all_hyp, all_ref),
        }

    def reset(self) -> None:
        self._hyps.clear()
        self._refs.clear()
