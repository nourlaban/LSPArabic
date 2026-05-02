from __future__ import annotations

from pathlib import Path

from lsparabic.data.interfaces import BaseTokenizer


class ArabicSentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: Path) -> None:
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    def id_to_piece(self, idx: int) -> str:
        return self._sp.id_to_piece(idx)

    def piece_to_id(self, piece: str) -> int:
        return self._sp.piece_to_id(piece)


class ArabicTokenizerTrainer:
    def __init__(
        self,
        vocab_size: int = 5000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage

    def train(self, corpus_file: Path, output_prefix: Path) -> None:
        import sentencepiece as spm
        spm.SentencePieceTrainer.train(
            input=str(corpus_file),
            model_prefix=str(output_prefix),
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=self.character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
        )
