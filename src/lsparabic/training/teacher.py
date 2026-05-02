from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor


class WhisperTeacher(nn.Module):
    """Frozen Whisper-Large-V3 that produces soft targets and hidden states.

    Used only during training with paired audio/video data.
    All parameters are frozen — gradients never flow through this module.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            model_id = f"openai/whisper-{self.model_size}"
            kwargs = {"cache_dir": str(self.cache_dir)} if self.cache_dir else {}
            self._processor = WhisperProcessor.from_pretrained(model_id, **kwargs)
            self._model = WhisperForConditionalGeneration.from_pretrained(model_id, **kwargs)
            self._model.to(self.device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            logger.info(f"Loaded frozen Whisper teacher: {model_id}")
        except ImportError as e:
            raise ImportError("transformers is required for WhisperTeacher.") from e

    @torch.no_grad()
    def forward(self, audio_features: Tensor) -> dict[str, Tensor]:
        """
        audio_features: (B, 80, T_mel) mel-spectrogram frames
        Returns:
            logits:        (B, seq, vocab_whisper)
            hidden_states: (B, seq, 1280)
        """
        self._load()
        encoder_out = self._model.model.encoder(
            audio_features,
            output_hidden_states=True,
            return_dict=True,
        )
        # Use the last encoder hidden state as teacher representation
        hidden_states = encoder_out.last_hidden_state  # (B, T_enc, 1280)

        # Run decoder one step to obtain logit distribution over vocab
        decoder_input_ids = torch.full(
            (audio_features.size(0), 1),
            self._model.config.decoder_start_token_id,
            device=audio_features.device,
            dtype=torch.long,
        )
        decoder_out = self._model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            return_dict=True,
        )
        logits = self._model.lm_head(decoder_out.last_hidden_state)  # (B, 1, vocab)
        logits = logits.expand(-1, hidden_states.size(1), -1)         # broadcast to (B, T_enc, vocab)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
        }

    def get_audio_features(self, waveform: Tensor, sample_rate: int = 16000) -> Tensor:
        """Convert raw waveform to mel-spectrogram for teacher forward pass."""
        self._load()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        numpy_audio = waveform.cpu().numpy()
        inputs = self._processor(
            numpy_audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        return inputs.input_features.to(self.device)
