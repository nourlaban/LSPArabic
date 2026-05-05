from __future__ import annotations

import math

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch import Tensor

from lsparabic.data.interfaces import BaseTokenizer
from lsparabic.models.vsr_model import VSRModel
from lsparabic.training.ctc_loss import CTCLossWrapper
from lsparabic.training.feature_regression_loss import FeatureRegressionLoss
from lsparabic.training.kd_loss import KnowledgeDistillationLoss
from lsparabic.training.teacher import WhisperTeacher
from lsparabic.utils.metrics import MetricsTracker


class VSRLightningModule(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: VSRModel,
        teacher: WhisperTeacher,
        tokenizer: BaseTokenizer,
    ) -> None:
        super().__init__()
        self.save_hyperparameters({"model_config": OmegaConf.to_container(cfg.model, resolve=True)})
        self.cfg = cfg
        self.model = model
        self.teacher = teacher
        self.tokenizer = tokenizer

        weights = cfg.training.loss_weights
        self.ctc_loss = CTCLossWrapper(weight=weights.ctc)
        self.kd_loss = KnowledgeDistillationLoss(
            temperature=cfg.training.kd.temperature,
            weight=weights.kd,
        )
        self.feat_loss = FeatureRegressionLoss(
            student_dim=cfg.training.feature_regression.student_dim,
            teacher_dim=cfg.training.feature_regression.teacher_dim,
            weight=weights.feature_regression,
        )
        self._val_metrics = MetricsTracker(tokenizer)
        self._test_metrics = MetricsTracker(tokenizer)

    def forward(self, batch: dict) -> dict[str, Tensor]:
        return self.model(batch["frames"], batch["frame_lengths"])

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        preds = self.model(batch["frames"], batch["frame_lengths"])
        targets = {
            "labels": batch["labels"],
            "label_lengths": batch["label_lengths"],
        }

        l_ctc = self.ctc_loss(preds, targets)
        total = self.ctc_loss.weight * l_ctc
        self.log("train/ctc_loss", l_ctc, on_step=True, on_epoch=False, prog_bar=False)

        # Teacher-student losses only when audio features are available in batch
        if "audio_feats" in batch and batch["audio_feats"] is not None:
            with torch.no_grad():
                teacher_out = self.teacher(batch["audio_feats"])

            teacher_targets = {
                "teacher_logits": teacher_out["logits"],
                "teacher_hidden": teacher_out["hidden_states"],
            }

            l_kd = self.kd_loss(preds, teacher_targets)
            l_feat = self.feat_loss(preds, teacher_targets)

            total = total + self.kd_loss.weight * l_kd + self.feat_loss.weight * l_feat
            self.log("train/kd_loss", l_kd, on_step=True, on_epoch=False)
            self.log("train/feat_loss", l_feat, on_step=True, on_epoch=False)

        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True)
        return total

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        preds = self.model(batch["frames"], batch["frame_lengths"])
        targets = {
            "labels": batch["labels"],
            "label_lengths": batch["label_lengths"],
        }
        l_ctc = self.ctc_loss(preds, targets)
        self.log("val/ctc_loss", l_ctc, on_epoch=True, prog_bar=False)

        hyps = self._decode_ctc_greedy(preds["logits"])
        refs = self._ids_to_text(batch["labels"], batch["label_lengths"])
        self._val_metrics.update(hyps, refs)

    def on_validation_epoch_end(self) -> None:
        metrics = self._val_metrics.compute()
        self.log("val/wer", metrics["wer"], prog_bar=True)
        self.log("val/cer", metrics["cer"], prog_bar=False)
        self._val_metrics.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        preds = self.model(batch["frames"], batch["frame_lengths"])
        hyps = self._decode_ctc_greedy(preds["logits"])
        refs = self._ids_to_text(batch["labels"], batch["label_lengths"])
        self._test_metrics.update(hyps, refs)

    def on_test_epoch_end(self) -> None:
        metrics = self._test_metrics.compute()
        self.log("test/wer", metrics["wer"])
        self.log("test/cer", metrics["cer"])
        self._test_metrics.reset()

    def configure_optimizers(self):
        opt_cfg = OmegaConf.to_container(self.cfg.optimizer, resolve=True)
        target = opt_cfg.pop("_target_")
        optimizer_cls = _import_class(target)
        optimizer = optimizer_cls(self.model.parameters(), **opt_cfg)

        sch_cfg = OmegaConf.to_container(self.cfg.scheduler, resolve=True)
        warmup_epochs = sch_cfg.pop("warmup_epochs", 0)
        target = sch_cfg.pop("_target_")
        scheduler_cls = _import_class(target)
        scheduler = scheduler_cls(optimizer, **sch_cfg)

        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, scheduler], milestones=[warmup_epochs]
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/wer"},
        }

    def _decode_ctc_greedy(self, logits: Tensor) -> list[str]:
        """Simple greedy CTC decode: argmax → collapse repeats → remove blanks."""
        ids = logits.argmax(dim=-1)  # (B, T)
        results = []
        for seq in ids:
            prev = -1
            tokens = []
            for t in seq.tolist():
                if t != 0 and t != prev:  # 0 = blank
                    tokens.append(t)
                prev = t
            results.append(self.tokenizer.decode(tokens))
        return results

    def _ids_to_text(self, labels: Tensor, lengths: Tensor) -> list[str]:
        results = []
        for i, llen in enumerate(lengths.tolist()):
            ids = labels[i, :llen].tolist()
            results.append(self.tokenizer.decode(ids))
        return results


def _import_class(dotted_path: str):
    parts = dotted_path.rsplit(".", 1)
    module = __import__(parts[0], fromlist=[parts[1]])
    return getattr(module, parts[1])
