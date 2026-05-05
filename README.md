# LSPArabic — Visual Speech Recognition for Arabic

A production-ready Arabic lip-reading system. Given a **silent video** of someone speaking Arabic, the model predicts the spoken text — and can **clone the speaker's voice** to produce a final video with synthesized Arabic speech that matches how the person sounds.

Training uses your own synchronized audio+video recordings as the data source.

---

## ملاحظات بالعربية

> هذا القسم يشرح المشروع كاملاً باللغة العربية. المحتوى الإنجليزي أدناه يبقى كما هو للرجوع إليه.

### ما هو هذا المشروع؟

**LSPArabic** نظام متكامل لقراءة الشفاه باللغة العربية. يأخذ النظام **فيديو صامت** لشخص يتكلم العربية، ويتنبأ بالنص المنطوق، ثم يستطيع **استنساخ صوت المتكلم** لإنتاج فيديو نهائي يحتوي على كلام عربي مُركَّب بصوت الشخص الأصلي.

---

### المكونات الرئيسية

#### ١. نموذج التعرف البصري على الكلام (VSR Model)

يتكون النموذج من ثلاث طبقات رئيسية:

| المكوّن | الوظيفة |
|---|---|
| **ResNet3D** | يستخرج ميزات بصرية من إطارات الفم `(B, T, 512)` |
| **ConformerEncoder أو MambaSSMEncoder** | يُحلِّل التسلسل الزمني للميزات `(B, T, 256)` |
| **CTCDecoder** | يُحوِّل الميزات إلى رموز نصية عربية |

أثناء **التدريب** فقط، يُستخدم نموذج Whisper-Large-V3 كمعلم مجمَّد لتوجيه النموذج عبر:
- خسارة CTC على تسلسل الرموز المتنبَّأ بها
- تباعد كولباك-لايبلر (KL) بين توزيعات الطالب والمعلم
- خسارة MSE بين الميزات الخفية للطالب والمعلم

أثناء **الاستدلال**، يُستخدم فرع الفيديو فقط — لا يحتاج إلى صوت.

#### ٢. خط أنابيب تركيب الكلام (Synthesis Pipeline)

```
فيديو صامت  ──►  تنبؤ بالنص  ──►  XTTSSynthesizer (XTTS v2)  ──►  فيديو + كلام مُركَّب
                                          ▲
                       استخراج صوت مرجعي (30 ثانية) من فيديو المتكلم
```

| المكوّن | الوظيفة |
|---|---|
| `ReferenceVoiceExtractor` | يستخرج أفضل مقطع صوتي (30 ث) من فيديو المرجع |
| `XTTSSynthesizer` | يُركِّب النص العربي بصوت المتكلم (استنساخ صوتي بدون تدريب) |
| `AudioVideoMuxer` | يدمج الصوت المُركَّب مع الفيديو الصامت عبر ffmpeg |

---

### هيكل الملفات المهمة

```
configs/              ← إعدادات Hydra (النموذج، التدريب، المعالجة المسبقة)
src/lsparabic/
  ├── data/           ← استخراج الإطارات، اقتصاص الفم (MediaPipe)، المُرمِّز
  ├── models/         ← ResNet3D، Conformer، Mamba، CTCDecoder
  ├── training/       ← دورة التدريب (Lightning)، خسارة CTC، تقطير المعرفة
  ├── inference/       ← النافذة المنزلقة، بحث الشعاع، المتنبئ
  ├── synthesis/      ← خط أنابيب تركيب الكلام كاملاً
  └── utils/          ← أدوات النص العربي، تحميل نقاط التفتيش
scripts/              ← preprocess ← train ← evaluate ← infer ← synthesize
```

---

### سير العمل خطوة بخطوة

#### الخطوة ١: المعالجة المسبقة
```bash
python scripts/preprocess.py
```
- يستخرج منطقة الفم (96×96 بكسل) من كل فيديو باستخدام MediaPipe FaceMesh
- يُولِّد النصوص العربية من الصوت عبر Whisper-Large-V3
- يُدرِّب مُرمِّز SentencePiece على النصوص العربية (5000 رمز)
- يُنشئ ملفات التقسيم: `train.csv`, `val.csv`, `test.csv`

#### الخطوة ٢: التدريب
```bash
python scripts/train.py           # Conformer (افتراضي)
python scripts/train.py model=mamba  # Mamba SSM (أسرع)
```
- نقاط التفتيش تُحفظ في `checkpoints/` باسم `epoch=N-val_wer=X.ckpt`
- `last.ckpt` يُكتب بعد كل حقبة
- أنشئ رابطاً رمزياً: `ln -sf "epoch=9-val_wer=0.85.ckpt" checkpoints/best.ckpt`

#### الخطوة ٣: التقييم
```bash
python scripts/evaluate.py inference.checkpoint_path=checkpoints/best.ckpt
```
يُخرج معدل خطأ الكلمات (WER) ومعدل خطأ الأحرف (CER) على مجموعة الاختبار.

#### الخطوة ٤: الاستدلال (نص فقط)
```bash
python scripts/infer.py \
  inference.checkpoint_path=checkpoints/best.ckpt \
  inference.video_path=data/input/my_video.mp4
```

#### الخطوة ٥: التركيب (فيديو + صوت مُستنسَخ)
```bash
pip install "coqui-tts[codec]"   # تثبيت لمرة واحدة (~1.7 جيجابايت)

python scripts/synthesize.py \
  synthesis.checkpoint_path=checkpoints/best.ckpt \
  synthesis.silent_video=data/input/silent_clip.mp4 \
  synthesis.reference_video=data/raw/session_001.mp4 \
  synthesis.output_video=output/result_with_speech.mp4
```

---

### ملاحظات تقنية مهمة

**WSL2 / Linux:**
- MediaPipe يحتاج OpenGL ES: `sudo apt-get install -y libgles2`
- رسالة `DRI3 error` وخطأ `TypeError: 'NoneType' object is not callable` عند الإغلاق — كلاهما غير ضار ويمكن تجاهلهما

**PyTorch 2.9+ و torchcodec:**
- إذا فشل `torchcodec` في التحميل بسبب عدم توافق ABI، يتحوَّل المُركِّب تلقائياً إلى `soundfile` لتحميل الصوت — لا يلزم أي إجراء

**جودة النص المُتنبَّأ:**
- النص المُتنبَّأ به يعكس جودة بيانات التدريب. النموذج المُدرَّب على جلسة واحدة قصيرة سيُنتج نصاً متكرراً ومبتوراً. لتحسين الدقة: أضف المزيد من الفيديوهات وأعد التدريب.

**بنية نقطة التفتيش:**
- تُخزِّن كل نقطة تفتيش إعدادات النموذج داخلها (Conformer أو Mamba). دالة `build_model_from_checkpoint` تقرأ هذه الإعدادات تلقائياً، لذا لا داعي لتحديد البنية يدوياً عند الاستدلال أو التقييم.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [VSR Model](#vsr-model)
  - [Speech Synthesis Pipeline](#speech-synthesis-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow](#workflow)
  - [1. Prepare Data](#1-prepare-data)
  - [2. Preprocess](#2-preprocess)
  - [3. Train](#3-train)
  - [4. Evaluate](#4-evaluate)
  - [5. Infer (text only)](#5-infer-text-only)
  - [6. Synthesize (video with cloned speech)](#6-synthesize-video-with-cloned-speech)
- [Configuration](#configuration)
- [Testing](#testing)
- [Design Principles](#design-principles)
- [Dependencies](#dependencies)

---

## Overview

```
Silent video  ──►  Mouth ROI crop  ──►  ResNet3D + Conformer  ──►  Arabic text
                                                                          │
                                             Reference video (with audio) ┤
                                                                          ▼
                                                           XTTS v2 voice cloning
                                                                          │
                                                                          ▼
                                              Silent video + Synthesized speech  ──►  Output video
```

During **training**, a frozen Whisper-Large-V3 teacher guides the video model via:
- CTC loss on the predicted token sequence
- KL divergence between student and teacher soft distributions
- MSE regression between student and teacher hidden states

During **inference**, only the video branch is used — no audio required.

---

## Architecture

### VSR Model

```
                   ┌─────────────────────────────────────────────────────┐
  Training only    │  WhisperTeacher (frozen Whisper-Large-V3)           │
                   │  audio → { logits (B,T,51865), hidden (B,T,1280) }  │
                   └──────────────┬──────────────────────────────────────┘
                                  │  KD loss + Feature regression loss
                                  ▼
  ┌──────────┐    ┌────────────┐  ┌──────────────────┐  ┌─────────────┐
  │  Frames  │───►│  ResNet3D  │─►│ ConformerEncoder │─►│ CTCDecoder  │──► Arabic text
  │(B,T,3,   │    │(B,T,3,H,W) │  │  (or Mamba SSM)  │  │(B,T,vocab) │
  │  H,W)    │    │→(B,T,512)  │  │  →(B,T,256)      │  └─────────────┘
  └──────────┘    └────────────┘  └──────────────────┘
                        ▲
          MediaPipe FaceMesh mouth ROI crop
          (96×96 px, EMA bbox stabilization)
```

| Component | Implementation | Notes |
|---|---|---|
| Visual encoder | `ResNet3D` — 3D-CNN with 5×7×7 stem | Processes `(B, T, C, H, W)` |
| Temporal model | `ConformerEncoder` — 6-layer Macaron | Primary; swap to `MambaSSMEncoder` for faster inference |
| Decoder | `CTCDecoder` | No alignment labels needed |
| Teacher | Frozen Whisper-Large-V3 | Training only; Arabic dialect-aware |
| Mouth ROI | MediaPipe FaceMesh (468 landmarks) | 96×96 px, crop_factor=1.5, EMA smoothed |
| Tokenization | SentencePiece BPE, vocab=5000 | Handles Arabic morphology |
| Config | Hydra 1.3 + OmegaConf | Modular YAML groups |
| Training loop | PyTorch Lightning | AMP, grad clipping, early stopping |

### Speech Synthesis Pipeline

```
  silent_video.mp4  ──────────────────────────────────────────────────────────────────┐
                                                                                       │
  reference_video.mp4 (same speaker, with audio)                                      │
       │                                                                               │
       ▼  ReferenceVoiceExtractor                                                      │
  reference_voice.wav  (best 30 s segment, 22 050 Hz mono, loudnorm)                  │
       │                                                                               ▼
       │             VSRPredictor                                             AudioVideoMuxer
       │      ──────────────────────                                         (ffmpeg: -c:v copy
       │      silent_video → Arabic text                                      -c:a aac -t <dur>)
       │             │                                                               ▲
       └─────────────┤                                                               │
                     ▼                                                               │
              XTTSSynthesizer  (XTTS v2 zero-shot voice cloning)                    │
              Arabic text + reference_voice → synthesized_speech.wav ───────────────┘
                                                                          │
                                                                          ▼
                                                               output_video.mp4
                                                          (silent video + cloned voice)
```

| Component | Implementation | Notes |
|---|---|---|
| Voice extractor | `ReferenceVoiceExtractor` | ffmpeg-based; picks loudest 30 s segment |
| Voice cloning TTS | `XTTSSynthesizer` (XTTS v2) | Zero-shot; 3–60 s reference, Arabic supported |
| Audio/video mux | `AudioVideoMuxer` | ffmpeg; copies video stream losslessly |
| Subtitle overlay | `AudioVideoMuxer.mux_with_subtitle` | Optional: burns predicted text into frames |

---

## Project Structure

```
LSPArabic/
│
├── configs/                        Hydra config root
│   ├── config.yaml                 Top-level defaults list
│   ├── model/
│   │   ├── conformer.yaml          ResNet3D + Conformer  (default)
│   │   └── mamba.yaml              ResNet3D + Mamba SSM  (optional)
│   ├── data/arabic_vsr.yaml        Dataset paths, batch size, augmentation
│   ├── optimizer/adamw.yaml
│   ├── scheduler/cosine_annealing.yaml
│   ├── training/default.yaml       Loss weights, KD temperature, callbacks
│   ├── preprocessing/default.yaml  MediaPipe + Whisper + tokenizer params
│   ├── inference/default.yaml      Sliding window, beam search
│   └── logger/wandb.yaml
│
├── src/lsparabic/
│   ├── data/
│   │   ├── interfaces.py           ABCs for all data components
│   │   ├── frame_extractor.py      OpenCV frame sampling at target FPS
│   │   ├── roi_cropper.py          MediaPipe mouth ROI + EMA stabilization
│   │   ├── whisper_labeler.py      Whisper ASR → Arabic transcript
│   │   ├── tokenizer.py            SentencePiece BPE tokenizer + trainer
│   │   ├── dataset.py              PyTorch Dataset (loads .npy ROIs)
│   │   ├── datamodule.py           LightningDataModule + collate_fn
│   │   ├── transforms.py           Video augmentation pipeline
│   │   └── preprocessing_pipeline.py  Orchestrates all preprocessing steps
│   │
│   ├── models/
│   │   ├── interfaces.py           BaseVisualEncoder, BaseSequenceModel
│   │   ├── visual_encoder.py       ResNet3D
│   │   ├── conformer.py            ConformerEncoder (Macaron FF→MHSA→Conv→FF)
│   │   ├── mamba_backend.py        MambaSSMEncoder (optional, CUDA only)
│   │   ├── decoder.py              CTCDecoder + AttentionDecoder
│   │   └── vsr_model.py            Assembled model; exposes get_intermediate_features()
│   │
│   ├── training/
│   │   ├── interfaces.py           BaseLossComponent
│   │   ├── ctc_loss.py             CTC loss wrapper
│   │   ├── kd_loss.py              KL divergence knowledge distillation
│   │   ├── feature_regression_loss.py  MSE between student/teacher hidden states
│   │   ├── teacher.py              Frozen WhisperTeacher
│   │   └── lightning_module.py     Full training/val/test loop
│   │
│   ├── inference/
│   │   ├── interfaces.py           BaseDecoder
│   │   ├── sliding_window.py       Overlapping window chunking + merge
│   │   ├── beam_search.py          CTC beam search (optional KenLM)
│   │   └── predictor.py            End-to-end MP4 → Arabic string
│   │
│   ├── synthesis/
│   │   ├── interfaces.py           BaseVoiceSynthesizer ABC
│   │   ├── voice_extractor.py      ReferenceVoiceExtractor (best segment, loudnorm)
│   │   ├── tts_synthesizer.py      XTTSSynthesizer — XTTS v2 zero-shot voice cloning
│   │   ├── audio_video_muxer.py    AudioVideoMuxer — ffmpeg audio+video mux
│   │   └── synthesis_pipeline.py  SynthesisPipeline — end-to-end orchestrator
│   │
│   └── utils/
│       ├── arabic_text.py          Normalize, strip diacritics, WER/CER
│       ├── checkpoint_utils.py     build_model_from_checkpoint (reads arch from ckpt)
│       ├── metrics.py              MetricsTracker (accumulates WER/CER)
│       ├── video_io.py             VideoReader / VideoWriter helpers
│       └── logging_utils.py        Loguru setup
│
├── scripts/
│   ├── preprocess.py               Step 1: extract ROIs + transcribe audio
│   ├── train.py                    Step 2: fit the model
│   ├── evaluate.py                 Step 3: WER/CER on test split
│   ├── infer.py                    Step 4: predict text from a silent video
│   └── synthesize.py               Step 5: lip-read + clone voice → output video with speech
│
├── tests/
│   ├── conftest.py                 Shared fixtures (dummy video, tokenizer, mocks)
│   ├── unit/                       14 unit tests (CPU-only)
│   │   └── synthesis/              3 synthesis unit tests
│   └── integration/                5 integration tests
│       └── synthesis/              1 synthesis integration test
│
└── data/                           (gitignored — populate locally)
    ├── raw/                        Your original .mp4 recordings
    ├── processed/
    │   ├── mouth_rois/             .npy files: (T, 96, 96, 3)
    │   └── labels/                 .json Whisper transcripts
    ├── tokenizer/                  Trained SentencePiece .model + .vocab
    └── splits/                     train.csv, val.csv, test.csv
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (recommended; CPU works for inference only)

### Setup

```bash
# Clone and enter the project
git clone <your-repo-url>
cd LSPArabic

# Install with dev extras
pip install -e ".[dev]"

# Install PyTorch with your CUDA version (example: CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Optional: Mamba SSM backend (CUDA only)
pip install "lsparabic[mamba]"

# Optional: LM-guided beam search
pip install "lsparabic[lm]"

# Optional: voice synthesis (XTTS v2, ~1.7 GB model auto-downloaded on first run)
pip install "coqui-tts[codec]"

# Configure secrets
cp .env.example .env
# Edit .env: set WANDB_API_KEY, CUDA_VISIBLE_DEVICES, etc.
```

> **PyTorch 2.9 + torchcodec note** — If `torchcodec` fails to load due to an ABI mismatch (symbol `torch_dtype_float4_e2m1fn_x2` missing), the synthesizer automatically falls back to `soundfile` for audio I/O. No action needed.

> **WSL2 note** — MediaPipe Tasks API requires OpenGL ES. Install Mesa: `sudo apt-get install -y libgles2`. The `DRI3 error` warnings and the `TypeError: 'NoneType' object is not callable` on shutdown are harmless.

---

## Workflow

### 1. Prepare Data

Place your Arabic video recordings (`.mp4`, `.avi`, `.mov`) in `data/raw/`. Each video should have clear, synchronized audio and frontal face visibility.

```
data/raw/
  ├── session_001.mp4
  ├── session_002.mp4
  └── ...
```

### 2. Preprocess

Extracts mouth ROIs, generates Arabic transcripts via Whisper, trains a SentencePiece tokenizer, and creates train/val/test split CSVs.

```bash
python scripts/preprocess.py
```

Override any parameter from the command line:

```bash
# Use a smaller Whisper model for faster preprocessing
python scripts/preprocess.py preprocessing.whisper.model_size=small

# Adjust ROI crop factor
python scripts/preprocess.py preprocessing.mouth_roi.crop_factor=2.0
```

Output after preprocessing:

```
data/processed/mouth_rois/   ← .npy files (T, 96, 96, 3)
data/processed/labels/       ← .json Arabic transcripts
data/tokenizer/arabic_spm.model
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

### 3. Train

```bash
python scripts/train.py
```

Common overrides:

```bash
# Use Mamba SSM backend instead of Conformer
python scripts/train.py model=mamba

# Use CSV logger instead of W&B
python scripts/train.py logger=csv

# Fast sanity check (1 batch per epoch)
python scripts/train.py trainer.fast_dev_run=true

# Tune loss weights
python scripts/train.py training.loss_weights.kd=1.0 training.loss_weights.feature_regression=0.2

# Increase batch size with gradient accumulation
python scripts/train.py data.batch_size=8 training.accumulate_grad_batches=8
```

Checkpoints are saved to `checkpoints/` and monitored by `val/wer`. Files are named `epoch=N-val_wer=X.ckpt`; a `last.ckpt` symlink is also written after every epoch. Create a `best.ckpt` symlink to the checkpoint you want to use for evaluation and inference:

```bash
ln -sf "epoch=9-val_wer=0.85.ckpt" checkpoints/best.ckpt
```

### 4. Evaluate

```bash
python scripts/evaluate.py inference.checkpoint_path=checkpoints/best.ckpt
```

Outputs `test/wer` and `test/cer` to the console.

### 5. Infer (text only)

Run lip reading on a silent video and print the predicted Arabic text:

```bash
python scripts/infer.py \
  inference.checkpoint_path=checkpoints/best.ckpt \
  inference.video_path=my_video.mp4
```

Expected output:

```
============================================================
Predicted Arabic text:
مرحبا بالعالم
============================================================
```

Tunable inference parameters:

```bash
# Wider beam search
python scripts/infer.py ... inference.beam_search.beam_width=20

# Use a different window size (frames)
python scripts/infer.py ... inference.sliding_window.window_size=100

# Use CPU
python scripts/infer.py ... inference.device=cpu
```

### 6. Synthesize (video with cloned speech)

Perform lip reading **and** clone the speaker's voice to produce a final video with Arabic speech:

```bash
pip install "coqui-tts[codec]"    # one-time; ~1.7 GB model auto-downloaded on first run

python scripts/synthesize.py \
  synthesis.checkpoint_path=checkpoints/best.ckpt \
  synthesis.silent_video=data/input/silent_clip.mp4 \
  synthesis.reference_video=data/raw/session_001.mp4 \
  synthesis.output_video=output/result_with_speech.mp4
```

What each argument means:

| Argument | Description |
|---|---|
| `synthesis.silent_video` | The video you want to lip-read (no audio required) |
| `synthesis.reference_video` | **Any video of the same speaker with clear audio** — used to clone the voice |
| `synthesis.output_video` | Where to write the result |

**Supply a reference WAV directly** (skip extraction):

```bash
python scripts/synthesize.py \
  synthesis.checkpoint_path=checkpoints/best.ckpt \
  synthesis.silent_video=data/input/clip.mp4 \
  synthesis.reference_audio=my_voice_sample.wav \
  synthesis.output_video=output/result.mp4
```

**Burn predicted Arabic subtitles into the video**:

```bash
python scripts/synthesize.py ... synthesis.burn_subtitles=true
```

**Batch process multiple silent videos** (same speaker):

```python
from lsparabic.synthesis import SynthesisPipeline

results = pipeline.run_batch(
    silent_videos=list(Path("input/").glob("*.mp4")),
    reference_video=Path("reference/speaker.mp4"),
    output_dir=Path("output/"),
)
```

Console output:

```
[1/4] Predicting text from: silent_clip.mp4
      Predicted: 'مرحبا بكم في هذا البرنامج'
[2/4] Extracting reference voice from: speaker_audio.mp4
[3/4] Synthesizing Arabic speech with voice cloning …
[4/4] Muxing audio into video → result_with_speech.mp4

=================================================================
  Predicted text : مرحبا بكم في هذا البرنامج
  Reference audio: /tmp/reference_voice.wav
  Output video   : output/result_with_speech.mp4
=================================================================
```

---

## Configuration

All configuration is managed by [Hydra](https://hydra.cc). The entry point is [configs/config.yaml](configs/config.yaml).

### Key parameters

| Config group | Key | Default | Description |
|---|---|---|---|
| `model` | `conformer` / `mamba` | `conformer` | Temporal backend |
| `training` | `loss_weights.ctc` | `1.0` | CTC loss weight |
| `training` | `loss_weights.kd` | `0.5` | Knowledge distillation weight |
| `training` | `loss_weights.feature_regression` | `0.1` | Feature MSE weight |
| `training` | `kd.temperature` | `4.0` | KD softmax temperature |
| `training` | `max_epochs` | `100` | Training epochs |
| `preprocessing` | `whisper.model_size` | `large-v3` | Whisper model for labeling |
| `preprocessing` | `mouth_roi.crop_factor` | `1.5` | ROI bounding box expansion |
| `inference` | `sliding_window.window_size` | `75` | Frames per window |
| `inference` | `beam_search.beam_width` | `10` | Beam search width |
| `synthesis` | `tts.temperature` | `0.65` | XTTS creativity vs stability |
| `synthesis` | `tts.speed` | `1.0` | Speech playback speed |
| `synthesis` | `voice_extraction.duration` | `30.0` | Reference clip length (seconds) |
| `synthesis` | `muxer.audio_bitrate` | `192k` | Output audio quality |
| `synthesis` | `burn_subtitles` | `false` | Burn predicted text into video frames |

### Switching to Mamba

```bash
python scripts/train.py model=mamba
```

Requires `pip install "lsparabic[mamba]"` on a CUDA machine.

---

## Testing

```bash
# Unit tests only (no GPU, no internet required)
make test-unit

# Integration tests
make test-integration

# Full suite with coverage
make test

# Type checking
make typecheck

# Linting
make lint
```

### Test coverage targets

| Test | What it verifies |
|---|---|
| `test_frame_extractor` | Correct FPS sampling, corrupt file error |
| `test_roi_cropper` | 96×96 output shape, EMA stabilization, fallback crop |
| `test_whisper_labeler` | ffmpeg command, transcription return type |
| `test_tokenizer` | Encode/decode roundtrip, vocab_size |
| `test_visual_encoder` | Output shape `(B, T, 512)` |
| `test_conformer` | Output shape, masked attention, no NaN |
| `test_mamba_backend` | Skipped if `mamba_ssm` not installed |
| `test_kd_loss` | Non-negative, ≈0 when student=teacher |
| `test_feature_regression_loss` | Non-negative MSE |
| `test_beam_search` | Valid Arabic string, batch results |
| `test_arabic_text` | Diacritic stripping, alef normalization, WER |
| `test_preprocessing_pipeline` | ROI + label files created on disk |
| `test_datamodule` | DataLoader yields correct tensor shapes |
| `test_lightning_module` | training_step / validation_step no crash |
| `test_predictor` | End-to-end MP4 → string |

---

## Design Principles

This project follows **SOLID** principles throughout:

- **Single Responsibility** — each class has one job; `PreprocessingPipeline` orchestrates but does not implement extraction or cropping
- **Open/Closed** — swap `ConformerEncoder` for `MambaSSMEncoder` without changing `VSRModel`
- **Liskov Substitution** — all concrete classes honour their ABC contracts
- **Interface Segregation** — `BaseFrameExtractor`, `BaseROICropper`, `BaseASRLabeler`, `BaseTokenizer`, `BaseVisualEncoder`, `BaseSequenceModel`, `BaseLossComponent`, `BaseDecoder` each cover exactly one concern
- **Dependency Inversion** — `VSRModel` depends on `BaseVisualEncoder` and `BaseSequenceModel`; concrete classes are injected via constructors, never imported directly across modules

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` + `pytorch-lightning` | Model training framework |
| `hydra-core` + `omegaconf` | Configuration management |
| `mediapipe` | Face landmark detection for mouth ROI |
| `opencv-python` | Video I/O and frame processing |
| `faster-whisper` | Arabic ASR teacher model |
| `sentencepiece` | Sub-word tokenization for Arabic |
| `einops` | Tensor shape manipulation |
| `jiwer` | WER / CER metrics |
| `pyarabic` | Arabic text normalization utilities |
| `loguru` | Structured logging |
| `wandb` | Experiment tracking |
| `mamba-ssm` *(optional)* | Mamba SSM temporal backend |
| `pyctcdecode` *(optional)* | KenLM-guided beam search |
| `coqui-tts[codec]` *(optional)* | Coqui XTTS v2 zero-shot voice cloning (Python 3.12 compatible fork) |
| `soundfile` *(optional)* | WAV I/O for long-text synthesis chunking; also used as torchaudio fallback |
| `librosa` *(optional)* | Audio analysis utilities |
