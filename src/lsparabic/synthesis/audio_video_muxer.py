from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


class AudioVideoMuxer:
    """Combine a silent video with a synthesized audio track using ffmpeg.

    The video stream is copied without re-encoding. The audio is encoded to
    AAC for broad compatibility. When audio is shorter than the video, silence
    pads the remainder; when longer, the output is trimmed to the video length.
    """

    def __init__(
        self,
        audio_bitrate: str = "192k",
        audio_codec: str = "aac",
        preserve_metadata: bool = True,
        pad_audio: bool = True,
    ) -> None:
        self.audio_bitrate = audio_bitrate
        self.audio_codec = audio_codec
        self.preserve_metadata = preserve_metadata
        self.pad_audio = pad_audio

    def mux(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        audio_offset: float = 0.0,
    ) -> Path:
        """Combine `video_path` (silent) with `audio_path` (synthesized speech).

        Args:
            video_path:   Silent input video.
            audio_path:   Synthesized WAV or any audio file.
            output_path:  Destination video path (.mp4).
            audio_offset: Seconds to delay the audio track (positive = later).

        Returns the path to the output video.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)

        cmd = ["ffmpeg", "-y"]

        if audio_offset > 0:
            cmd += ["-itsoffset", str(audio_offset)]

        cmd += ["-i", str(video_path), "-i", str(audio_path)]

        # Stream mappings: video from file 0, audio from file 1
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

        # Copy video, encode audio
        cmd += ["-c:v", "copy", "-c:a", self.audio_codec, "-b:a", self.audio_bitrate]

        # Pad audio with silence if shorter than video
        if self.pad_audio:
            cmd += ["-af", f"apad=whole_dur={video_duration}"]

        # Trim to video length so no silent tail when audio is longer
        cmd += ["-t", str(video_duration)]

        if self.preserve_metadata:
            cmd += ["-map_metadata", "0", "-metadata:s:a:0", "language=ara"]

        cmd.append(str(output_path))

        logger.info(f"Muxing: {video_path.name} + {audio_path.name} → {output_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux failed:\n{result.stderr[-2000:]}"
            )

        logger.info(f"Output video written: {output_path}")
        return output_path

    def mux_with_subtitle(
        self,
        video_path: Path,
        audio_path: Path,
        subtitle_text: str,
        output_path: Path,
    ) -> Path:
        """Mux audio and burn the predicted Arabic text as a subtitle overlay."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False,
                                        encoding="utf-8") as srt_file:
            duration = self._get_duration(video_path)
            # Single subtitle spanning the whole video
            h, rem = divmod(int(duration), 3600)
            m, s = divmod(rem, 60)
            srt_file.write(
                f"1\n"
                f"00:00:00,000 --> {h:02d}:{m:02d}:{s:02d},000\n"
                f"{subtitle_text}\n\n"
            )
            srt_path = Path(srt_file.name)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-vf", f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=24,"
                   f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,"
                   f"Alignment=2'",
            "-c:a", self.audio_codec, "-b:a", self.audio_bitrate,
            "-t", str(self._get_duration(video_path)),
            "-map_metadata", "0",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        srt_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg subtitle mux failed:\n{result.stderr[-2000:]}")
        logger.info(f"Output video with subtitles: {output_path}")
        return output_path

    @staticmethod
    def _get_duration(path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0
