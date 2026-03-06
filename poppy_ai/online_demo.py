from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from yt_dlp import YoutubeDL


class OnlineDemoBuilder:
    """Downloads a public gameplay video and converts it to local demo frames/actions."""

    def __init__(
        self,
        source_url: str,
        frames_dir: str | Path,
        actions_file: str | Path,
        target_fps: int,
        frame_width: int,
        frame_height: int,
        actions: list[str],
    ) -> None:
        self.source_url = source_url
        self.frames_dir = Path(frames_dir)
        self.actions_file = Path(actions_file)
        self.target_fps = target_fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.actions = actions

    def _download_video(self) -> Path:
        output_dir = self.frames_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "online_demo.%(ext)s")

        with YoutubeDL({"outtmpl": output_template, "format": "mp4/bestvideo+bestaudio/best"}) as ydl:
            result = ydl.extract_info(self.source_url, download=True)
            filepath = ydl.prepare_filename(result)

        video_path = Path(filepath)
        if video_path.suffix.lower() != ".mp4":
            mp4_candidate = video_path.with_suffix(".mp4")
            if mp4_candidate.exists():
                video_path = mp4_candidate
        return video_path

    def _infer_action(self, prev_gray: np.ndarray, gray: np.ndarray) -> str:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mean_x = float(np.mean(flow[..., 0]))
        mean_y = float(np.mean(flow[..., 1]))

        motion_mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        if motion_mag < 0.10:
            return "w" if "w" in self.actions else self.actions[0]

        if abs(mean_x) > abs(mean_y):
            if mean_x > 0:
                return "d" if "d" in self.actions else self.actions[0]
            return "a" if "a" in self.actions else self.actions[0]

        if mean_y > 0:
            return "s" if "s" in self.actions else self.actions[0]
        return "w" if "w" in self.actions else self.actions[0]

    def build(self) -> int:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = self._download_video()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(int(round(source_fps / self.target_fps)), 1) if source_fps > 0 else 1

        idx = 0
        kept = 0
        prev_gray: np.ndarray | None = None
        with self.actions_file.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["frame_file", "action", "timestamp"])
            writer.writeheader()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if idx % step != 0:
                    idx += 1
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

                frame_name = f"frame_{kept:06d}.png"
                cv2.imwrite(str(self.frames_dir / frame_name), gray)

                if prev_gray is None:
                    action = "w" if "w" in self.actions else self.actions[0]
                else:
                    action = self._infer_action(prev_gray, gray)
                writer.writerow({"frame_file": frame_name, "action": action, "timestamp": kept / max(self.target_fps, 1)})

                prev_gray = gray
                kept += 1
                idx += 1

        cap.release()
        return kept
