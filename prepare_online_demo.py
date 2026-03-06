from __future__ import annotations

from poppy_ai.config import load_config
from poppy_ai.online_demo import OnlineDemoBuilder


def main() -> None:
    cfg = load_config("config.json")
    builder = OnlineDemoBuilder(
        source_url=cfg.demo_video_url,
        frames_dir=cfg.demo_frames_dir,
        actions_file=cfg.demo_actions_file,
        target_fps=cfg.capture_fps,
        frame_width=cfg.frame_width,
        frame_height=cfg.frame_height,
        actions=cfg.actions,
    )
    total = builder.build()
    print(f"Prepared online demo with {total} frames at {cfg.demo_frames_dir}")


if __name__ == "__main__":
    main()
