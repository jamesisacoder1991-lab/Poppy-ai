# Poppy Playtime Evolution AI (Starter Kit)

This project gives you a **one-click launcher** to run an AI trainer for Poppy Playtime Chapter 1 on Windows.

## What it does
- Captures live game frames.
- Starts from a demonstration dataset as generation 1.
- Can auto-download a **public successful run** and convert it into bootstrap data.
- Uses deep learning (behavior cloning) to initialize a strong base brain.
- Evolves a population by keeping the best brain and mutating clones.
- Uses a reward function (movement + exploration + survival).
- Saves checkpoints automatically so progress is never lost.

## One-click setup (Windows)
1. Download this repo as a ZIP.
2. Unzip it.
3. Double-click `launch_ai.bat`.
4. Keep Poppy Playtime running in windowed mode.

The launcher creates `.venv`, installs dependencies, and starts training.

## Use online successful runs first (no self-recording required)
Default config enables online bootstrap:
- `auto_download_demo: true`
- `demo_video_url: <public gameplay URL>`

At training start, if no local demo exists, trainer will:
1. Download the video.
2. Extract resized grayscale frames into `demos/frames`.
3. Infer approximate actions with optical-flow motion cues.
4. Save `demos/actions.csv`.
5. Pretrain generation-1 brain from this dataset.

You can also run it directly:
```bash
python prepare_online_demo.py
```

## Optional: record your own demo
If you want better labels than inferred actions:
```bash
python record_demo.py
```

## Notes
- Tune settings in `config.json`.
- Saves are written to `saves/` by default.
- Online bootstrap requires internet access and a valid public gameplay URL.

## Files
- `launch_ai.bat` - one-click Windows launcher.
- `prepare_online_demo.py` - downloads and converts online runs.
- `record_demo.py` - records a local human demonstration dataset.
- `start_ai.py` - Python entrypoint.
- `poppy_ai/` - training code.
- `config.json` - editable settings.
