@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
  py -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Optional: build bootstrap data from online successful run:
echo   python prepare_online_demo.py

echo Optional: record your own demonstration:
echo   python record_demo.py

echo Starting AI training...
python start_ai.py
pause
