@echo off
cd /d %~dp0
echo Working directory: %CD%

if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

where ollama >nul 2>&1
if errorlevel 1 (
    echo Ollama is not installed or not in PATH.
    echo Install from https://ollama.com/download and run: ollama pull llama3
    pause
    exit /b 1
)

ollama list >nul 2>&1
if errorlevel 1 (
    echo Ollama appears unavailable. Start Ollama and run: ollama pull llama3
    pause
    exit /b 1
)

echo Ollama detected. Launching Streamlit frontend...
streamlit run frontend/streamlit_app.py
