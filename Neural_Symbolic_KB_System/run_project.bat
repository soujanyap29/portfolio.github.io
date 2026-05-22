@echo off
cd /d %~dp0

if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Ensure Ollama is running and llama3 model is pulled before starting Streamlit.
streamlit run frontend/streamlit_app.py
