#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p /home/lab3/min/project/{data,notebooks,src,app,outputs,docs}
mkdir -p /home/lab3/min/output/{models,checkpoints,logs,plots,gradcam,evaluation,predictions,reports}

echo "Setup complete. Activate with: source .venv/bin/activate"
