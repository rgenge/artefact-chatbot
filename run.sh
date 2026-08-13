#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
MODE="${1:-test}"

cd "${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install Python 3.10 or newer and try again." >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Creating the local virtual environment..."
  python3 -m venv "${VENV_DIR}"
fi

echo "Installing dependencies..."
"${PYTHON}" -m pip install --disable-pip-version-check -r requirements.txt

case "${MODE}" in
  test)
    echo "Running unit and frontend tests..."
    "${PYTHON}" -m unittest discover -s tests -v
    echo "Running the source-backed conversation evaluation..."
    "${PYTHON}" tests_ai/run_evaluation.py
    ;;
  live)
    echo "Running the live Gemini conversation evaluation..."
    "${PYTHON}" tests_ai/run_evaluation.py --live
    ;;
  web)
    exec "${PYTHON}" app.py --web
    ;;
  cli)
    exec "${PYTHON}" app.py
    ;;
  *)
    echo "Usage: bash ./run.sh [test|live|web|cli]" >&2
    exit 2
    ;;
esac
