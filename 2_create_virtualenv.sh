#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Activating virtual environment in $SCRIPT_DIR/venv/bin/activate"
source "$SCRIPT_DIR/venv/bin/activate"

pip install --upgrade pip

REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
if [ -f "${REQUIREMENTS_FILE}" ]; then
    echo "Installing required Python packages from ${REQUIREMENTS_FILE}..."
    pip install --upgrade -r "${REQUIREMENTS_FILE}"
    echo "Required packages installed successfully."
else
    echo "requirements.txt not found in ${SCRIPT_DIR}. Exiting."
    deactivate
    exit 1
fi
