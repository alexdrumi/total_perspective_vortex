#!/bin/bash

# Get the location where this script currently is
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set directory paths
DEST_DIR="$SCRIPT_DIR/data"
MODELS_DIR="$SCRIPT_DIR/models"
MLFLOW_DIR="$SCRIPT_DIR/src/mlruns"
VENV_DIR="${SCRIPT_DIR}/venv"
SRC_DIR="${SCRIPT_DIR}/src"

# Check if data directory exists, if not, create it.
if [ ! -d "${DEST_DIR}" ]; then
    echo "Data directory not found. Creating data directory at ${DEST_DIR}..."
    mkdir -p "${DEST_DIR}"
    echo "Please manually download and place the data into the '${DEST_DIR}' folder."
else
    echo "Data directory exists at ${DEST_DIR}."
fi

# Function to check and install Python
check_and_install_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python3 is not installed. Installing the latest version of Python..."
        # Install Python based on OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt update && sudo apt install -y python3 python3-venv python3-pip
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if ! command -v brew &> /dev/null; then
                echo "Homebrew is not installed. Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python
        else
            echo "Unsupported OS. Please install Python manually."
            exit 1
        fi
    else
        echo "Python3 is already installed."
    fi
}

# Virtual environment setup
setup_venv() {
    # Install Python if missing
    check_and_install_python

    # Create virtual environment
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment in ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    else
        echo "Virtual environment already exists at ${VENV_DIR}."
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source "${VENV_DIR}/bin/activate"

    # Upgrade pip and install dependencies
    pip install --upgrade pip
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        echo "Installing required Python packages from ${REQUIREMENTS_FILE}..."
        pip install --upgrade -r "${REQUIREMENTS_FILE}"
    else
        echo "requirements.txt not found in ${SCRIPT_DIR}. Exiting."
        deactivate
        exit 1
    fi
}

# Set up the virtual environment
setup_venv
