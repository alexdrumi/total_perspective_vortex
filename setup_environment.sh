#!/bin/bash

#get location where this script currently is

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Correct DEST_DIR path
DEST_DIR="$SCRIPT_DIR/data"
MODELS_DIR="$SCRIPT_DIR/models"
MLFLOW_DIR="$SCRIPT_DIR/src/mlruns"
VENV_DIR="${SCRIPT_DIR}/venv"
SRC_DIR="${SCRIPT_DIR}/src"

#check if S001 already exists
if [ -d "${DEST_DIR}/S001" ]; then
    echo "Data files already exist in ${DEST_DIR}. Skipping download and organization."
else
    echo "Data files not found in ${DEST_DIR}. Proceeding with downloading the provided data from: https://physionet.org/files/eegmmidb/1.0.0/. File size is 1.9GB, this might take a while."
    # Wait before downloading
    sleep 5

    TEMP_DIR="$SCRIPT_DIR/temp_download"
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"

    wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/

    SOURCE_DIR="$TEMP_DIR/physionet.org/files/eegmmidb/1.0.0"

    if [ -d "$SOURCE_DIR" ]; then
        echo "Organizing downloaded files..."
        mv "$SOURCE_DIR"/* "$DEST_DIR"/

        # Clean up temporary directory
        rm -rf "$TEMP_DIR"

        cd "$DEST_DIR"

        # Remove unwanted directories/files
        rm -rf 044
        rm -f SHA256SUMS.txt
        rm -f wfdbcal

        echo "Data organization complete."
    else
        echo "Error: Expected directory structure not found."
        exit 1
    fi
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

setup_venv
# source "$SCRIPT_DIR/activate_virtualenv.sh"










































#get location where this script currently is

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"



# # Correct DEST_DIR path
# DEST_DIR="$SCRIPT_DIR/data"
# MODELS_DIR="$SCRIPT_DIR/models"
# MLFLOW_DIR="$SCRIPT_DIR/src/mlruns"
# VENV_DIR="${SCRIPT_DIR}/venv"
# SRC_DIR="${SCRIPT_DIR}/src"


# #check if S001 already exists
# if [ -d "${DEST_DIR}/S001" ]; then
#     echo "Data files already exist in ${DEST_DIR}. Skipping download and organization."
# else
#     echo "Data files not found in ${DEST_DIR}. Proceeding with downloading the provided data from:https://physionet.org/content/eegmmidb/1.0.0/${folder}/#files-panel. File size is 1.9GB, this might take a while."
#     # Wait before downloading
#     sleep 5

#     for i in $(seq 1 109); do
#         if [ $i -lt 10 ]; then
#             folder="S00$i"
#         else
#             folder="S0$i"
#         fi
#         wget -r -N -c -np "https://physionet.org/content/eegmmidb/1.0.0/${folder}/#files-panel"
#         sleep 1
#     done

#     SOURCE_DIR="$SCRIPT_DIR/physionet.org/content/eegmmidb/1.0.0/"
#     cp -R ${SOURCE_DIR}/S* ${DEST_DIR}/

#     cd ${DEST_DIR}

#     # Remove unwanted directories/files
#    # Remove unwanted directories/files
#     rm -rf 044
#     rm -f SHA256SUMS.txt
#     rm -f wfdbcal

#     # Correctly remove the physionet.org directory
#     rm -rf "${SCRIPT_DIR}/physionet.org"

#     echo "Data organization complete."

# fi

# # Function to check and install Python
# check_and_install_python() {
#     if ! command -v python3 &> /dev/null; then
#         echo "Python3 is not installed. Installing the latest version of Python..."
        
#         # Install Python based on OS
#         if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#             sudo apt update && sudo apt install -y python3 python3-venv python3-pip
#         elif [[ "$OSTYPE" == "darwin"* ]]; then
#             # macOS
#             if ! command -v brew &> /dev/null; then
#                 echo "Homebrew is not installed. Installing Homebrew..."
#                 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#             fi
#             brew install python
#         else
#             echo "Unsupported OS. Please install Python manually."
#             exit 1
#         fi
#     else
#         echo "Python3 is already installed."
#     fi
# }

# # Virtual environment setup
# setup_venv() {
#     # Install Python if missing
#     check_and_install_python

#     # Create virtual environment
#     if [ ! -d "${VENV_DIR}" ]; then
#         echo "Creating virtual environment in ${VENV_DIR}..."
#         python3 -m venv "${VENV_DIR}"
#     else
#         echo "Virtual environment already exists at ${VENV_DIR}."
#     fi

#     # Activate virtual environment
#     echo "Activating virtual environment..."
#     source "${VENV_DIR}/bin/activate"

#     # Upgrade pip and install dependencies
#     pip install --upgrade pip
#     REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
#     if [ -f "${REQUIREMENTS_FILE}" ]; then
#         echo "Installing required Python packages from ${REQUIREMENTS_FILE}..."
#         pip install --upgrade -r "${REQUIREMENTS_FILE}"
#     else
#         echo "requirements.txt not found in ${SCRIPT_DIR}. Exiting."
#         deactivate
#         exit 1
#     fi
# }

# setup_venv

# chmod +x "$SCRIPT_DIR/2_create_virtualenv.sh"
# chmod +x "$SCRIPT_DIR/3_cleanup.sh"


# source "$SCRIPT_DIR/activate_virtualenv.sh"



