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
    echo "Data files not found in ${DEST_DIR}. Proceeding with download and organization."
    # Wait before downloading
    sleep 5

    for i in $(seq 1 60); do
        if [ $i -lt 10 ]; then
            folder="S00$i"
        else
            folder="S0$i"
        fi
        wget -r -N -c -np "https://physionet.org/content/eegmmidb/1.0.0/${folder}/#files-panel"
        sleep 1
    done

    SOURCE_DIR="$SCRIPT_DIR/physionet.org/content/eegmmidb/1.0.0/"
    cp -R ${SOURCE_DIR}/S* ${DEST_DIR}/

    cd ${DEST_DIR}

    # Remove unwanted directories/files
    rm -rf 044
    rm -f SHA256SUMS.txt
    rm -f wfdbcal

    curr_dir=$PWD
    rm -rf "${curr_dir}/physionet.org"

    echo "Data organization complete."

fi

#setup venv
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

chmod +x "$SCRIPT_DIR/2_create_virtualenv.sh"
chmod +x "$SCRIPT_DIR/3_cleanup.sh"
# source "$SCRIPT_DIR/activate_virtualenv.sh"



