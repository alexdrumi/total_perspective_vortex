#!/bin/bash

# Get the directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths to clean
VENV_DIR="${SCRIPT_DIR}/venv"
MLFLOW_1_DIR="${SCRIPT_DIR}/src/main_app/mlartifacts"
MLFLOW_2_DIR="${SCRIPT_DIR}/mlartifacts"
MLFLOW_3_DIR="${SCRIPT_DIR}/src/main_app/mlruns"
MLRUNS_DIR="${SCRIPT_DIR}/mlruns"
PHYSIONET_DIR="${SCRIPT_DIR}/physionet.org"
MODELS_DIR="${SCRIPT_DIR}/models"

# Function to remove all __pycache__ directories recursively
remove_pycache() {
    echo "Searching for and removing all __pycache__ directories..."
    find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} +
    echo "All __pycache__ directories removed."
}

# Function to kill MLflow server running on port 5000
kill_mlflow_server() {
    echo "Attempting to kill process running on port 5000..."
    if command -v lsof >/dev/null 2>&1; then
        PID=$(lsof -ti:5000)
        if [ -n "$PID" ]; then
            kill -9 $PID
            echo "Process on port 5000 killed."
        else
            echo "No process running on port 5000 found."
        fi
    else
        echo "lsof command not found, unable to kill process on port 5000."
    fi
}

# Function to deactivate and clean up
cleanup() {
    # Kill MLflow server on port 5000
    kill_mlflow_server

    # Deactivate virtual environment if active
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Deactivating virtual environment..."
        deactivate
    else
        echo "No active virtual environment to deactivate."
    fi

    # Remove the virtual environment directory
    if [ -d "$VENV_DIR" ]; then
        echo "Removing virtual environment directory at $VENV_DIR..."
        rm -rf "$VENV_DIR"
    else
        echo "Virtual environment directory not found at $VENV_DIR."
    fi

    # Remove the MLFLOW directories
    for dir in "$MLFLOW_1_DIR" "$MLFLOW_2_DIR" "$MLFLOW_3_DIR" "$MLRUNS_DIR"; do
        if [ -d "$dir" ]; then
            echo "Removing MLFLOW directory at $dir..."
            rm -rf "$dir"
        else
            echo "MLFLOW directory not found at $dir."
        fi
    done

    # Remove model files
    if [ -d "$MODELS_DIR" ]; then
        echo "Removing contents of directory at $MODELS_DIR..."
        rm -f "$MODELS_DIR"/*.joblib
    else
        echo "Models directory not found at $MODELS_DIR."
    fi

    # Remove the PHYSIONET directory
    if [ -d "$PHYSIONET_DIR" ]; then
        echo "Removing PHYSIONET directory at $PHYSIONET_DIR..."
        rm -rf "$PHYSIONET_DIR"
    else
        echo "PHYSIONET directory not found at $PHYSIONET_DIR."
    fi

    # Remove all __pycache__ directories
    remove_pycache

    echo "Cleanup complete."
}

# Run the cleanup function
cleanup
