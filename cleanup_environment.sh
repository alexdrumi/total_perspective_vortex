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


# Function to deactivate and clean up
cleanup() {
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

    # Remove the MLFLOW directory

    if [ -d "$MLFLOW_3_DIR" ]; then
        echo "Removing MLFLOW directory at $MLFLOW_3_DIR..."
        rm -rf "$MLFLOW_3_DIR"
    else
        echo "MLFLOW directory not found at $MLFLOW_2_DIR."
    fi

    if [ -d "$MLFLOW_2_DIR" ]; then
        echo "Removing MLFLOW directory at $MLFLOW_2_DIR..."
        rm -rf "$MLFLOW_2_DIR"
    else
        echo "MLFLOW directory not found at $MLFLOW_2_DIR."
    fi

    if [ -d "$MLFLOW_1_DIR" ]; then
        echo "Removing MLFLOW directory at $MLFLOW_1_DIR..."
        rm -rf "$MLFLOW_1_DIR"
    else
        echo "MLFLOW directory not found at $MLFLOW_1_DIR."
    fi

    if [ -d "$MLRUNS_DIR" ]; then
        echo "Removing MLFLOW directory at $MLRUNS_DIR..."
        rm -rf "$MLRUNS_DIR"
    else
        echo "MLFLOW directory not found at $MLRUNS_DIR."
    fi

    #remover models
    if [ -d "$MODELS_DIR" ]; then
        echo "Removing contents of directory at $MODELS_DIR..."
        rm -f "$MODELS_DIR"/*.joblib
    else
        echo "MLFLOW directory not found at $MLRUNS_DIR."
    fi


    # Remove the PHYSIONET directory
    if [ -d "$PHYSIONET_DIR" ]; then
        echo "Removing PHYSIONET directory at $PHYSIONET_DIR..."
        rm -rf "$PHYSIONET_DIR"
    else
        echo "PHYSIONET directory not found at $PHYSIONET_DIR."
    fi

    echo "Cleanup complete."
}

#run the cleanup function
cleanup
