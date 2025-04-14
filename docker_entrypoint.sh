#!/bin/bash
set -e

# Define script shortcuts with friendly names
declare -A SCRIPT_OPTIONS
SCRIPT_OPTIONS["cutie_demo"]="demo/sam_cutie_video_segmentation.py"
SCRIPT_OPTIONS["dsn_demo"]="demo/dsn_cutie_video_segmentation.py"
# Add more predefined options here as needed
# SCRIPT_OPTIONS["another_demo"]="path/to/another_script.py"

# Default option if none specified
DEFAULT_OPTION="cutie_demo"
DEFAULT_SCRIPT=${SCRIPT_OPTIONS[$DEFAULT_OPTION]}

# Function to list all available options
list_options() {
    echo "Available script options:"
    for option in "${!SCRIPT_OPTIONS[@]}"; do
        echo "  - $option: ${SCRIPT_OPTIONS[$option]}"
    done
    echo ""
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: docker run ... [-e SCRIPT_OPTION=option_name] [script_args...]"
    echo "   or: docker run ... [-e SCRIPT_TO_RUN=path/to/script.py] [script_args...]"
    echo ""
    list_options
    exit 0
fi

# Determine which script to run
SCRIPT_PATH=""

# First check if a predefined option was selected
if [ ! -z "$SCRIPT_OPTION" ]; then
    if [[ -n "${SCRIPT_OPTIONS[$SCRIPT_OPTION]}" ]]; then
        echo "Using predefined option: $SCRIPT_OPTION"
        SCRIPT_PATH="${SCRIPT_OPTIONS[$SCRIPT_OPTION]}"
    else
        echo "Warning: Unknown option '$SCRIPT_OPTION'"
        list_options
        echo "Defaulting to: ${DEFAULT_SCRIPT}"
        SCRIPT_PATH="${DEFAULT_SCRIPT}"
    fi
# Check if SCRIPT_TO_RUN environment variable is set
elif [ -z "$SCRIPT_TO_RUN" ]; then
    echo "SCRIPT_TO_RUN not specified, defaulting to ${DEFAULT_SCRIPT}"
    SCRIPT_PATH="${DEFAULT_SCRIPT}"
else
    echo "Running specified script: ${SCRIPT_TO_RUN}"
    SCRIPT_PATH="${SCRIPT_TO_RUN}"
fi

# Check if the script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "Error: Script ${SCRIPT_PATH} does not exist!"
    exit 1
fi

# Check if additional args were provided
if [ $# -gt 0 ]; then
    echo "Running: python ${SCRIPT_PATH} $@"
    exec python "${SCRIPT_PATH}" "$@"
else
    echo "Running: python ${SCRIPT_PATH}"
    exec python "${SCRIPT_PATH}"
fi
