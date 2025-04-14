#!/bin/bash

# Display usage information
function show_help {
    echo "Usage: $0 [options] [-- script_args]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -s, --script NAME         Specify which script to run (e.g., cutie_demo, dsn_demo)"
    echo "  -v, --video PATH          Specify video file path (will be mounted inside container)"
    echo "  -o, --output-dir DIR      Specify output directory"
    echo ""
    echo "Additional script arguments can be passed after --"
    echo "Example: $0 -s cutie_demo -- --save-frames --save-yolo"
    echo ""
}

# Default values
SCRIPT_OPTION="cutie_demo"
VIDEO_PATH=""
OUTPUT_DIR="output"
ADDITIONAL_ARGS=""

# Parse command line arguments before --
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--script)
            SCRIPT_OPTION="$2"
            shift 2
            ;;
        -v|--video)
            # Get absolute path for proper mounting
            VIDEO_PATH=$(realpath "$2")
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --)
            shift
            # Collect all remaining arguments as script arguments
            ADDITIONAL_ARGS="$@"
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH=$(realpath "$OUTPUT_DIR")

# Prepare Docker volume mounts
VOLUMES="-v $OUTPUT_PATH:/MASK_ML/output"

# If video path was provided, add it as a volume and prepare the script argument
if [ -n "$VIDEO_PATH" ]; then
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    VOLUMES="$VOLUMES -v $VIDEO_DIR:/input"
    # Add video flag to additional args if not already there
    if [[ ! "$ADDITIONAL_ARGS" =~ "--video" ]]; then
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS --video /input/$VIDEO_FILENAME"
    fi
fi

# Allow local X server connections for GUI
xhost +local:docker

# Run the Docker container with all specified options
echo "Running with script option: $SCRIPT_OPTION"
echo "Additional arguments: $ADDITIONAL_ARGS"

# Use docker compose run with --no-deps to isolate this service run and explicitly override the command
docker compose run --rm --no-deps $VOLUMES \
    -e SCRIPT_OPTION="$SCRIPT_OPTION" \
    mask_ml_app $ADDITIONAL_ARGS



