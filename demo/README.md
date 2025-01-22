
# SAM Cutie Video Segmentation

**SAM Cutie Video Segmentation** is a Python-based application that leverages the SAM (Segment Anything Model) pipeline to perform real-time video segmentation. Whether you're processing pre-recorded videos or utilizing your webcam for live segmentation, this tool provides a flexible and interactive interface to annotate and analyze video frames with ease.


## Installation

1. **Clone the Repository with Cutie**

   ```bash
   git clone --recursive https://github.com/Cruiz102/MASK_ML.git
   cd MASK_ML
   ```

2. **Set Up a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -e . 
   pip install ultralytics
   ```

   *Ensure that you install the correct version of PyTorch compatible with your CUDA version if you intend to use GPU acceleration.*

   ```bash
   #Default is using 
   pip install torch
   ```


## Usage

The primary script for video segmentation is located at `demo/sam_cutie_video_segmentation.py`. This script can process either a video file or live webcam feed based on the provided command-line arguments.

### Running with a Video File

To process a pre-recorded video, use the `--video` argument followed by the path to your video file.

```bash
python demo/sam_cutie_video_segmentation.py --video path/to/your/video.mp4
```

### Running with the Webcam

If you prefer to use your webcam for live segmentation, simply omit the `--video` argument. The script will automatically default to the webcam input.

```bash
python demo/sam_cutie_video_segmentation.py
```

### Additional Options

You can combine additional flags to enhance functionality:

- `--save-yolo`: Save bounding boxes in YOLO format for each frame.
- `--save-frames`: Store extracted frames in the output directory.

**Example:**

```bash
python demo/sam_cutie_video_segmentation.py --video path/to/video.mp4 --save-yolo --save-frames
```

Or, using the webcam with additional options:

```bash
python demo/sam_cutie_video_segmentation.py --save-yolo --save-frames
```

## Command-Line Arguments

The script accepts the following command-line arguments:

| Argument        | Type  | Required | Description                                                                                   |
| --------------- | ----- | -------- | --------------------------------------------------------------------------------------------- |
| `--video`       | `str` | No       | Path to the video file. If not provided, the webcam will be used.                             |
| `--output-dir`  | `str` | No       | Directory to store optional outputs (default: `output`).                                     |
| `--save-yolo`   | Flag  | No       | If set, save bounding boxes in YOLO format for each frame.                                   |
| `--save-frames` | Flag  | No       | If set, store extracted frames in the output folder.                                         |

**Detailed Descriptions:**

- **`--video`**: Specify the path to a video file you wish to process. If this argument is omitted, the script will default to using the system's primary webcam for live video capture.
  
- **`--output-dir`**: Define the directory where output files (like YOLO bounding boxes and frames) will be saved. The default directory is named `output`.

- **`--save-yolo`**: Enable this flag to save detected bounding boxes in YOLO format (`.txt` files) corresponding to each frame.

- **`--save-frames`**: Enable this flag to save individual video frames as image files (`.jpg`) in the designated output directory.

## Interactive Controls

While the script is running, you can interact with the video segmentation process using the following keyboard controls:

| Key            | Action                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `ESC`          | Exit the application.                                                                            |
| `p`            | Toggle pause/play of the video.                                                                  |
| `a`            | Save the current object via the pipeline.                                                       |
| `r`            | Re-run inference on the current frame using the provided annotation points.                     |
| `Left Arrow`   | Navigate to the previous frame (only applicable when processing a video file).                   |
| `Right Arrow`  | Navigate to the next frame (only applicable when processing a video file).                       |
| `f`            | Move forward by one frame (only applicable when processing a video file).                        |
| `b`            | Move backward by one frame (only applicable when processing a video file).                       |
| `Left Click`   | Add a positive point for segmentation annotation.                                               |
| `Right Click`  | Add a negative point for segmentation annotation.                                               |

**Mouse Interactions:**

- **Left Click (`LBUTTON`)**: Add a positive annotation point on the video frame. These points guide the segmentation model to include specific areas.

- **Right Click (`RBUTTON`)**: Add a negative annotation point on the video frame. These points guide the segmentation model to exclude specific areas.

## Examples

### Processing a Video File and Saving Outputs

```bash
python demo/sam_cutie_video_segmentation.py --video videos/sample_video.mp4 --save-yolo --save-frames
```

This command will process `sample_video.mp4`, save YOLO-formatted bounding boxes for each frame, and store all extracted frames in the `output` directory.

### Using the Webcam Without Saving Outputs

```bash
python demo/sam_cutie_video_segmentation.py
```

This command will activate the webcam for live video segmentation without saving any outputs.

### Using the Webcam and Saving YOLO Bounding Boxes

```bash
python demo/sam_cutie_video_segmentation.py --save-yolo
```

This command will activate the webcam and save YOLO-formatted bounding boxes for each captured frame.

## Troubleshooting

- **Cannot Open Video/Webcam:**
  - Ensure the video file path is correct.
  - Verify that your webcam is properly connected and not being used by another application.

- **Missing Dependencies:**
  - Double-check that all required Python packages are installed in your virtual environment.
  - Install any missing packages using `pip`.

- **Performance Issues:**
  - Processing video in real-time can be resource-intensive. Ensure your system meets the necessary hardware requirements.
  - If using a GPU, verify that PyTorch is configured to utilize CUDA.
