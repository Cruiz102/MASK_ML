import argparse
import os
import cv2
import numpy as np
import torch
from typing import List

# 3rd-Party / Local imports
from third_party.sam_cutie import SamCutiePipeline
from mask_ml.utils.annotations import DrawAnnotator
from utils import BBoxOutput, generate_random_rgb, to_bbox

# -----------------------------------
# Global variables
# -----------------------------------
points = []
points_labels = []
objects_colors = {}
pipeline = SamCutiePipeline()

# For demonstration, we keep track of bounding boxes and masks in these dictionaries:
#   cached_bboxes[frame_idx] = [list of BBoxOutput]
#   cached_masks[frame_idx] = torch.Tensor or np.ndarray mask
cached_bboxes = {}
cached_masks = {}

# -----------------------------------
# Argparse for command-line options
# -----------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SAM Cutie Extended App")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store optional outputs (e.g., YOLO bounding boxes, frames)."
    )
    parser.add_argument(
        "--save-yolo",
        action="store_true",
        help="If set, save bounding boxes in YOLO format for each frame."
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="If set, store extracted frames in the output folder."
    )
    return parser.parse_args()


# -----------------------------------
# Mouse callback to collect positive/negative points
# -----------------------------------
def update_global_variables(event, x, y, flags, param):
    global points, points_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        points_labels.append(1)  # Positive
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append((x, y))
        points_labels.append(-1)  # Negative


# -----------------------------------
# Keep track of unique object colors
# -----------------------------------
def update_objects(bbox: List[BBoxOutput]):
    global objects_colors
    for box in bbox:
        if box.track_id not in objects_colors:
            objects_colors[box.track_id] = generate_random_rgb()


# -----------------------------------
# Helper: Save YOLO bounding boxes
# -----------------------------------
def save_yolo_bboxes(
    bboxes: List[BBoxOutput], 
    output_dir: str, 
    frame_idx: int,
    img_width: int, 
    img_height: int
):
    """
    Save bounding boxes in YOLO format:
      class_id x_center y_center width height
    All values normalized (0..1).
    """
    os.makedirs(output_dir, exist_ok=True)
    yolo_class_id = 0  
    yolo_filename = os.path.join(output_dir, f"{frame_idx:06d}.txt")
    with open(yolo_filename, "w") as f:
        for box in bboxes:
            x_center = (box.x1 + box.x2) / 2.0
            y_center = (box.y1 + box.y2) / 2.0
            width = box.x2 - box.x1
            height = box.y2 - box.y1

            # Normalize
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# -----------------------------------
# Main App Logic
# -----------------------------------
@torch.inference_mode()
@torch.cuda.amp.autocast()
def run_app(args):
    global points, points_labels, pipeline

    annotator = DrawAnnotator()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video {args.video}")
        return

    # Read all frames into a list so we can navigate back and forth
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    num_frames = len(frames)
    if num_frames == 0:
        print("No frames found in video.")
        return

    # Create output directories if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    frames_dir = os.path.join(args.output_dir, "frames")
    if args.save_frames and not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Setup display window
    cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("demo", update_global_variables)

    # State for playback
    frame_idx = 0
    paused = False

    object_counter = 0

    while True:
        # Ensure frame_idx is in valid range
        frame_idx = max(0, min(frame_idx, num_frames - 1))
        image = frames[frame_idx].copy()

        # Optionally save the extracted frame to disk
        if args.save_frames:
            frame_filename = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
            # You could check if file already exists, but we'll just overwrite
            cv2.imwrite(frame_filename, image)

        # Retrieve cached bboxes/masks if they exist
        # otherwise, run pipeline (only if pipeline.objects exist, i.e., known objects)
        if frame_idx in cached_bboxes and frame_idx in cached_masks:
            detected_bbox = cached_bboxes[frame_idx]
            mask = cached_masks[frame_idx]
        else:
            detected_bbox = []
            mask = None
            if pipeline.objects:
                # Predict mask for current frame
                mask = pipeline.predict_mask(image)
                # Convert mask to bounding boxes
                detected_bbox = to_bbox(mask)
                # Update objects color dictionary
                update_objects(detected_bbox)
                # Cache them
                cached_bboxes[frame_idx] = detected_bbox
                cached_masks[frame_idx] = mask

        # If desired, save bounding boxes in YOLO format
        if args.save_yolo and len(detected_bbox) > 0:
            height, width, _ = image.shape
            yolo_dir = os.path.join(args.output_dir, "yolo_bboxes")
            save_yolo_bboxes(detected_bbox, yolo_dir, frame_idx, width, height)

        # If we got boxes, draw them
        if detected_bbox:
            image = annotator.draw_bbox(
                image,
                detected_bbox,
                object_color=objects_colors,
                draw_mask=True
            )

        # Draw the positive/negative “click” points
        image_with_circles = image.copy()
        for i, point in enumerate(points):
            color = (255, 0, 0) if points_labels[i] == 1 else (0, 0, 255)
            cv2.circle(image_with_circles, point, 5, color, 2)

        # Show final image
        if len(points) > 0:
            cv2.imshow("demo", image_with_circles)
        else:
            cv2.imshow("demo", image)

        # Key handling
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('p'):
            # Toggle paused
            paused = not paused
        elif key == ord('a'):
            # Save object via pipeline
            pipeline.save_object(
                f"object-{object_counter}",
                frames[frame_idx],  # current raw frame
                points=np.array(points),
                point_labels=points_labels
            )
            points.clear()
            points_labels.clear()
            object_counter += 1

        # Navigation: left/right arrows
        elif key == 81:  # left arrow
            frame_idx -= 1
        elif key == 83:  # right arrow
            frame_idx += 1

        # Optional: If you want keys like 'f' to step forward, 'b' to step backward:
        elif key == ord('f'):  # forward 1 frame
            frame_idx += 1
        elif key == ord('b'):  # backward 1 frame
            frame_idx -= 1

        else:
            # Some other key, do nothing
            pass

        # If not paused, just advance one frame
        if not paused:
            frame_idx += 1

        if frame_idx >= num_frames:
            # We reached the end
            print("End of video.")
            break

    cv2.destroyAllWindows()


# -----------------------------------
# Script entrypoint
# -----------------------------------
if __name__ == '__main__':
    args = parse_args()
    run_app(args)
