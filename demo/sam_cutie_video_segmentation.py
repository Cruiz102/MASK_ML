import argparse
import os
import cv2
import numpy as np
import torch
from typing import List, Union

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
        default=None,  # Make it optional
        help="Path to the video file. If not provided, the webcam will be used."
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
@torch.amp.autocast(device_type='cuda', enabled=True)  # Updated autocast
def run_app(args):
    global points, points_labels, pipeline, cached_bboxes, cached_masks

    annotator = DrawAnnotator()
    
    # Initialize VideoCapture based on whether a video path is provided
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source = f"video file {args.video}"
    else:
        cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
        source = "webcam"
    
    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return

    # Read all frames into a list if using video file
    frames = []
    if args.video:
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
    else:
        # For webcam, we'll read frames on-the-fly
        pass  # No need to preload frames

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
        if args.video:
            # Ensure frame_idx is in valid range
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            image = frames[frame_idx].copy()
        else:
            if not paused:
                ret, image = cap.read()
                if not ret:
                    print("Failed to grab frame from webcam.")
                    break
            # When paused, keep displaying the last frame
            # No need to manage frame_idx for webcam

        # Optionally save the extracted frame to disk
        if args.save_frames and args.video:
            frame_filename = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_filename, image)

        # See if we have cached results
        if args.video:
            if frame_idx in cached_bboxes and frame_idx in cached_masks:
                detected_bbox = cached_bboxes[frame_idx]
                mask = cached_masks[frame_idx]
            else:
                detected_bbox = []
                mask = None
                # Only run auto-inference if we already have known objects
                if pipeline.objects:
                    mask = pipeline.predict_mask(image)
                    detected_bbox = to_bbox(mask)
                    update_objects(detected_bbox)

                    # Cache
                    cached_bboxes[frame_idx] = detected_bbox
                    cached_masks[frame_idx] = mask

            # If we got boxes, optionally save YOLO & draw them
            if detected_bbox and args.save_yolo:
                height, width, _ = image.shape
                yolo_dir = os.path.join(args.output_dir, "yolo_bboxes")
                save_yolo_bboxes(detected_bbox, yolo_dir, frame_idx, width, height)

            if detected_bbox:
                image = annotator.draw_bbox(
                    image,
                    detected_bbox,
                    object_color=objects_colors,
                    draw_mask=True
                )
        else:
            # Webcam: handle caching differently or skip caching
            detected_bbox = []
            mask = None
            # Example: perform detection on each frame if pipeline has objects
            if pipeline.objects:
                mask = pipeline.predict_mask(image)
                detected_bbox = to_bbox(mask)
                update_objects(detected_bbox)

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

        if key == 27:  # ESC to quit
            break

        elif key == ord('p'):
            # Toggle paused
            paused = not paused

        elif key == ord('a'):
            # Save object via pipeline (persists object in the pipeline)
            pipeline.save_object(
                f"object-{object_counter}",
                image,  # current frame (webcam or video)
                points=np.array(points),
                point_labels=points_labels
            )
            points.clear()
            points_labels.clear()
            object_counter += 1

        elif key == ord('r'):
            # Re-run inference on the *current frame* using the current points
            if paused and len(points) > 0:
                ultra_sam_prediction = pipeline.sam_model.predict(
                    image, points=points, labels=points_labels
                )
                re_mask = ultra_sam_prediction[0].masks.data.to(dtype=torch.int).cpu().numpy()[0]
                re_bboxes = to_bbox(re_mask)
                update_objects(re_bboxes)

                if args.video:
                    cached_bboxes[frame_idx] = re_bboxes
                    cached_masks[frame_idx] = re_mask
                else:
                    # For webcam, you might handle caching differently or skip
                    pass

        # Navigation keys for video files
        if args.video:
            if key == 81:  # left arrow
                frame_idx -= 1
            elif key == 83:  # right arrow
                frame_idx += 1
            elif key == ord('f'):  # forward 1 frame
                frame_idx += 1
            elif key == ord('b'):  # backward 1 frame
                frame_idx -= 1
            # Auto-advance for video files
            if not paused:
                frame_idx += 1
            if frame_idx >= num_frames:
                print("End of video.")
                break
        else:
            # For webcam, handle frame capture differently
            if not paused:
                pass  # Already capturing frames in the loop

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------
# Script entrypoint
# -----------------------------------
if __name__ == '__main__':
    args = parse_args()
    run_app(args)
