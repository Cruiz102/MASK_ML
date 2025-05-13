#!/usr/bin/env python3
import os
import argparse
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on data generated from sam_cutie_video_segmentation tool")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output",
        help="Directory containing the YOLO formatted data (with yolo_bboxes subdirectory)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model to start training from (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (use -1 for auto-batch)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on (e.g., '0', '0,1', 'cpu', 'mps', or -1 for auto-select)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="sam_cutie_yolo_training",
        help="Project name for saving results"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name"
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Use pretrained model"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images to disk to reduce memory usage"
    )
    return parser.parse_args()


def setup_dataset_yaml(data_dir):
    """
    Create a YAML dataset configuration file based on the extracted frames and annotations
    """
    # Define paths relative to the data_dir
    frames_dir = os.path.join(data_dir, "frames")
    labels_dir = os.path.join(data_dir, "yolo_bboxes")
    
    if not os.path.exists(frames_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Missing frames or labels directories in {data_dir}. Make sure to run sam_cutie_video_segmentation.py with --save-frames and --save-yolo flags.")
    
    # Count image files and annotation files
    image_files = list(Path(frames_dir).glob("*.jpg"))
    if not image_files:
        image_files = list(Path(frames_dir).glob("*.png"))
    
    label_files = list(Path(labels_dir).glob("*.txt"))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    if not label_files:
        raise FileNotFoundError(f"No annotation files found in {labels_dir}")
    
    print(f"Found {len(image_files)} images and {len(label_files)} annotation files")
    
    # Check for existing YAML file and classes.txt
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    classes_path = os.path.join(data_dir, "classes.txt")
    
    # Default class is "object" unless specified in classes.txt
    names = ["object"]
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create train/val split (80/20)
    all_indices = list(range(len(image_files)))
    from random import shuffle
    shuffle(all_indices)
    split_idx = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    # Create train and val directories
    os.makedirs(os.path.join(data_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val", "labels"), exist_ok=True)
    
    # Copy files to train/val dirs
    for idx, img_path in enumerate(image_files):
        img_filename = img_path.name
        label_filename = img_filename.split('.')[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        if idx in train_indices:
            split = "train"
        else:
            split = "val"
            
        # Copy image
        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(data_dir, split, "images", img_filename))
            
        # Copy label if exists
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(data_dir, split, "labels", label_filename))
    
    # Create the YAML file
    dataset_config = {
        'path': os.path.abspath(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': '',  # No test set
        'names': {i: name for i, name in enumerate(names)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, sort_keys=False)
    
    print(f"Created dataset configuration at {yaml_path}")
    print(f"Classes: {names}")
    
    return yaml_path


def main():
    args = parse_args()
    
    # Setup the dataset
    yaml_path = setup_dataset_yaml(args.data_dir)
    
    # Handle device configuration
    device = args.device
    if device == "-1":
        device = -1  # Auto-select most idle GPU
    elif "," in device:
        device = [int(x) for x in device.split(",")]
    elif device.isdigit():
        device = int(device)
    
    # Load the model
    if args.resume:
        # Find most recent weight file in the project dir
        weights_dir = os.path.join(args.project, args.name)
        if os.path.exists(weights_dir):
            weight_files = list(Path(weights_dir).glob("*.pt"))
            if weight_files:
                latest_weight = max(weight_files, key=os.path.getctime)
                print(f"Resuming from {latest_weight}")
                model = YOLO(str(latest_weight))
                # Resume training
                results = model.train(resume=True)
                return
    
    # Load a new model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Reduce workers to avoid shared memory issues in Docker
    num_workers = min(args.workers, 2)
    print(f"Using {num_workers} workers to avoid memory issues")
    
    # Add cache option to reduce memory usage
    cache_option = 'disk' if args.cache else False
    
    # Train the model with reduced memory usage
    print(f"Starting training with {args.epochs} epochs on device {device}")
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        workers=num_workers,  # Reduce workers
        cache=cache_option,   # Use disk caching if enabled
    )
    
    print(f"Training complete! Results saved to {os.path.join(args.project, args.name)}")


if __name__ == "__main__":
    main()