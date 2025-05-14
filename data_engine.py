import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from hydra.utils import instantiate
from docker_runner import DockerRunner


def run_video_segmentation(cfg: DictConfig):
    """
    Run video segmentation using SAM Cutie
    
    Args:
        cfg: Configuration from Hydra
    """
    print("Running video segmentation using SAM Cutie")
    
    output_dir = Path(cfg.output_dir)
    video_dir = Path(cfg.video_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if cfg.use_docker:
        docker_runner = DockerRunner(cfg)
        video_path = str(video_dir) if video_dir.is_file() else None
        
        # Build args from config
        additional_args = ["--save-yolo"]
        if cfg.segmentation.save_frames:
            additional_args.append("--save-frames")
        if cfg.segmentation.visualize:
            additional_args.append("--visualize")
        
        additional_args.extend([
            "--confidence-threshold", str(cfg.segmentation.confidence_threshold),
            "--mask-threshold", str(cfg.segmentation.mask_threshold),
            "--nms-threshold", str(cfg.segmentation.nms_threshold)
        ])
        
        if cfg.segmentation.frames_per_second > 0:
            additional_args.extend(["--fps", str(cfg.segmentation.frames_per_second)])
        
        return docker_runner.run_container(
            script_option="cutie_demo",
            video_path=video_path,
            output_dir=str(output_dir),
            additional_args=additional_args
        )
    else:
        # Direct execution path (not through Docker)
        try:
            from demo.sam_cutie_video_segmentation import main as cutie_main
            cutie_main(
                video_path=str(video_dir), 
                output_dir=str(output_dir),
                save_yolo=cfg.segmentation.save_yolo,
                save_frames=cfg.segmentation.save_frames,
                visualize=cfg.segmentation.visualize,
                confidence_threshold=cfg.segmentation.confidence_threshold,
                mask_threshold=cfg.segmentation.mask_threshold,
                nms_threshold=cfg.segmentation.nms_threshold,
                fps=cfg.segmentation.frames_per_second
            )
        except ImportError:
            print("Could not import SAM Cutie segmentation module.")
            print("Please ensure demo/sam_cutie_video_segmentation.py is available.")
            return 1
    
    return 0


def run_model_training(cfg: DictConfig):
    """
    Run model training on segmented data
    
    Args:
        cfg: Configuration from Hydra
    """
    print("Running model training on segmented data")
    
    output_dir = Path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if cfg.use_docker:
        docker_runner = DockerRunner(cfg)
        
        # Build args from config
        additional_args = [
            "--data-dir", "output", 
            "--model", f"yolov8{cfg.training.model_size}.pt",
            "--epochs", str(cfg.training.epochs),
            "--batch-size", str(cfg.training.batch_size),
            "--img", str(cfg.training.image_size),
            "--lr", str(cfg.training.learning_rate),
            "--patience", str(cfg.training.patience)
        ]
        
        return docker_runner.run_container(
            script_option="train_yolo",
            output_dir=str(output_dir),
            additional_args=additional_args
        )
    else:
        # Direct execution path (not through Docker)
        try:
            from demo.ultralytics_training import main as training_main
            training_main(
                data_dir=str(output_dir),
                model=f"yolov8{cfg.training.model_size}.pt",
                epochs=cfg.training.epochs,
                batch_size=cfg.training.batch_size,
                image_size=cfg.training.image_size,
                learning_rate=cfg.training.learning_rate,
                patience=cfg.training.patience
            )
        except ImportError:
            print("Could not import Ultralytics training module.")
            print("Please ensure demo/ultralytics_training.py is available.")
            return 1
    
    return 0


def run_data_pipeline(cfg: DictConfig):
    """
    Main function to run the data processing pipeline based on the configuration.
    
    Args:
        cfg: Configuration from Hydra
        
    Returns:
        Exit code (0 for success)
    """
    print(f"Starting data engine with task: {cfg.data_engine_task}")
    
    # Set random seed for reproducibility
    if cfg.advanced.seed:
        torch.manual_seed(cfg.advanced.seed)
        if torch.cuda.is_available() and cfg.advanced.gpu_enabled:
            torch.cuda.manual_seed(cfg.advanced.seed)
    
    # Set debug mode if enabled
    if cfg.advanced.debug_mode:
        print("Debug mode enabled. Verbose output will be shown.")
    
    # Run the appropriate task
    if cfg.data_engine_task == "video_segmentation":
        return run_video_segmentation(cfg)
    elif cfg.data_engine_task == "model_training":
        return run_model_training(cfg)
    elif cfg.data_engine_task == "full_pipeline":
        # Run segmentation first
        result = run_video_segmentation(cfg)
        if result != 0:
            return result
        
        # If segmentation succeeded, run training
        return run_model_training(cfg)
    else:
        print(f"Error: Unknown task type: {cfg.data_engine_task}")
        return 1


@hydra.main(version_base=None, config_path="config", config_name="data_engine")
def main(cfg: DictConfig) -> int:
    """
    Main entry point for the data engine.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Exit code (0 for success)
    """
    # Print configuration for debugging
    print(OmegaConf.to_yaml(cfg))
    
    # Run the data processing pipeline
    return run_data_pipeline(cfg)


if __name__ == "__main__":
    main()

