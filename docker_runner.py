#!/usr/bin/env python
import os
import sys
import docker
import argparse
import hydra
import subprocess
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Optional


class DockerRunner:
    """
    A class to handle Docker container operations for MASK_ML pipeline.
    Replaces the functionality of run_docker_demo.sh using the Docker Python API.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.client = docker.from_env()
        self.project_dir = Path(__file__).parent.absolute()
        
    def run_container(self, 
                      script_option: str, 
                      video_path: Optional[str] = None,
                      output_dir: Optional[str] = None,
                      additional_args: Optional[List[str]] = None):
        """
        Run the specified script in a Docker container
        
        Args:
            script_option: Which script to run (e.g., cutie_demo, train_yolo)
            video_path: Path to the video file (will be mounted)
            output_dir: Directory for output files
            additional_args: Additional arguments to pass to the script
        """
        # Use config values or defaults
        output_dir = output_dir or self.config.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.abspath(output_dir)
        
        # Prepare Docker volume mounts
        volumes = {
            output_path: {'bind': '/MASK_ML/output', 'mode': 'rw'},
            str(self.project_dir): {'bind': '/MASK_ML/volume', 'mode': 'rw'}
        }
        
        # Add video path as a volume if provided
        video_mount_arg = ""
        if video_path:
            video_path = os.path.abspath(video_path)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            video_dir = os.path.dirname(video_path)
            video_filename = os.path.basename(video_path)
            volumes[video_dir] = {'bind': '/input', 'mode': 'ro'}
            
            # Add video flag to additional args if not already there
            if not additional_args:
                additional_args = []
            if not any(arg.startswith('--video') for arg in additional_args):
                additional_args.extend(['--video', f'/input/{video_filename}'])
        
        # Allow local X server connections for GUI (for visualization)
        try:
            subprocess.run(['xhost', '+local:docker'], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: Could not set X server permissions. GUI might not work.")
        
        # Prepare environment variables
        environment = {
            'SCRIPT_OPTION': script_option,
            'DISPLAY': os.environ.get('DISPLAY', ''),
            'TASK_TYPE': self.config.data_engine_task
        }
        
        # Prepare command
        command = additional_args if additional_args else []
        
        print(f"Running with script option: {script_option}")
        print(f"Additional arguments: {' '.join(command) if command else 'None'}")
        
        try:
            # Run the container
            container = self.client.containers.run(
                'mask_ml_app',
                command=command,
                environment=environment,
                volumes=volumes,
                network_mode='host',  # For X11 forwarding
                remove=True,  # Remove the container after it exits
                detach=True,  # Run in the background
            )
            
            # Stream logs from the container
            for line in container.logs(stream=True, follow=True):
                print(line.decode('utf-8').strip())
                
            # Wait for the container to finish
            exit_code = container.wait()['StatusCode']
            if exit_code != 0:
                print(f"Container exited with non-zero status: {exit_code}")
                return exit_code
                
        except docker.errors.APIError as e:
            print(f"Docker API error: {e}")
            return 1
        except docker.errors.ContainerError as e:
            print(f"Container error: {e}")
            return 1
        except KeyboardInterrupt:
            print("Operation interrupted by user")
            print("Stopping container...")
            try:
                container.stop()
            except:
                pass
            return 130
        
        return 0


def show_help():
    """Display usage information"""
    help_text = """
    Docker Runner for MASK_ML
    
    This script runs the MASK_ML pipeline tasks in Docker containers
    using configuration from data_engine.yaml.
    
    Options:
      -h, --help                Show this help message
      -s, --script NAME         Specify which script to run (e.g., cutie_demo, train_yolo)
      -v, --video PATH          Specify video file path (will be mounted inside container)
      -o, --output-dir DIR      Specify output directory
    
    Common script options:
      - cutie_demo: Run SAM Cutie video segmentation tool
      - train_yolo: Run Ultralytics training on data generated by SAM Cutie
      - volume_cutie_demo: Run SAM Cutie from the volume directory
      - volume_ultralytics_training: Run Ultralytics training from the volume directory
    
    Examples:
      # Run the SAM Cutie tool and save frames & YOLO annotations:
      python docker_runner.py -s cutie_demo -- --save-yolo
    
      # Run Ultralytics training on the generated data:
      python docker_runner.py -s train_yolo -- --data-dir output --model yolov8n.pt
    
      # Run the SAM Cutie tool with a specific video file:
      python docker_runner.py -s cutie_demo -v /path/to/video.mp4 -- --save-yolo
    """
    print(help_text)


@hydra.main(version_base=None, config_path="config", config_name="data_engine")
def main(cfg: DictConfig) -> int:
    """
    Main entry point for the Docker runner.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Exit code (0 for success)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MASK_ML Docker containers', add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Show help message')
    parser.add_argument('-s', '--script', default='cutie_demo', help='Script to run')
    parser.add_argument('-v', '--video', help='Path to video file')
    parser.add_argument('-o', '--output-dir', help='Output directory')
    
    # Parse known args first, to separate from the args after --
    args, additional_args = parser.parse_known_args()
    
    # If help is requested, show help and exit
    if args.help:
        show_help()
        return 0
    
    # Find the -- separator index and extract arguments after it
    script_args = []
    for i, arg in enumerate(additional_args):
        if arg == '--':
            script_args = additional_args[i+1:]
            break
    
    # Update output directory from arguments if provided
    output_dir = args.output_dir or cfg.output_dir
    
    # Check if Docker should be used
    if not cfg.use_docker:
        print("Docker is disabled in config. Running scripts directly...")
        # Would need to implement direct script running here
        # For now, just print a message
        print("Direct script execution not implemented yet.")
        return 1
    
    # Create and run the Docker runner
    runner = DockerRunner(cfg)
    return runner.run_container(
        script_option=args.script,
        video_path=args.video or cfg.video_dir,
        output_dir=output_dir,
        additional_args=script_args
    )


if __name__ == "__main__":
    # Set OmegaConf to resolve variables
    OmegaConf.register_resolver("env", lambda name, default=None: os.environ.get(name, default))
    sys.exit(main())