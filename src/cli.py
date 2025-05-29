"""Command-line interface for NRHA Reining Score Predictor."""
import os
import json
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from src.data.video_processor import VideoProcessor
from src.preprocessing.pose_detector import PoseDetector
from src.utils.config import load_config, get_project_root
from src.data.pattern_processor import PatternProcessor


@click.group()
def cli():
    """NRHA Reining Score Predictor CLI."""
    pass


@cli.command()
@click.option('--year', '-y', default='2024', help='Year of patterns to download')
@click.option('--force/--no-force', '-f', default=False, help='Force re-download of existing patterns')
def download_patterns(year: str, force: bool):
    """Download and process NRHA patterns."""
    processor = PatternProcessor()
    
    try:
        patterns = processor.process_patterns(year)
        click.echo(f"Successfully processed {len(patterns)} patterns")
    except Exception as e:
        click.echo(f"Error processing patterns: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for processed data')
@click.option('--batch-size', '-b', type=int, default=32, help='Number of frames to process at once')
@click.option('--resume/--no-resume', default=True, help='Resume from last checkpoint if available')
def process_video(video_path: str, output_dir: Optional[str], batch_size: int, resume: bool):
    """Process a video file for reining analysis."""
    # Load configuration
    config_path = os.path.join(get_project_root(), 'configs', 'config.yaml')
    config = load_config(config_path)

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize processors
    video_processor = VideoProcessor(
        target_resolution=tuple(config['video']['target_resolution']),
        target_fps=config['video']['target_fps']
    )
    pose_detector = PoseDetector(
        min_detection_confidence=config['pose']['min_detection_confidence'],
        min_tracking_confidence=config['pose']['min_tracking_confidence']
    )

    # Set up checkpoint file
    video_name = Path(video_path).stem
    checkpoint_file = os.path.join(output_dir, f'{video_name}_checkpoint.json')
    
    # Load checkpoint if resuming
    processed_frames = 0
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_frames = checkpoint.get('processed_frames', 0)
            click.echo(f"Resuming from frame {processed_frames}")

    try:
        # Load video
        cap, properties = video_processor.load_video(video_path)
        total_frames = properties['frame_count']

        # Process video in batches
        with tqdm(total=total_frames, initial=processed_frames) as pbar:
            while True:
                frames = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)

                if not frames:
                    break

                # Process batch
                processed_frames = video_processor.process_video(frames)
                pose_results = pose_detector.process_video(processed_frames)

                # Save results
                batch_start = pbar.n
                batch_end = batch_start + len(frames)
                output_file = os.path.join(output_dir, f'{video_name}_frames_{batch_start:06d}_{batch_end:06d}.npy')
                np.save(output_file, pose_results)

                # Update checkpoint
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed_frames': batch_end}, f)

                pbar.update(len(frames))

        cap.release()
        click.echo("Video processing completed successfully!")

    except Exception as e:
        click.echo(f"Error processing video: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('data-dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for model checkpoints')
@click.option('--batch-size', '-b', type=int, default=32, help='Training batch size')
@click.option('--epochs', '-e', type=int, default=100, help='Number of training epochs')
@click.option('--resume/--no-resume', default=True, help='Resume training from last checkpoint')
def train(data_dir: str, output_dir: Optional[str], batch_size: int, epochs: int, resume: bool):
    """Train the reining analysis models."""
    # Load configuration
    config_path = os.path.join(get_project_root(), 'configs', 'config.yaml')
    config = load_config(config_path)

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), 'models', 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Implement training logic with checkpointing
    click.echo("Training functionality to be implemented")


@cli.command()
@click.argument('video-path', type=click.Path(exists=True))
@click.option('--model-dir', '-m', type=click.Path(), help='Directory containing trained models')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for predictions')
def predict(video_path: str, model_dir: Optional[str], output_dir: Optional[str]):
    """Predict scores for a reining run."""
    # Load configuration
    config_path = os.path.join(get_project_root(), 'configs', 'config.yaml')
    config = load_config(config_path)

    # Set up directories
    if model_dir is None:
        model_dir = os.path.join(get_project_root(), 'models', 'checkpoints')
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), 'data', 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Implement prediction logic
    click.echo("Prediction functionality to be implemented")


if __name__ == '__main__':
    cli() 