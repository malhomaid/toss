import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def extract_video_info(video_path):
    """Extract frame rate and duration information from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

def create_adjusted_video(input_path, output_path, target_fps=30, target_duration=None, playback_speed=1.0):
    """
    Create a new video with adjusted frame rate and duration.
    
    Args:
        input_path: Path to input video
        output_path: Path to save adjusted video
        target_fps: Target frame rate for output video
        target_duration: Optional target duration (seconds)
        playback_speed: Speed factor (>1 speeds up, <1 slows down)
    """
    # Get video info
    video_info = extract_video_info(input_path)
    if not video_info:
        return False
    
    # Calculate effective playback speed
    effective_speed = playback_speed
    if target_duration:
        # Adjust speed to match target duration
        effective_speed = video_info['duration'] / target_duration * playback_speed
    
    print(f"Adjusting {input_path}")
    print(f"  Original: {video_info['fps']:.2f} fps, {video_info['duration']:.2f} seconds")
    print(f"  Effective playback speed: {effective_speed:.2f}x")
    
    # Calculate frame sampling parameters
    input_cap = cv2.VideoCapture(input_path)
    input_frame_count = video_info['frame_count']
    
    # Calculate how many frames to generate
    if target_duration:
        output_frame_count = int(target_duration * target_fps)
    else:
        output_frame_count = int(video_info['duration'] / effective_speed * target_fps)
    
    print(f"  Output: {target_fps} fps, {output_frame_count} frames, {output_frame_count/target_fps:.2f} seconds")
    
    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, 
                          (video_info['width'], video_info['height']))
    
    # Create frame indices for sampling (accounting for playback speed)
    input_indices = np.linspace(0, input_frame_count - 1, output_frame_count)
    
    # Read and write frames
    for i, frame_idx in enumerate(input_indices):
        # Set frame position
        input_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = input_cap.read()
        
        if not ret:
            print(f"Warning: Failed to read frame at index {int(frame_idx)}")
            continue
        
        # Write frame to output
        out.write(frame)
        
        # Show progress
        if i % 30 == 0:
            print(f"  Processed {i}/{output_frame_count} frames")
    
    # Release resources
    input_cap.release()
    out.release()
    
    print(f"Adjusted video saved to {output_path}")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Adjust video frame rates for tennis analysis')
    parser.add_argument('--pro_video', default="jannik-sinner-forehands.mp4", help='Path to professional player video')
    parser.add_argument('--amateur_video', default="amateur-player-forehands.mov", help='Path to amateur player video')
    parser.add_argument('--target_fps', type=int, default=30, help='Target frame rate for output videos')
    parser.add_argument('--target_duration', type=float, default=None, help='Target duration for both videos (seconds)')
    parser.add_argument('--output_dir', default="adjusted_videos", help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract info from both videos
    print("Analyzing videos...")
    pro_info = extract_video_info(args.pro_video)
    amateur_info = extract_video_info(args.amateur_video)
    
    if not pro_info or not amateur_info:
        print("Error analyzing videos. Exiting.")
        return
    
    print("\nVideo Information:")
    print(f"Pro: {pro_info['fps']:.2f} fps, {pro_info['duration']:.2f} seconds")
    print(f"Amateur: {amateur_info['fps']:.2f} fps, {amateur_info['duration']:.2f} seconds")
    
    # Determine target duration if not specified
    if not args.target_duration:
        # Use the longer of the two durations as target
        target_duration = max(pro_info['duration'], amateur_info['duration'])
    else:
        target_duration = args.target_duration
    
    print(f"\nTarget: {args.target_fps} fps, {target_duration:.2f} seconds")
    
    # Create adjusted videos
    pro_output = str(output_dir / "pro_adjusted.mp4")
    amateur_output = str(output_dir / "amateur_adjusted.mp4")
    
    print("\nCreating adjusted pro video...")
    create_adjusted_video(args.pro_video, pro_output, args.target_fps, target_duration)
    
    print("\nCreating adjusted amateur video...")
    create_adjusted_video(args.amateur_video, amateur_output, args.target_fps, target_duration)
    
    print("\nAdjustment complete!")
    print(f"Adjusted videos saved to:")
    print(f"  Pro: {pro_output}")
    print(f"  Amateur: {amateur_output}")
    print(f"\nYou can now use these videos with your analysis scripts.")

if __name__ == "__main__":
    main()