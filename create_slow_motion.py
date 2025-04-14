import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def create_slow_motion(input_path, output_path, speed_factor=0.25, target_fps=60):
    """
    Create a slow-motion version of a video while maintaining or 
    increasing the frame rate for smooth playback.
    
    Args:
        input_path: Path to input video
        output_path: Path to save slow-motion video
        speed_factor: How much to slow down (0.5 = half speed, 0.25 = quarter speed)
        target_fps: Target frame rate for output (higher = smoother slow motion)
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {input_path}")
        return False
    
    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / input_fps
    
    # Calculate new duration and needed frames
    new_duration = duration / speed_factor
    output_frame_count = int(new_duration * target_fps)
    
    print(f"Input: {input_path}")
    print(f"  Frame rate: {input_fps} fps")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Frames: {frame_count}")
    
    print(f"Output: {output_path}")
    print(f"  Speed: {speed_factor:.2f}x (slowed down)")
    print(f"  Frame rate: {target_fps} fps")
    print(f"  Duration: {new_duration:.2f} seconds")
    print(f"  Frames to generate: {output_frame_count}")
    
    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Generate frame indices for smooth interpolation
    input_indices = np.linspace(0, frame_count - 1, output_frame_count)
    
    # Process frames
    prev_frame = None
    prev_index = -1
    
    for i, idx in enumerate(input_indices):
        current_index = int(idx)
        fraction = idx - current_index  # For interpolation
        
        # Optimization: Only seek when necessary
        if current_index != prev_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
            ret, current_frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame at index {current_index}")
                continue
            
            # If we need the next frame for interpolation and it's not the last frame
            if fraction > 0 and current_index < frame_count - 1:
                ret, next_frame = cap.read()
                if ret:
                    # Linear interpolation between frames for smoother slow motion
                    frame = cv2.addWeighted(current_frame, 1 - fraction, next_frame, fraction, 0)
                else:
                    frame = current_frame
            else:
                frame = current_frame
                
            prev_frame = current_frame
            prev_index = current_index
        else:
            frame = prev_frame
        
        # Write frame
        out.write(frame)
        
        # Show progress
        if i % 100 == 0:
            print(f"  Processed {i}/{output_frame_count} frames ({i/output_frame_count*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Slow motion video saved to {output_path}")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create slow-motion videos for tennis analysis')
    parser.add_argument('--pro_video', default="jannik-sinner-forehands.mp4", help='Path to professional player video')
    parser.add_argument('--amateur_video', default="amateur-player-forehands.mov", help='Path to amateur player video')
    parser.add_argument('--speed_factor', type=float, default=0.25, help='Speed factor (lower = slower, e.g., 0.25 = quarter speed)')
    parser.add_argument('--target_fps', type=int, default=60, help='Target frame rate for smooth slow motion')
    parser.add_argument('--output_dir', default="slow_motion", help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create slow motion videos
    pro_output = str(output_dir / "pro_slow_motion.mp4")
    amateur_output = str(output_dir / "amateur_slow_motion.mp4")
    
    print("\nCreating slow motion pro video...")
    create_slow_motion(args.pro_video, pro_output, args.speed_factor, args.target_fps)
    
    print("\nCreating slow motion amateur video...")
    create_slow_motion(args.amateur_video, amateur_output, args.speed_factor, args.target_fps)
    
    print("\nSlowdown complete!")
    print(f"Slow motion videos saved to:")
    print(f"  Pro: {pro_output}")
    print(f"  Amateur: {amateur_output}")
    
    # Create side-by-side comparison script
    comparison_script = """
import cv2
import numpy as np
from pathlib import Path

def create_side_by_side(pro_path, amateur_path, output_path, output_fps=60):
    # Open videos
    pro_cap = cv2.VideoCapture(pro_path)
    amateur_cap = cv2.VideoCapture(amateur_path)
    
    if not pro_cap.isOpened() or not amateur_cap.isOpened():
        print("Error opening videos")
        return
    
    # Get dimensions
    pro_width = int(pro_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pro_height = int(pro_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    amateur_width = int(amateur_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    amateur_height = int(amateur_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create consistent height
    target_height = 720
    pro_new_width = int(pro_width * (target_height / pro_height))
    amateur_new_width = int(amateur_width * (target_height / amateur_height))
    
    # Combined width
    combined_width = pro_new_width + amateur_new_width
    
    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (combined_width, target_height))
    
    frame_count = 0
    while True:
        pro_ret, pro_frame = pro_cap.read()
        amateur_ret, amateur_frame = amateur_cap.read()
        
        if not pro_ret or not amateur_ret:
            break
        
        # Resize frames
        pro_resized = cv2.resize(pro_frame, (pro_new_width, target_height))
        amateur_resized = cv2.resize(amateur_frame, (amateur_new_width, target_height))
        
        # Combine frames
        combined = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
        combined[0:target_height, 0:pro_new_width] = pro_resized
        combined[0:target_height, pro_new_width:combined_width] = amateur_resized
        
        # Add labels
        cv2.putText(combined, "Professional", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Amateur", (pro_new_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame
        out.write(combined)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    pro_cap.release()
    amateur_cap.release()
    out.release()
    print(f"Side-by-side comparison saved to {output_path}")

if __name__ == "__main__":
    pro_path = "slow_motion/pro_slow_motion.mp4"
    amateur_path = "slow_motion/amateur_slow_motion.mp4"
    output_path = "slow_motion/side_by_side_slow_motion.mp4"
    
    print("Creating side-by-side comparison of slow motion videos...")
    create_side_by_side(pro_path, amateur_path, output_path)
    print("Done!")
"""
    
    # Write comparison script
    with open(str(output_dir / "create_comparison.py"), "w") as f:
        f.write(comparison_script)
    
    print(f"\nA script to create a side-by-side comparison has been saved to:")
    print(f"  {output_dir / 'create_comparison.py'}")
    print(f"\nRun it with: python {output_dir / 'create_comparison.py'}")

if __name__ == "__main__":
    main()