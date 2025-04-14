import cv2
import numpy as np
import os
from pathlib import Path

def create_side_by_side_comparison(pro_video_path, amateur_video_path, output_path, output_fps=10):
    """
    Create a simple side-by-side video comparison of pro and amateur tennis players.
    
    Args:
        pro_video_path: Path to the professional player's video
        amateur_video_path: Path to the amateur player's video
        output_path: Path to save the side-by-side comparison video
        output_fps: Frame rate for the output video (default: 10)
    """
    # Open video files
    pro_cap = cv2.VideoCapture(pro_video_path)
    amateur_cap = cv2.VideoCapture(amateur_video_path)
    
    # Check if videos opened successfully
    if not pro_cap.isOpened():
        print(f"Error: Could not open professional video at {pro_video_path}")
        return
    if not amateur_cap.isOpened():
        print(f"Error: Could not open amateur video at {amateur_video_path}")
        return
    
    # Get video properties
    pro_width = int(pro_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pro_height = int(pro_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    amateur_width = int(amateur_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    amateur_height = int(amateur_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine target height (use the larger of the two)
    target_height = max(pro_height, amateur_height)
    
    # Calculate new widths maintaining aspect ratio
    pro_new_width = int(pro_width * (target_height / pro_height))
    amateur_new_width = int(amateur_width * (target_height / amateur_height))
    
    # Combined frame dimensions
    combined_width = pro_new_width + amateur_new_width
    
    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (combined_width, target_height))
    
    # Process frames
    frame_count = 0
    while True:
        # Read frames from both videos
        pro_ret, pro_frame = pro_cap.read()
        amateur_ret, amateur_frame = amateur_cap.read()
        
        # Break if either video ends
        if not pro_ret or not amateur_ret:
            break
        
        # Resize frames preserving aspect ratio
        pro_frame_resized = cv2.resize(pro_frame, (pro_new_width, target_height))
        amateur_frame_resized = cv2.resize(amateur_frame, (amateur_new_width, target_height))
        
        # Create combined frame
        combined_frame = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
        combined_frame[0:target_height, 0:pro_new_width] = pro_frame_resized
        combined_frame[0:target_height, pro_new_width:combined_width] = amateur_frame_resized
        
        # Add labels
        cv2.putText(combined_frame, "Professional", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, "Amateur", (pro_new_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add frame counter
        cv2.putText(combined_frame, f"Frame: {frame_count}", (combined_width // 2 - 70, target_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to output
        out.write(combined_frame)
        
        # Update frame counter
        frame_count += 1
        
        # Show progress every 50 frames
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    pro_cap.release()
    amateur_cap.release()
    out.release()
    
    print(f"Side-by-side comparison completed. Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")

def main():
    # Set video paths
    pro_video_path = "jannik-sinner-forehands.mp4"
    amateur_video_path = "amateur-player-forehands.mov"
    
    # Create output directory
    output_dir = Path("tennis_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Set output path
    output_path = str(output_dir / "simple_side_by_side.mp4")
    
    # Create side-by-side comparison
    print("Creating side-by-side comparison...")
    create_side_by_side_comparison(pro_video_path, amateur_video_path, output_path)
    print("Done!")

if __name__ == "__main__":
    main()