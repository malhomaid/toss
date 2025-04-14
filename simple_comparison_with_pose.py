import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
from pathlib import Path

class SimpleTennisComparison:
    def __init__(self, model_path='pose_landmarker.task'):
        """Initialize with MediaPipe pose detection."""
        # Set up the PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Create detector
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Colors for visualization
        self.pro_color = (0, 255, 0)  # Green for pro player
        self.amateur_color = (0, 0, 255)  # Red for amateur player

        # Key joint indices for tennis forehand analysis
        self.key_joints = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }

    def draw_landmarks_on_image(self, rgb_image, detection_result, color):
        """Draw landmarks on frame with specified color."""
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in pose_landmarks
            ])

            # Custom drawing style with player-specific color
            custom_style = solutions.drawing_styles.get_default_pose_landmarks_style()
            for connection_style in custom_style.values():
                connection_style.color = color

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                custom_style)

            # Highlight key joints specifically (larger circles)
            height, width, _ = annotated_image.shape
            for joint_name, joint_idx in self.key_joints.items():
                if joint_idx < len(pose_landmarks):
                    joint = pose_landmarks[joint_idx]
                    joint_x = int(joint.x * width)
                    joint_y = int(joint.y * height)

                    # Draw a larger circle around key joints
                    circle_size = 10 if 'knee' in joint_name or 'wrist' in joint_name or 'elbow' in joint_name else 8
                    cv2.circle(annotated_image, (joint_x, joint_y), circle_size, color, -1)

        return annotated_image

    def process_frame(self, frame, color):
        """Process a single frame with pose detection."""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect pose landmarks
        detection_result = self.detector.detect(mp_image)

        # Draw landmarks if detected
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            annotated_frame = self.draw_landmarks_on_image(frame_rgb, detection_result, color)
            return cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            # Return original frame if no pose detected
            return frame

    def create_comparison(self, pro_video_path, amateur_video_path, output_path, output_fps=10, skip_frames=2):
        """
        Create side-by-side comparison with pose detection.

        Args:
            pro_video_path: Path to professional player video
            amateur_video_path: Path to amateur player video
            output_path: Path to save the output video
            output_fps: Frame rate for output video
            skip_frames: Process every nth frame to speed up processing
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
        pro_total_frames = int(pro_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        amateur_total_frames = int(amateur_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Pro video: {pro_width}x{pro_height}, {pro_total_frames} frames")
        print(f"Amateur video: {amateur_width}x{amateur_height}, {amateur_total_frames} frames")

        # Determine target height (use the larger of the two)
        target_height = 720  # Fixed target height for consistency

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

            # Skip frames for faster processing
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            # Process frames with pose detection
            pro_processed = self.process_frame(pro_frame, self.pro_color)
            amateur_processed = self.process_frame(amateur_frame, self.amateur_color)

            # Resize frames preserving aspect ratio
            pro_frame_resized = cv2.resize(pro_processed, (pro_new_width, target_height))
            amateur_frame_resized = cv2.resize(amateur_processed, (amateur_new_width, target_height))

            # Create combined frame
            combined_frame = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
            combined_frame[0:target_height, 0:pro_new_width] = pro_frame_resized
            combined_frame[0:target_height, pro_new_width:combined_width] = amateur_frame_resized

            # Add labels
            cv2.putText(combined_frame, "Professional", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.pro_color, 2)
            cv2.putText(combined_frame, "Amateur", (pro_new_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.amateur_color, 2)

            # Add frame counter
            cv2.putText(combined_frame, f"Frame: {frame_count}", (combined_width // 2 - 70, target_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame to output
            out.write(combined_frame)

            # Show progress
            if frame_count % 10 == 0:
                print(f"Processed frame {frame_count}")

            frame_count += 1

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
    output_path = str(output_dir / "pose_comparison.mp4")

    # Create side-by-side comparison
    print("Creating side-by-side comparison with pose detection...")
    analyzer = SimpleTennisComparison()
    analyzer.create_comparison(pro_video_path, amateur_video_path, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
