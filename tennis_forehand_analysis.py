import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import math

class TennisSwingAnalyzer:
    def __init__(self, model_path='pose_landmarker.task'):
        """Initialize the tennis swing analyzer with MediaPipe pose detection."""
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
        
        # Store joint data for both players
        self.pro_joint_data = {joint: [] for joint in self.key_joints.keys()}
        self.amateur_joint_data = {joint: [] for joint in self.key_joints.keys()}
        
        # Store frame numbers
        self.pro_frames = []
        self.amateur_frames = []
        
        # Store angle data
        self.pro_angles = {
            'elbow_angle': [],
            'knee_flexion': [],
            'hip_shoulder_angle': []
        }
        self.amateur_angles = {
            'elbow_angle': [],
            'knee_flexion': [],
            'hip_shoulder_angle': []
        }

    def draw_landmarks_on_image(self, rgb_image, detection_result, color):
        """Draw landmarks on each frame with specified color."""
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

    def process_frame(self, frame, player_type, frame_num, analyze=True):
        """Process a single frame with pose detection and optionally analyze angles."""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect pose landmarks
        detection_result = self.detector.detect(mp_image)
        
        # Check if pose landmarks detected
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            # Get first detected pose
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # Store joint data if analyzing
            if analyze:
                # Store frame number
                if player_type == 'pro':
                    self.pro_frames.append(frame_num)
                else:
                    self.amateur_frames.append(frame_num)
                
                # Store joint positions
                joint_data = self.pro_joint_data if player_type == 'pro' else self.amateur_joint_data
                for joint_name, joint_idx in self.key_joints.items():
                    if joint_idx < len(pose_landmarks):
                        landmark = pose_landmarks[joint_idx]
                        joint_data[joint_name].append(
                            (landmark.x, landmark.y, landmark.z, landmark.visibility)
                        )
                    else:
                        joint_data[joint_name].append(None)
                
                # Calculate and store angles
                self.calculate_angles(pose_landmarks, player_type, frame_num)
            
            # Draw landmarks on image
            color = self.pro_color if player_type == 'pro' else self.amateur_color
            annotated_frame = self.draw_landmarks_on_image(frame_rgb, detection_result, color)
            return cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            # Return original frame if no pose detected
            return frame

    def calculate_angles(self, pose_landmarks, player_type, frame_num):
        """Calculate key angles for tennis forehand analysis."""
        # Select angle data structure based on player type
        angle_data = self.pro_angles if player_type == 'pro' else self.amateur_angles
        
        # Get joint positions
        joints = {}
        for joint_name, joint_idx in self.key_joints.items():
            if joint_idx < len(pose_landmarks):
                landmark = pose_landmarks[joint_idx]
                joints[joint_name] = (landmark.x, landmark.y, landmark.z)
            else:
                joints[joint_name] = None
        
        # Calculate elbow angle (for right-handed player focusing on right arm)
        if all(joints.get(j) is not None for j in ['right_shoulder', 'right_elbow', 'right_wrist']):
            shoulder = np.array(joints['right_shoulder'][:2])  # x, y only
            elbow = np.array(joints['right_elbow'][:2])
            wrist = np.array(joints['right_wrist'][:2])
            
            v1 = shoulder - elbow
            v2 = wrist - elbow
            
            elbow_angle = self.calculate_angle_between_vectors(v1, v2)
            angle_data['elbow_angle'].append((frame_num, elbow_angle))
        
        # Calculate knee flexion (for right leg)
        if all(joints.get(j) is not None for j in ['right_hip', 'right_knee', 'right_ankle']):
            hip = np.array(joints['right_hip'][:2])
            knee = np.array(joints['right_knee'][:2])
            ankle = np.array(joints['right_ankle'][:2])
            
            v1 = hip - knee
            v2 = ankle - knee
            
            knee_angle = self.calculate_angle_between_vectors(v1, v2)
            angle_data['knee_flexion'].append((frame_num, knee_angle))
        
        # Calculate hip to shoulder rotation angle
        if all(joints.get(j) is not None for j in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
            left_hip = np.array(joints['left_hip'][:2])
            right_hip = np.array(joints['right_hip'][:2])
            left_shoulder = np.array(joints['left_shoulder'][:2])
            right_shoulder = np.array(joints['right_shoulder'][:2])
            
            hip_vector = right_hip - left_hip
            shoulder_vector = right_shoulder - left_shoulder
            
            hip_shoulder_angle = self.calculate_angle_between_vectors(hip_vector, shoulder_vector)
            angle_data['hip_shoulder_angle'].append((frame_num, hip_shoulder_angle))

    def calculate_angle_between_vectors(self, v1, v2):
        """Calculate the angle between two 2D vectors in degrees."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return None
        
        cos_angle = dot_product / (norm_v1 * norm_v2)
        # Clip to handle floating point errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def create_comparison_video(self, pro_video_path, amateur_video_path, output_path, output_fps=10, skip_frames=2):
        """
        Create side-by-side comparison with pose detection and analysis.
        
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
        
        # Determine target height
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
        
        # Reset angle data
        self.pro_angles = {k: [] for k in self.pro_angles.keys()}
        self.amateur_angles = {k: [] for k in self.amateur_angles.keys()}
        self.pro_joint_data = {joint: [] for joint in self.key_joints.keys()}
        self.amateur_joint_data = {joint: [] for joint in self.key_joints.keys()}
        self.pro_frames = []
        self.amateur_frames = []
        
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
            
            # Process frames with pose detection and analysis
            pro_processed = self.process_frame(pro_frame, 'pro', frame_count)
            amateur_processed = self.process_frame(amateur_frame, 'amateur', frame_count)
            
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
            if frame_count % 20 == 0:
                print(f"Processed frame {frame_count}")
            
            frame_count += 1
        
        # Release resources
        pro_cap.release()
        amateur_cap.release()
        out.release()
        
        print(f"Side-by-side comparison completed. Output saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")
        
        # Generate analysis visualizations
        self.generate_analysis(output_dir=os.path.dirname(output_path))

    def generate_analysis(self, output_dir):
        """Generate analysis visualizations for the comparison."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Generate elbow angle comparison
        self.plot_angle_comparison(
            self.pro_angles['elbow_angle'], 
            self.amateur_angles['elbow_angle'],
            "Elbow Angle Comparison", 
            "Frame Number", 
            "Angle (degrees)",
            str(output_dir / "elbow_angle_comparison.png")
        )
        
        # 2. Generate knee flexion comparison
        self.plot_angle_comparison(
            self.pro_angles['knee_flexion'], 
            self.amateur_angles['knee_flexion'],
            "Knee Flexion Comparison", 
            "Frame Number", 
            "Angle (degrees)",
            str(output_dir / "knee_flexion_comparison.png")
        )
        
        # 3. Generate hip-shoulder rotation comparison
        self.plot_angle_comparison(
            self.pro_angles['hip_shoulder_angle'], 
            self.amateur_angles['hip_shoulder_angle'],
            "Hip-Shoulder Rotation Comparison", 
            "Frame Number", 
            "Angle (degrees)",
            str(output_dir / "hip_shoulder_angle_comparison.png")
        )
        
        # 4. Generate wrist path analysis
        self.plot_joint_trajectory_comparison(
            'right_wrist',
            "Wrist Path Comparison",
            str(output_dir / "wrist_path_comparison.png")
        )
        
        # Generate a summary of insights
        self.generate_insights_summary(str(output_dir / "analysis_summary.txt"))

    def plot_angle_comparison(self, pro_data, amateur_data, title, xlabel, ylabel, output_path):
        """Plot angle comparison between pro and amateur player."""
        plt.figure(figsize=(12, 6))
        
        # Plot pro data
        if pro_data:
            frames, angles = zip(*pro_data)
            plt.plot(frames, angles, 'g-', label='Professional')
        
        # Plot amateur data
        if amateur_data:
            frames, angles = zip(*amateur_data)
            plt.plot(frames, angles, 'r-', label='Amateur')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        print(f"Saved {title} to {output_path}")

    def plot_joint_trajectory_comparison(self, joint_name, title, output_path):
        """Plot joint trajectory comparison for a specific joint."""
        plt.figure(figsize=(10, 8))
        
        # Pro player joint trajectory
        pro_joint_data = self.pro_joint_data[joint_name]
        if pro_joint_data:
            x_coords = [data[0] for data in pro_joint_data if data is not None]
            y_coords = [data[1] for data in pro_joint_data if data is not None]
            plt.plot(x_coords, y_coords, 'g-', label='Professional')
            plt.scatter(x_coords, y_coords, c='green', s=30, alpha=0.7)
        
        # Amateur player joint trajectory
        amateur_joint_data = self.amateur_joint_data[joint_name]
        if amateur_joint_data:
            x_coords = [data[0] for data in amateur_joint_data if data is not None]
            y_coords = [data[1] for data in amateur_joint_data if data is not None]
            plt.plot(x_coords, y_coords, 'r-', label='Amateur')
            plt.scatter(x_coords, y_coords, c='red', s=30, alpha=0.7)
        
        plt.title(title)
        plt.xlabel("X position (normalized)")
        plt.ylabel("Y position (normalized)")
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        plt.grid(True)
        plt.legend()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        print(f"Saved {title} to {output_path}")

    def generate_insights_summary(self, output_path):
        """Generate a summary of insights from the analysis."""
        insights = []
        
        # Analyze elbow angle
        if self.pro_angles['elbow_angle'] and self.amateur_angles['elbow_angle']:
            pro_max_elbow = max([angle for _, angle in self.pro_angles['elbow_angle']])
            amateur_max_elbow = max([angle for _, angle in self.amateur_angles['elbow_angle']])
            elbow_diff = pro_max_elbow - amateur_max_elbow
            
            if abs(elbow_diff) > 10:
                if elbow_diff > 0:
                    insights.append(f"The professional player achieves greater elbow extension ({pro_max_elbow:.1f}° vs {amateur_max_elbow:.1f}°), which typically allows for more power generation.")
                else:
                    insights.append(f"The amateur player shows greater elbow extension ({amateur_max_elbow:.1f}° vs {pro_max_elbow:.1f}°), which could indicate overextension or improper technique.")
            else:
                insights.append(f"Both players have similar maximum elbow extension (Pro: {pro_max_elbow:.1f}°, Amateur: {amateur_max_elbow:.1f}°).")
        
        # Analyze knee flexion
        if self.pro_angles['knee_flexion'] and self.amateur_angles['knee_flexion']:
            pro_min_knee = min([angle for _, angle in self.pro_angles['knee_flexion']])
            amateur_min_knee = min([angle for _, angle in self.amateur_angles['knee_flexion']])
            knee_diff = pro_min_knee - amateur_min_knee
            
            if abs(knee_diff) > 10:
                if knee_diff < 0:
                    insights.append(f"The professional demonstrates more knee bend (minimum angle: {pro_min_knee:.1f}° vs {amateur_min_knee:.1f}°), providing better stability and power transfer from the ground up.")
                else:
                    insights.append(f"The amateur shows more knee flexion (minimum angle: {amateur_min_knee:.1f}° vs {pro_min_knee:.1f}°), which might affect balance and power generation.")
            else:
                insights.append(f"Both players have similar knee flexion patterns (Pro min: {pro_min_knee:.1f}°, Amateur min: {amateur_min_knee:.1f}°).")
        
        # Analyze hip-shoulder rotation
        if self.pro_angles['hip_shoulder_angle'] and self.amateur_angles['hip_shoulder_angle']:
            pro_max_rotation = max([angle for _, angle in self.pro_angles['hip_shoulder_angle']])
            amateur_max_rotation = max([angle for _, angle in self.amateur_angles['hip_shoulder_angle']])
            rotation_diff = pro_max_rotation - amateur_max_rotation
            
            if abs(rotation_diff) > 10:
                if rotation_diff > 0:
                    insights.append(f"The professional achieves greater hip-shoulder separation ({pro_max_rotation:.1f}° vs {amateur_max_rotation:.1f}°), which is crucial for generating rotational power.")
                else:
                    insights.append(f"The amateur shows more hip-shoulder rotation ({amateur_max_rotation:.1f}° vs {pro_max_rotation:.1f}°), which could indicate different technique or timing.")
            else:
                insights.append(f"Both players show similar hip-shoulder rotation (Pro: {pro_max_rotation:.1f}°, Amateur: {amateur_max_rotation:.1f}°).")
        
        # General insights and recommendations
        insights.append("\nKey elements of a good forehand technique:")
        insights.append("1. Proper weight transfer from back foot to front foot during the swing")
        insights.append("2. Good hip-shoulder separation for generating rotational power")
        insights.append("3. Appropriate elbow extension - not too straight, not too bent")
        insights.append("4. Smooth follow-through with relaxed arm and wrist")
        
        insights.append("\nRecommendations for improvement:")
        insights.append("1. Focus on proper sequencing: legs → hips → torso → shoulder → elbow → wrist → racket")
        insights.append("2. Maintain balanced stance throughout the stroke")
        insights.append("3. Use video analysis regularly to track technical improvements")
        insights.append("4. Practice specific elements separately before combining them")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write("# Tennis Forehand Analysis\n\n")
            f.write("## Key Insights\n\n")
            for insight in insights:
                f.write(f"{insight}\n")
            
            f.write("\n\n*Note: This analysis is based on computer vision and should be reviewed by a tennis coach for comprehensive feedback.*")
        
        print(f"Generated insights summary saved to {output_path}")

def main():
    # Set video paths
    pro_video_path = "jannik-sinner-forehands.mp4"
    amateur_video_path = "amateur-player-forehands.mov"
    
    # Create output directory
    output_dir = Path("tennis_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Set output path
    output_path = str(output_dir / "forehand_comparison.mp4")
    
    # Create analyzer and run comparison
    print("Creating forehand comparison with analysis...")
    analyzer = TennisSwingAnalyzer()
    analyzer.create_comparison_video(pro_video_path, amateur_video_path, output_path)
    print("Analysis complete!")

if __name__ == "__main__":
    main()