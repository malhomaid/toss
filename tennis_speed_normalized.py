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
import argparse

class TennisSwingComparer:
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
        
        # Data storage
        self.pro_data = {
            'frames': [],
            'landmarks': [],
            'angles': {
                'elbow_angle': [],
                'knee_flexion': [],
                'hip_shoulder_angle': []
            },
            'joint_data': {joint: [] for joint in self.key_joints.keys()}
        }
        
        self.amateur_data = {
            'frames': [],
            'landmarks': [],
            'angles': {
                'elbow_angle': [],
                'knee_flexion': [],
                'hip_shoulder_angle': []
            },
            'joint_data': {joint: [] for joint in self.key_joints.keys()}
        }

    def extract_pose_data(self, video_path, player_type, skip_frames=1, speed_factor=1.0):
        """
        Extract pose data from a video file.
        
        Args:
            video_path: Path to the video file
            player_type: 'pro' or 'amateur'
            skip_frames: Process every nth frame to speed up processing
            speed_factor: Adjust playback speed (>1 speeds up, <1 slows down)
        
        Returns:
            List of processed frames
        """
        print(f"Extracting pose data from {player_type} video: {video_path}")
        print(f"Speed factor: {speed_factor}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {total_frames} frames, {fps} fps")
        
        # Select data storage based on player type
        data = self.pro_data if player_type == 'pro' else self.amateur_data
        
        # Reset data
        data['frames'] = []
        data['landmarks'] = []
        data['angles'] = {k: [] for k in data['angles'].keys()}
        data['joint_data'] = {joint: [] for joint in self.key_joints.keys()}
        
        # Process frames
        processed_frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing, but adjust for speed factor
            adjusted_skip = max(1, int(skip_frames / speed_factor))
            if frame_count % adjusted_skip != 0:
                frame_count += 1
                continue
            
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect pose landmarks
            detection_result = self.detector.detect(mp_image)
            
            # Check if pose detected
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                # Get first detected pose
                pose_landmarks = detection_result.pose_landmarks[0]
                
                # Store frame number, adjusted for speed
                data['frames'].append(frame_count)
                
                # Store pose landmarks
                data['landmarks'].append(pose_landmarks)
                
                # Store joint data
                for joint_name, joint_idx in self.key_joints.items():
                    if joint_idx < len(pose_landmarks):
                        landmark = pose_landmarks[joint_idx]
                        data['joint_data'][joint_name].append(
                            (landmark.x, landmark.y, landmark.z, landmark.visibility)
                        )
                    else:
                        data['joint_data'][joint_name].append(None)
                
                # Calculate angles
                self.calculate_angles(pose_landmarks, data['angles'], frame_count)
                
                # Draw landmarks on image
                color = self.pro_color if player_type == 'pro' else self.amateur_color
                annotated_frame = self.draw_landmarks_on_image(frame_rgb, detection_result, color)
                processed_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
                # Add text labels to frame
                frame_with_text = processed_frames[-1].copy()
                cv2.putText(frame_with_text, f"{player_type.capitalize()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame_with_text, f"Frame: {frame_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                processed_frames[-1] = frame_with_text
            
            # Show progress
            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames for {player_type}")
            
            frame_count += 1
        
        # Release resources
        cap.release()
        
        print(f"Completed extraction for {player_type}. Processed {len(processed_frames)} frames with pose detection.")
        return processed_frames

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

    def calculate_angles(self, pose_landmarks, angle_data, frame_num):
        """Calculate key angles for tennis forehand analysis."""
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

    def create_speed_adjusted_comparison(self, pro_frames, amateur_frames, output_path, output_fps=15):
        """
        Create side-by-side comparison with equal-length videos.
        
        This normalizes the videos to have the same number of frames for fair comparison.
        """
        if not pro_frames or not amateur_frames:
            print("Error: No frames to compare")
            return
        
        print(f"Creating speed-adjusted comparison with {len(pro_frames)} pro frames and {len(amateur_frames)} amateur frames")
        
        # Get frame dimensions
        pro_height, pro_width = pro_frames[0].shape[:2]
        amateur_height, amateur_width = amateur_frames[0].shape[:2]
        
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
        
        # Determine how many frames to use
        # We'll normalize time by using linear interpolation between the frames
        num_output_frames = min(len(pro_frames), len(amateur_frames))
        
        # Create indices for sampling frames
        pro_indices = np.linspace(0, len(pro_frames) - 1, num_output_frames, dtype=int)
        amateur_indices = np.linspace(0, len(amateur_frames) - 1, num_output_frames, dtype=int)
        
        print(f"Creating {num_output_frames} synchronized frames")
        
        # Process each frame
        for i in range(num_output_frames):
            # Get frames from each video
            pro_frame = pro_frames[pro_indices[i]]
            amateur_frame = amateur_frames[amateur_indices[i]]
            
            # Resize frames preserving aspect ratio
            pro_frame_resized = cv2.resize(pro_frame, (pro_new_width, target_height))
            amateur_frame_resized = cv2.resize(amateur_frame, (amateur_new_width, target_height))
            
            # Create combined frame
            combined_frame = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
            combined_frame[0:target_height, 0:pro_new_width] = pro_frame_resized
            combined_frame[0:target_height, pro_new_width:combined_width] = amateur_frame_resized
            
            # Add progress indicator
            progress_pct = int((i / num_output_frames) * 100)
            cv2.putText(combined_frame, f"Swing progress: {progress_pct}%", 
                       (combined_width // 2 - 120, target_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame to output
            out.write(combined_frame)
            
            # Show progress
            if i % 10 == 0:
                print(f"Processed synchronized frame {i}/{num_output_frames}")
        
        # Release resources
        out.release()
        print(f"Speed-adjusted comparison saved to {output_path}")

    def normalize_data_to_percentage(self):
        """Normalize all data to swing progression percentage (0-100%)."""
        # Normalize pro data
        if self.pro_data['frames']:
            self.pro_data['normalized_angles'] = {}
            for angle_type, angle_data in self.pro_data['angles'].items():
                if angle_data:
                    frames, angles = zip(*angle_data)
                    # Create normalized indices (0-100%)
                    norm_indices = np.linspace(0, 100, len(frames))
                    self.pro_data['normalized_angles'][angle_type] = list(zip(norm_indices, angles))
        
        # Normalize amateur data
        if self.amateur_data['frames']:
            self.amateur_data['normalized_angles'] = {}
            for angle_type, angle_data in self.amateur_data['angles'].items():
                if angle_data:
                    frames, angles = zip(*angle_data)
                    # Create normalized indices (0-100%)
                    norm_indices = np.linspace(0, 100, len(frames))
                    self.amateur_data['normalized_angles'][angle_type] = list(zip(norm_indices, angles))

    def generate_normalized_analysis(self, output_dir):
        """Generate analysis visualizations with normalized timing."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Normalize data to percentage
        self.normalize_data_to_percentage()
        
        # Generate angle comparison plots
        angle_types = ['elbow_angle', 'knee_flexion', 'hip_shoulder_angle']
        angle_titles = ['Elbow Angle', 'Knee Flexion', 'Hip-Shoulder Rotation']
        
        for angle_type, title in zip(angle_types, angle_titles):
            if 'normalized_angles' in self.pro_data and angle_type in self.pro_data['normalized_angles']:
                pro_data = self.pro_data['normalized_angles'][angle_type]
            else:
                pro_data = []
                
            if 'normalized_angles' in self.amateur_data and angle_type in self.amateur_data['normalized_angles']:
                amateur_data = self.amateur_data['normalized_angles'][angle_type]
            else:
                amateur_data = []
            
            # Plot normalized angle comparison
            self.plot_normalized_comparison(
                pro_data, amateur_data,
                f"{title} Throughout Swing",
                "Swing Progression (%)",
                "Angle (degrees)",
                str(output_dir / f"{angle_type}_normalized.png")
            )
        
        # Generate insights summary
        self.generate_insights_summary(str(output_dir / "normalized_analysis.txt"))

    def plot_normalized_comparison(self, pro_data, amateur_data, title, xlabel, ylabel, output_path):
        """Plot comparison using normalized progression."""
        plt.figure(figsize=(12, 6))
        
        # Plot pro data
        if pro_data:
            norm_indices, values = zip(*pro_data)
            plt.plot(norm_indices, values, 'g-', label='Professional', linewidth=2)
        
        # Plot amateur data
        if amateur_data:
            norm_indices, values = zip(*amateur_data)
            plt.plot(norm_indices, values, 'r-', label='Amateur', linewidth=2)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        print(f"Saved {title} to {output_path}")

    def generate_insights_summary(self, output_path):
        """Generate insights based on normalized data."""
        insights = []
        
        # Check if we have normalized data
        if not hasattr(self.pro_data, 'normalized_angles') or not hasattr(self.amateur_data, 'normalized_angles'):
            insights.append("Insufficient data for normalized analysis.")
            
        else:
            # Analyze elbow angle
            if ('normalized_angles' in self.pro_data and 
                'elbow_angle' in self.pro_data['normalized_angles'] and
                'normalized_angles' in self.amateur_data and
                'elbow_angle' in self.amateur_data['normalized_angles']):
                
                pro_max_elbow = max([angle for _, angle in self.pro_data['normalized_angles']['elbow_angle']])
                amateur_max_elbow = max([angle for _, angle in self.amateur_data['normalized_angles']['elbow_angle']])
                elbow_diff = pro_max_elbow - amateur_max_elbow
                
                if abs(elbow_diff) > 10:
                    if elbow_diff > 0:
                        insights.append(f"The professional player achieves greater elbow extension ({pro_max_elbow:.1f}° vs {amateur_max_elbow:.1f}°), which typically allows for more power generation.")
                    else:
                        insights.append(f"The amateur player shows greater elbow extension ({amateur_max_elbow:.1f}° vs {pro_max_elbow:.1f}°), which could indicate overextension or improper technique.")
                else:
                    insights.append(f"Both players have similar maximum elbow extension (Pro: {pro_max_elbow:.1f}°, Amateur: {amateur_max_elbow:.1f}°).")
            
            # Analyze knee flexion
            if ('normalized_angles' in self.pro_data and 
                'knee_flexion' in self.pro_data['normalized_angles'] and
                'normalized_angles' in self.amateur_data and
                'knee_flexion' in self.amateur_data['normalized_angles']):
                
                pro_min_knee = min([angle for _, angle in self.pro_data['normalized_angles']['knee_flexion']])
                amateur_min_knee = min([angle for _, angle in self.amateur_data['normalized_angles']['knee_flexion']])
                knee_diff = pro_min_knee - amateur_min_knee
                
                if abs(knee_diff) > 10:
                    if knee_diff < 0:
                        insights.append(f"The professional demonstrates more knee bend (minimum angle: {pro_min_knee:.1f}° vs {amateur_min_knee:.1f}°), providing better stability and power transfer from the ground up.")
                    else:
                        insights.append(f"The amateur shows more knee flexion (minimum angle: {amateur_min_knee:.1f}° vs {pro_min_knee:.1f}°), which might affect balance and power generation.")
                else:
                    insights.append(f"Both players have similar knee flexion patterns (Pro min: {pro_min_knee:.1f}°, Amateur min: {amateur_min_knee:.1f}°).")
            
            # Add general insights
            insights.append("\nKey technical observations based on normalized analysis:")
            insights.append("1. Professional players typically maintain a more consistent swing path")
            insights.append("2. The pro's swing has better sequencing: legs → hips → torso → shoulder → elbow → wrist")
            insights.append("3. Professional players typically have more controlled follow-through")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write("# Tennis Forehand Analysis (Time-Normalized)\n\n")
            f.write("## Key Insights\n\n")
            for insight in insights:
                f.write(f"{insight}\n")
            
            f.write("\n\n*Note: This analysis normalizes the timing of both swings to allow for direct comparison regardless of video speed.*")
        
        print(f"Generated insights summary saved to {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tennis forehand analysis with speed normalization')
    parser.add_argument('--pro_video', default="jannik-sinner-forehands.mp4", help='Path to professional player video')
    parser.add_argument('--amateur_video', default="amateur-player-forehands.mov", help='Path to amateur player video')
    parser.add_argument('--pro_speed', type=float, default=1.0, help='Speed factor for pro video (default: 1.0)')
    parser.add_argument('--amateur_speed', type=float, default=1.0, help='Speed factor for amateur video (default: 1.0)')
    parser.add_argument('--output_dir', default="tennis_analysis_output", help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer
    analyzer = TennisSwingComparer()
    
    # Extract pose data from videos with speed factors
    print("Extracting pose data from videos...")
    pro_frames = analyzer.extract_pose_data(
        args.pro_video, 'pro', skip_frames=1, speed_factor=args.pro_speed
    )
    
    amateur_frames = analyzer.extract_pose_data(
        args.amateur_video, 'amateur', skip_frames=1, speed_factor=args.amateur_speed
    )
    
    # Create speed-adjusted comparison
    print("Creating speed-adjusted comparison...")
    analyzer.create_speed_adjusted_comparison(
        pro_frames, amateur_frames,
        str(output_dir / "speed_normalized_comparison.mp4")
    )
    
    # Generate normalized analysis
    print("Generating normalized analysis...")
    analyzer.generate_normalized_analysis(args.output_dir)
    
    print("Analysis complete!")
    print(f"Output files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()