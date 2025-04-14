import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
from pathlib import Path

class TennisStrokeAnalyzer:
    def __init__(self, model_path='pose_landmarker.task'):
        """Initialize the tennis stroke analyzer with MediaPipe pose detection."""
        # Set up the PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Create detector
        self.detector = vision.PoseLandmarker.create_from_options(options)

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

        # Store joint trajectory data for analysis
        self.joint_trajectories = {}

        # Colors for visualization
        self.pro_color = (0, 255, 0)  # Green for pro player
        self.amateur_color = (0, 0, 255)  # Red for amateur player

    def process_video(self, video_path, player_type='amateur', output_path=None, skip_frames=1):
        """Process a video and extract pose data for the player."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return None

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer for output if path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Initialize trajectory storage for this player
        self.joint_trajectories[player_type] = {joint: [] for joint in self.key_joints.keys()}
        self.joint_trajectories[player_type]['frame_numbers'] = []

        # Process video frame by frame
        frame_count = 0
        processed_frames = []
        # Store all frames instead of limiting to improve analysis
        max_frames_to_store = total_frames

        print(f"Processing {player_type} video. Total frames: {total_frames}, using skip factor: {skip_frames}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Skip frames if specified (reduced to ensure we capture the full stroke)
            if frame_count % skip_frames != 0:
                continue

            # Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect pose landmarks
            detection_result = self.detector.detect(mp_image)

            # Draw landmarks and store trajectory data
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                # Store the frame number
                self.joint_trajectories[player_type]['frame_numbers'].append(frame_count)

                # Get first detected pose (assuming single player)
                pose_landmarks = detection_result.pose_landmarks[0]

                # Store key joint positions
                for joint_name, joint_idx in self.key_joints.items():
                    if joint_idx < len(pose_landmarks):
                        landmark = pose_landmarks[joint_idx]
                        self.joint_trajectories[player_type][joint_name].append(
                            (landmark.x, landmark.y, landmark.z, landmark.visibility)
                        )
                    else:
                        # If joint not detected, store None
                        self.joint_trajectories[player_type][joint_name].append(None)

                # Store processed frames for later comparison
                if len(processed_frames) < max_frames_to_store:
                    player_color = self.pro_color if player_type == 'pro' else self.amateur_color
                    annotated_frame = self.draw_landmarks_on_image(frame_rgb, detection_result, player_color)
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                    # Add frame counter and player type
                    cv2.putText(annotated_frame_bgr, f"Frame: {frame_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(annotated_frame_bgr, f"Player: {player_type.capitalize()}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, player_color, 2)

                    processed_frames.append(annotated_frame_bgr)

                # Write to output video if specified
                if out:
                    player_color = self.pro_color if player_type == 'pro' else self.amateur_color
                    annotated_frame = self.draw_landmarks_on_image(frame_rgb, detection_result, player_color)
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                    # Add frame counter and player type
                    cv2.putText(annotated_frame_bgr, f"Frame: {frame_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(annotated_frame_bgr, f"Player: {player_type.capitalize()}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, player_color, 2)

                    out.write(annotated_frame_bgr)

            # Print progress periodically
            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames for {player_type} player")

        # Release resources
        cap.release()
        if out:
            out.release()
            cv2.destroyAllWindows()
            print(f"Analysis for {player_type} complete. Output saved to {output_path}")

        print(f"Completed processing {player_type} video. Stored trajectory data for {len(self.joint_trajectories[player_type]['frame_numbers'])} frames.")

        return processed_frames

    def draw_landmarks_on_image(self, rgb_image, detection_result, color=(0, 255, 0)):
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

    def normalize_trajectories(self):
        """Normalize trajectories to make them comparable (time-alignment)."""
        for player in self.joint_trajectories.keys():
            # Check if we have data for this player
            if not self.joint_trajectories[player]['frame_numbers']:
                continue

            # Get the length of the trajectory
            trajectory_length = len(self.joint_trajectories[player]['frame_numbers'])

            # Normalize to 100 points (percentages of the swing)
            normalized_frames = np.linspace(0, trajectory_length-1, 100, dtype=int)

            # Create normalized trajectories
            normalized_trajectories = {joint: [] for joint in self.key_joints.keys()}

            for joint in self.key_joints.keys():
                joint_data = self.joint_trajectories[player][joint]
                for i in normalized_frames:
                    if i < len(joint_data) and joint_data[i] is not None:
                        normalized_trajectories[joint].append(joint_data[i])
                    else:
                        normalized_trajectories[joint].append(None)

            # Replace the original trajectories with normalized ones
            for joint in self.key_joints.keys():
                self.joint_trajectories[player][joint] = normalized_trajectories[joint]

            # Update frame numbers to percentages
            self.joint_trajectories[player]['frame_numbers'] = list(range(100))

    def calculate_joint_angles(self):
        """Calculate key joint angles for each player throughout the swing."""
        angle_data = {}

        for player in self.joint_trajectories.keys():
            angle_data[player] = {
                'elbow_angle': [],  # Angle at the elbow (arm extension)
                'shoulder_rotation': [],  # Shoulder rotation
                'hip_knee_ankle_angle': [],  # Lower body positioning
                'wrist_path': [],  # Wrist path relative to body
                'knee_flexion': []  # Knee bend
            }

            frames = self.joint_trajectories[player]['frame_numbers']

            for frame_idx in range(len(frames)):
                # Extract joint positions for this frame
                joints = {}
                for joint_name in self.key_joints.keys():
                    if (frame_idx < len(self.joint_trajectories[player][joint_name]) and
                        self.joint_trajectories[player][joint_name][frame_idx] is not None):
                        joints[joint_name] = self.joint_trajectories[player][joint_name][frame_idx][:3]  # x, y, z
                    else:
                        joints[joint_name] = None

                # Calculate elbow angle (for right-handed player focusing on right arm)
                if all(joints.get(j) is not None for j in ['right_shoulder', 'right_elbow', 'right_wrist']):
                    shoulder = np.array(joints['right_shoulder'])
                    elbow = np.array(joints['right_elbow'])
                    wrist = np.array(joints['right_wrist'])

                    v1 = shoulder - elbow
                    v2 = wrist - elbow

                    elbow_angle = self.calculate_angle(v1, v2)
                    angle_data[player]['elbow_angle'].append(elbow_angle)
                else:
                    angle_data[player]['elbow_angle'].append(None)

                # Calculate knee flexion (for right leg)
                if all(joints.get(j) is not None for j in ['right_hip', 'right_knee', 'right_ankle']):
                    hip = np.array(joints['right_hip'])
                    knee = np.array(joints['right_knee'])
                    ankle = np.array(joints['right_ankle'])

                    v1 = hip - knee
                    v2 = ankle - knee

                    knee_angle = self.calculate_angle(v1, v2)
                    angle_data[player]['knee_flexion'].append(knee_angle)
                else:
                    angle_data[player]['knee_flexion'].append(None)

                # Additional angle calculations can be added here

        return angle_data

    def calculate_angle(self, v1, v2):
        """Calculate the angle between two vectors in degrees."""
        dot_product = np.dot(v1[:2], v2[:2])  # Use only x,y for 2D angle
        norm_v1 = np.linalg.norm(v1[:2])
        norm_v2 = np.linalg.norm(v2[:2])

        if norm_v1 == 0 or norm_v2 == 0:
            return None

        cos_angle = dot_product / (norm_v1 * norm_v2)
        # Clip to handle floating point errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def visualize_joint_trajectories(self, joint_name, output_path=None):
        """Visualize the trajectory of a specific joint for both players."""
        plt.figure(figsize=(10, 8))

        for player, color, marker in [('pro', 'green', 'o'), ('amateur', 'red', 'x')]:
            if player not in self.joint_trajectories:
                continue

            joint_data = self.joint_trajectories[player][joint_name]
            frames = self.joint_trajectories[player]['frame_numbers']

            # Extract x, y coordinates
            x_coords = []
            y_coords = []

            for i, data in enumerate(joint_data):
                if data is not None:
                    x_coords.append(data[0])
                    y_coords.append(data[1])

            plt.scatter(x_coords, y_coords, color=color, marker=marker,
                       label=f"{player.capitalize()} {joint_name.replace('_', ' ')}")
            plt.plot(x_coords, y_coords, color=color, alpha=0.5)

        plt.title(f"Trajectory of {joint_name.replace('_', ' ')}")
        plt.xlabel("X position (normalized)")
        plt.ylabel("Y position (normalized)")
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(output_path)
            print(f"Trajectory visualization saved to {output_path}")
        else:
            plt.show()

    def visualize_angle_comparison(self, angle_data, angle_type, output_path=None):
        """Visualize the comparison of a specific angle between pro and amateur."""
        plt.figure(figsize=(12, 6))

        for player, color in [('pro', 'green'), ('amateur', 'red')]:
            if player in angle_data and angle_type in angle_data[player]:
                angles = angle_data[player][angle_type]
                frames = list(range(len(angles)))

                # Filter out None values
                valid_frames = []
                valid_angles = []
                for i, angle in enumerate(angles):
                    if angle is not None:
                        valid_frames.append(i)
                        valid_angles.append(angle)

                plt.plot(valid_frames, valid_angles, color=color,
                         label=f"{player.capitalize()} {angle_type.replace('_', ' ')}")

        plt.title(f"Comparison of {angle_type.replace('_', ' ')}")
        plt.xlabel("Swing progression (%)")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(output_path)
            print(f"Angle comparison visualization saved to {output_path}")
        else:
            plt.show()

    def create_side_by_side_comparison(self, pro_frames, amateur_frames, output_path):
        """Create a side-by-side video comparison of pro and amateur players."""
        # Ensure we have frames from both players
        if not pro_frames or not amateur_frames:
            print("Error: Missing frames for comparison")
            return

        # Determine the number of frames to use (minimum of both videos)
        num_frames = min(len(pro_frames), len(amateur_frames))
        print(f"Creating side-by-side comparison with {num_frames} frames")

        if num_frames == 0:
            print("No frames available for comparison")
            return

        # Get frame dimensions
        pro_height, pro_width = pro_frames[0].shape[:2]
        amateur_height, amateur_width = amateur_frames[0].shape[:2]

        # Preserve aspect ratio while resizing to a common height
        target_height = max(pro_height, amateur_height)

        # Calculate new widths maintaining aspect ratio
        pro_new_width = int(pro_width * (target_height / pro_height))
        amateur_new_width = int(amateur_width * (target_height / amateur_height))

        # Create a combined frame
        combined_width = pro_new_width + amateur_new_width
        combined_height = target_height

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (combined_width, combined_height))

        # Process each frame
        for i in range(num_frames):
            # Create a blank canvas
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            # Resize frames preserving aspect ratio
            pro_frame_resized = cv2.resize(pro_frames[i], (pro_new_width, target_height))
            amateur_frame_resized = cv2.resize(amateur_frames[i], (amateur_new_width, target_height))

            # Add pro frame on the left
            combined_frame[0:target_height, 0:pro_new_width] = pro_frame_resized

            # Add amateur frame on the right
            combined_frame[0:target_height, pro_new_width:combined_width] = amateur_frame_resized

            # Add labels
            cv2.putText(combined_frame, "Professional", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.pro_color, 2)
            cv2.putText(combined_frame, "Amateur", (pro_new_width + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.amateur_color, 2)

            # Add frame counter
            cv2.putText(combined_frame, f"Frame: {i}", (combined_width // 2 - 70, combined_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write to output
            out.write(combined_frame)

            # Progress update
            if i % 10 == 0:
                print(f"Processed {i}/{num_frames} frames for side-by-side comparison")

        # Release resources
        out.release()
        print(f"Side-by-side comparison saved to {output_path}")

    def generate_analysis_report(self, angle_data, output_path):
        """Generate a comprehensive analysis report highlighting key differences."""
        # Create key metrics dataframe
        metrics = {
            'Metric': [
                'Max Elbow Extension',
                'Avg Elbow Angle',
                'Max Knee Flexion',
                'Avg Knee Flexion'
            ],
            'Pro': [None, None, None, None],
            'Amateur': [None, None, None, None],
            'Difference': [None, None, None, None]
        }

        # Calculate metrics with error checking
        angle_types = ['elbow_angle', 'elbow_angle', 'knee_flexion', 'knee_flexion']

        for i, angle_type in enumerate(angle_types):
            for player in ['pro', 'amateur']:
                # Check if player and angle_type exist in the data
                if (player in angle_data and
                    angle_type in angle_data[player] and
                    angle_data[player][angle_type]):

                    # Filter out None values
                    angles = [a for a in angle_data[player][angle_type] if a is not None]

                    if angles:
                        if i % 2 == 0:  # Max metrics (even indices)
                            metrics[player][i] = max(angles)
                        else:  # Avg metrics (odd indices)
                            metrics[player][i] = sum(angles) / len(angles)

        # Calculate differences
        for i in range(len(metrics['Metric'])):
            if metrics['Pro'][i] is not None and metrics['Amateur'][i] is not None:
                metrics['Difference'][i] = metrics['Pro'][i] - metrics['Amateur'][i]

        # Create a DataFrame
        df = pd.DataFrame(metrics)

        # Round numeric values
        for col in ['Pro', 'Amateur', 'Difference']:
            df[col] = df[col].apply(lambda x: round(x, 2) if x is not None else None)

        # Generate insights
        insights = []

        # Add insight about data quality if needed
        if all(metrics['Pro'][i] is None for i in range(len(metrics['Metric']))):
            insights.append("Note: Insufficient data for professional player analysis. Please ensure the professional player video is processed correctly.")

        if all(metrics['Amateur'][i] is None for i in range(len(metrics['Metric']))):
            insights.append("Note: Insufficient data for amateur player analysis. Please ensure the amateur player video is processed correctly.")

        # Elbow extension insight
        if metrics['Pro'][0] is not None and metrics['Amateur'][0] is not None:
            diff = metrics['Difference'][0]
            if abs(diff) > 10:
                if diff > 0:
                    insights.append("The professional player achieves greater elbow extension during the forehand, "
                                   "which typically allows for more power generation.")
                else:
                    insights.append("The amateur player shows greater elbow extension than the professional. "
                                   "This could indicate overextension or timing issues.")

        # Knee flexion insight
        if metrics['Pro'][2] is not None and metrics['Amateur'][2] is not None:
            diff = metrics['Difference'][2]
            if abs(diff) > 10:
                if diff > 0:
                    insights.append("The professional demonstrates more knee bend during the stroke, "
                                   "which provides better stability and power transfer from the ground up.")
                else:
                    insights.append("The amateur shows more knee flexion than the professional. "
                                   "While knee bend is important, excessive flexion might indicate balance issues.")

        # General insights
        insights.append("Professional players typically maintain better posture throughout the swing, "
                       "with the head remaining stable and the non-dominant arm providing balance.")
        insights.append("The kinetic chain in professional forehands starts from the ground up: "
                       "legs → hips → torso → shoulder → elbow → wrist → racket. "
                       "Any break in this chain reduces power and consistency.")

        # Create recommendations based on insights
        recommendations = [
            "Focus on proper weight transfer from back foot to front foot during the swing",
            "Practice maintaining a stable head position throughout the stroke",
            "Work on timing the rotation of hips, shoulders, and arm extension",
            "Develop core strength to improve the connection between lower and upper body",
            "Film your stroke regularly to track improvements and identify issues"
        ]

        # Save report to file
        with open(output_path, 'w') as f:
            f.write("# Tennis Forehand Analysis Report\n\n")

            f.write("## Key Metrics Comparison\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

            f.write("## Key Insights\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
            f.write("\n\n")

            f.write("## Recommendations for Improvement\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n\n")

            f.write("*Note: This analysis is based on computer vision and should be reviewed by a tennis coach for comprehensive feedback.*")

        print(f"Analysis report saved to {output_path}")

    def visualize_all_joints(self, output_dir):
        """Visualize trajectories for all tracked joints."""
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)

        # Generate visualizations for all joints
        for joint_name in self.key_joints.keys():
            output_path = str(Path(output_dir) / f"{joint_name}_trajectory.png")
            self.visualize_joint_trajectories(joint_name, output_path)
            print(f"Generated visualization for {joint_name}")

    def visualize_all_angles(self, angle_data, output_dir):
        """Visualize all calculated angle data."""
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)

        # Get all angle types from the data
        if 'pro' in angle_data:
            angle_types = angle_data['pro'].keys()
        elif 'amateur' in angle_data:
            angle_types = angle_data['amateur'].keys()
        else:
            print("No angle data available to visualize")
            return

        # Generate visualizations for all angle types
        for angle_type in angle_types:
            output_path = str(Path(output_dir) / f"{angle_type}_comparison.png")
            self.visualize_angle_comparison(angle_data, angle_type, output_path)
            print(f"Generated visualization for {angle_type}")

# Example usage
def main():
    # Initialize the analyzer
    analyzer = TennisStrokeAnalyzer(model_path='pose_landmarker.task')

    # Create output directory
    output_dir = Path("tennis_analysis_output")
    output_dir.mkdir(exist_ok=True)

    # Set video paths
    pro_video_path = "jannik-sinner-forehands.mp4"
    amateur_video_path = "amateur-player-forehands.mov"

    print("Starting analysis pipeline - Fixed version")

    # Process videos with no frame skipping to capture full strokes
    print("\nStep 1: Processing professional player video...")
    pro_frames = analyzer.process_video(
        pro_video_path,
        player_type='pro',
        output_path=str(output_dir / "pro_analysis.mp4"),
        skip_frames=1  # Process all frames
    )

    print("\nStep 2: Processing amateur player video...")
    amateur_frames = analyzer.process_video(
        amateur_video_path,
        player_type='amateur',
        output_path=str(output_dir / "amateur_analysis.mp4"),
        skip_frames=1  # Process all frames
    )

    print("\nStep 3: Normalizing trajectory data...")
    # Normalize trajectories for fair comparison
    analyzer.normalize_trajectories()

    print("\nStep 4: Calculating joint angles...")
    # Calculate joint angles
    angle_data = analyzer.calculate_joint_angles()

    print("\nStep 5: Creating visualizations for all joints...")
    # Visualize all joint trajectories
    analyzer.visualize_all_joints(str(output_dir))

    print("\nStep 6: Creating visualizations for all angles...")
    # Visualize all angle comparisons
    analyzer.visualize_all_angles(angle_data, str(output_dir))

    # Create side-by-side comparison if frames are available
    if pro_frames and amateur_frames:
        print("\nStep 7: Creating side-by-side comparison video...")
        analyzer.create_side_by_side_comparison(
            pro_frames,
            amateur_frames,
            str(output_dir / "side_by_side_comparison.mp4")
        )
    else:
        print("\nSkipping side-by-side comparison due to insufficient frames")

    print("\nStep 8: Generating analysis report...")
    # Generate analysis report with error handling
    analyzer.generate_analysis_report(
        angle_data,
        str(output_dir / "analysis_report.md")
    )

    print("\nAnalysis complete! Results saved to:", output_dir)
    print("Key output files:")
    print("  - analysis_report.md: Detailed analysis and recommendations")
    print("  - Multiple joint trajectory visualizations")
    print("  - Multiple angle comparison visualizations")
    print("  - side_by_side_comparison.mp4: Visual comparison of both players")

if __name__ == "__main__":
    main()
