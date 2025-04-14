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
import pandas as pd
import argparse
from datetime import datetime

class TennisMetricsEnhanced:
    def __init__(self, model_path='pose_landmarker.task'):
        """Initialize the enhanced tennis metrics analyzer with MediaPipe pose detection."""
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
        
        # Load metrics definitions
        self.metrics_df = self.load_metrics_definitions('tennis-metrics.csv')
        
        # Data storage for each player
        self.pro_data = self.initialize_data_structure()
        self.amateur_data = self.initialize_data_structure()
        
        # Store video properties
        self.video_info = {
            'pro': {'width': 0, 'height': 0, 'fps': 0, 'frames': 0},
            'amateur': {'width': 0, 'height': 0, 'fps': 0, 'frames': 0}
        }

    def load_metrics_definitions(self, metrics_csv_path):
        """Load metrics definitions from CSV file."""
        try:
            return pd.read_csv(metrics_csv_path)
        except Exception as e:
            print(f"Error loading metrics definitions: {e}")
            return pd.DataFrame(columns=['Category', 'Metric', 'How to Calculate (AI method)', 'Optimal Range', 'Issue if Outside Optimal Range'])

    def initialize_data_structure(self):
        """Initialize data structure for storing player data."""
        return {
            'frames': [],
            'landmarks': [],
            'joint_data': {joint: [] for joint in self.key_joints.keys()},
            'swing_phases': {
                'preparation_start': None,
                'backswing_peak': None,
                'forward_swing_start': None,
                'impact': None,
                'follow_through_peak': None,
                'recovery_complete': None
            },
            'metrics': {
                'shoulder_rotation_timing': None,
                'shoulder_rotation_angle': None,
                'center_of_mass_stability': None,
                'knee_bend': None,
                'vertical_wrist_displacement': None,
                'follow_through_length': None,
                'contact_position_horizontal': None,
                'contact_height': None,
                'elbow_angle_at_contact': None,
                'wrist_stability_at_contact': None,
                'hip_rotation': None,
                'post_contact_rotation': None,
                'recovery_time': None
            },
            'metric_values_over_time': {
                'shoulder_rotation_angle': [],
                'knee_bend': [],
                'elbow_angle': [],
                'hip_rotation': [],
                'wrist_position': []
            },
            'normalized_metrics': {}  # Will store metrics normalized by swing phase %
        }

    def extract_pose_data(self, video_path, player_type, skip_frames=1):
        """
        Extract pose data from a video file.
        
        Args:
            video_path: Path to the video file
            player_type: 'pro' or 'amateur'
            skip_frames: Process every nth frame to speed up processing
        
        Returns:
            List of processed frames
        """
        print(f"Extracting pose data from {player_type} video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {total_frames} frames, {fps} fps, {width}x{height} resolution")
        
        # Store video info
        self.video_info[player_type] = {
            'width': width,
            'height': height,
            'fps': fps,
            'frames': total_frames
        }
        
        # Select data storage based on player type
        data = self.pro_data if player_type == 'pro' else self.amateur_data
        
        # Reset data
        data['frames'] = []
        data['landmarks'] = []
        data['joint_data'] = {joint: [] for joint in self.key_joints.keys()}
        for key in data['metric_values_over_time'].keys():
            data['metric_values_over_time'][key] = []
        
        # Process frames
        processed_frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % skip_frames != 0:
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
                
                # Store frame number
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
                
                # Calculate metrics for each frame
                self.calculate_frame_metrics(pose_landmarks, data, frame_count)
                
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
        
        # Identify swing phases
        self.identify_swing_phases(data, player_type)
        
        # Calculate swing metrics
        self.calculate_swing_metrics(data, player_type)
        
        # Normalize metrics by swing phase
        self.normalize_metrics_by_phase(data, player_type)
        
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

    def calculate_frame_metrics(self, pose_landmarks, data, frame_num):
        """Calculate metrics for each frame."""
        # Get joint positions
        joints = {}
        for joint_name, joint_idx in self.key_joints.items():
            if joint_idx < len(pose_landmarks):
                landmark = pose_landmarks[joint_idx]
                joints[joint_name] = (landmark.x, landmark.y, landmark.z)
            else:
                joints[joint_name] = None
        
        # Calculate shoulder rotation angle (between shoulders and vertical)
        if all(joints.get(j) is not None for j in ['left_shoulder', 'right_shoulder']):
            left_shoulder = np.array(joints['left_shoulder'][:2])
            right_shoulder = np.array(joints['right_shoulder'][:2])
            shoulder_vector = right_shoulder - left_shoulder
            reference_vector = np.array([0, 1])  # Vertical axis
            
            shoulder_angle = self.calculate_angle_between_vectors(shoulder_vector, reference_vector)
            if shoulder_vector[0] < 0:  # Adjust for direction
                shoulder_angle = 180 - shoulder_angle
            data['metric_values_over_time']['shoulder_rotation_angle'].append((frame_num, shoulder_angle))
        
        # Calculate knee bend angle (right knee for right-handed player)
        if all(joints.get(j) is not None for j in ['right_hip', 'right_knee', 'right_ankle']):
            hip = np.array(joints['right_hip'][:2])
            knee = np.array(joints['right_knee'][:2])
            ankle = np.array(joints['right_ankle'][:2])
            
            v1 = hip - knee
            v2 = ankle - knee
            
            knee_angle = self.calculate_angle_between_vectors(v1, v2)
            data['metric_values_over_time']['knee_bend'].append((frame_num, knee_angle))
        
        # Calculate elbow angle (right elbow for right-handed player)
        if all(joints.get(j) is not None for j in ['right_shoulder', 'right_elbow', 'right_wrist']):
            shoulder = np.array(joints['right_shoulder'][:2])
            elbow = np.array(joints['right_elbow'][:2])
            wrist = np.array(joints['right_wrist'][:2])
            
            v1 = shoulder - elbow
            v2 = wrist - elbow
            
            elbow_angle = self.calculate_angle_between_vectors(v1, v2)
            data['metric_values_over_time']['elbow_angle'].append((frame_num, elbow_angle))
        
        # Calculate hip rotation angle
        if all(joints.get(j) is not None for j in ['left_hip', 'right_hip']):
            left_hip = np.array(joints['left_hip'][:2])
            right_hip = np.array(joints['right_hip'][:2])
            hip_vector = right_hip - left_hip
            reference_vector = np.array([0, 1])  # Vertical axis
            
            hip_angle = self.calculate_angle_between_vectors(hip_vector, reference_vector)
            if hip_vector[0] < 0:  # Adjust for direction
                hip_angle = 180 - hip_angle
            data['metric_values_over_time']['hip_rotation'].append((frame_num, hip_angle))
        
        # Store wrist position for tracking
        if joints.get('right_wrist') is not None:
            data['metric_values_over_time']['wrist_position'].append(
                (frame_num, joints['right_wrist'][0], joints['right_wrist'][1])
            )

    def identify_swing_phases(self, data, player_type):
        """
        Identify key phases of the tennis swing.
        
        Phases:
        1. Preparation start
        2. Backswing peak
        3. Forward swing start
        4. Impact
        5. Follow-through peak
        6. Recovery complete
        """
        print(f"Identifying swing phases for {player_type}...")
        
        # Check if we have enough data
        if not data['metric_values_over_time']['wrist_position']:
            print(f"Insufficient data to identify swing phases for {player_type}")
            return
        
        # Extract wrist position data
        wrist_data = data['metric_values_over_time']['wrist_position']
        frames = [item[0] for item in wrist_data]
        x_positions = [item[1] for item in wrist_data]
        y_positions = [item[2] for item in wrist_data]
        
        # Calculate wrist velocity (change in position)
        x_velocity = [0] + [x_positions[i] - x_positions[i-1] for i in range(1, len(x_positions))]
        y_velocity = [0] + [y_positions[i] - y_positions[i-1] for i in range(1, len(y_positions))]
        
        # 2. Backswing peak: Point where wrist is furthest back (minimum x for right-handed player)
        backswing_peak_idx = np.argmin(x_positions)
        data['swing_phases']['backswing_peak'] = frames[backswing_peak_idx]
        
        # 4. Impact: Approximate as the point where wrist velocity changes from increasing to decreasing x
        # Find where x_velocity changes from positive to negative after backswing
        impact_candidates = []
        for i in range(backswing_peak_idx + 1, len(x_velocity) - 1):
            if x_velocity[i] > 0 and x_velocity[i+1] < 0:
                impact_candidates.append(i)
        
        if impact_candidates:
            impact_idx = impact_candidates[0]  # Take the first instance after backswing
            data['swing_phases']['impact'] = frames[impact_idx]
            
            # 3. Forward swing start: Midway between backswing peak and impact
            forward_start_idx = (backswing_peak_idx + impact_idx) // 2
            data['swing_phases']['forward_swing_start'] = frames[forward_start_idx]
            
            # 5. Follow-through peak: Maximum extension after impact
            follow_through_candidates = list(range(impact_idx + 1, len(x_positions)))
            if follow_through_candidates:
                # Maximum extension typically involves both horizontal and vertical components
                extension = [np.sqrt((x_positions[i] - x_positions[impact_idx])**2 + 
                                    (y_positions[i] - y_positions[impact_idx])**2) 
                            for i in follow_through_candidates]
                max_extension_idx = follow_through_candidates[np.argmax(extension)]
                data['swing_phases']['follow_through_peak'] = frames[max_extension_idx]
                
                # 1. Preparation start: Estimate as 20 frames before backswing peak
                prep_start_idx = max(0, backswing_peak_idx - 20)
                data['swing_phases']['preparation_start'] = frames[prep_start_idx]
                
                # 6. Recovery complete: Estimate as 20 frames after follow-through peak
                recovery_idx = min(len(frames) - 1, max_extension_idx + 20)
                data['swing_phases']['recovery_complete'] = frames[recovery_idx]
        
        # Print identified phases
        print(f"Swing phases for {player_type}:")
        for phase, frame in data['swing_phases'].items():
            if frame is not None:
                print(f"  {phase}: frame {frame}")
            else:
                print(f"  {phase}: Not identified")

    def normalize_metrics_by_phase(self, data, player_type):
        """Normalize metrics data by swing phase percentage (0-100%)."""
        # Check if all essential phases are identified
        essential_phases = ['preparation_start', 'backswing_peak', 'impact', 'follow_through_peak']
        if not all(data['swing_phases'].get(phase) is not None for phase in essential_phases):
            print(f"Cannot normalize metrics: missing essential swing phases for {player_type}")
            return
        
        # Get phase frames
        prep_start = data['swing_phases']['preparation_start']
        backswing = data['swing_phases']['backswing_peak']
        impact = data['swing_phases']['impact']
        follow_through = data['swing_phases']['follow_through_peak']
        
        # Define phase percentages (0-100%)
        # 0-20%: Preparation to backswing
        # 20-60%: Backswing to impact
        # 60-100%: Impact to follow-through
        phase_mapping = {
            prep_start: 0,
            backswing: 20, 
            impact: 60,
            follow_through: 100
        }
        
        # Initialize normalized metrics
        normalized_metrics = {
            'shoulder_rotation_angle': [],
            'knee_bend': [],
            'elbow_angle': [],
            'hip_rotation': [],
            'wrist_position': []
        }
        
        # Normalize each metric
        for metric_name, metric_data in data['metric_values_over_time'].items():
            if not metric_data:
                continue
                
            for item in metric_data:
                frame = item[0]
                # Only process frames within the swing
                if prep_start <= frame <= follow_through:
                    # Find which phase segment the frame is in
                    if frame <= backswing:
                        # Preparation to backswing (0-20%)
                        phase_pct = 0 + (frame - prep_start) / (backswing - prep_start) * 20
                    elif frame <= impact:
                        # Backswing to impact (20-60%)
                        phase_pct = 20 + (frame - backswing) / (impact - backswing) * 40
                    else:
                        # Impact to follow-through (60-100%)
                        phase_pct = 60 + (frame - impact) / (follow_through - impact) * 40
                    
                    # Store normalized data
                    if metric_name == 'wrist_position':
                        normalized_metrics[metric_name].append((phase_pct, item[1], item[2]))
                    else:
                        normalized_metrics[metric_name].append((phase_pct, item[1]))
        
        # Sort all normalized metrics by phase percentage
        for metric_name in normalized_metrics:
            normalized_metrics[metric_name].sort(key=lambda x: x[0])
        
        # Store normalized metrics in data
        data['normalized_metrics'] = normalized_metrics
        print(f"Metrics normalized by swing phase for {player_type}")

    def calculate_swing_metrics(self, data, player_type):
        """Calculate comprehensive swing metrics based on identified phases."""
        print(f"Calculating swing metrics for {player_type}...")
        
        # Check if phases have been identified
        if data['swing_phases']['impact'] is None:
            print(f"Cannot calculate metrics without identified impact point for {player_type}")
            return
        
        metrics = data['metrics']
        
        # 1. Shoulder Rotation Timing
        if data['swing_phases']['preparation_start'] is not None:
            # Find maximum shoulder rotation during preparation
            prep_start = data['swing_phases']['preparation_start']
            impact = data['swing_phases']['impact']
            
            shoulder_data = [(frame, angle) for frame, angle in 
                            data['metric_values_over_time']['shoulder_rotation_angle']
                            if prep_start <= frame <= impact]
            
            if shoulder_data:
                # Calculate time from preparation to maximum shoulder rotation
                max_rotation_idx = np.argmax([angle for _, angle in shoulder_data])
                max_rotation_frame = shoulder_data[max_rotation_idx][0]
                metrics['shoulder_rotation_timing'] = (max_rotation_frame - prep_start) / self.video_info[player_type]['fps']
                print(f"Shoulder rotation timing: {metrics['shoulder_rotation_timing']:.2f} seconds")
        
        # 2. Shoulder Rotation Angle
        if data['swing_phases']['backswing_peak'] is not None:
            # Find shoulder angle at backswing peak
            backswing_peak = data['swing_phases']['backswing_peak']
            
            shoulder_angles = [angle for frame, angle in 
                              data['metric_values_over_time']['shoulder_rotation_angle']
                              if frame == backswing_peak]
            
            if shoulder_angles:
                metrics['shoulder_rotation_angle'] = shoulder_angles[0]
                print(f"Shoulder rotation angle: {metrics['shoulder_rotation_angle']:.2f} degrees")
        
        # 3. Center of Mass Stability
        if data['swing_phases']['preparation_start'] is not None and data['swing_phases']['follow_through_peak'] is not None:
            # Track hip position stability throughout swing
            prep_start = data['swing_phases']['preparation_start']
            follow_through = data['swing_phases']['follow_through_peak']
            
            # Calculate COM proxy (average of hip positions)
            com_positions = []
            for frame_idx, frame in enumerate(data['frames']):
                if prep_start <= frame <= follow_through and frame_idx < len(data['landmarks']):
                    left_hip = data['joint_data']['left_hip'][frame_idx]
                    right_hip = data['joint_data']['right_hip'][frame_idx]
                    
                    if left_hip and right_hip:
                        # Calculate average position of hips
                        com_x = (left_hip[0] + right_hip[0]) / 2
                        com_y = (left_hip[1] + right_hip[1]) / 2
                        com_positions.append((com_x, com_y))
            
            if com_positions:
                # Calculate maximum displacement
                base_com = com_positions[0]
                max_displacement = max([np.sqrt((pos[0] - base_com[0])**2 + (pos[1] - base_com[1])**2) 
                                      for pos in com_positions])
                
                # Convert to real-world units (approximate)
                # Using screen width as reference (typical shoulder width ~45cm)
                screen_width = self.video_info[player_type]['width']
                metrics['center_of_mass_stability'] = max_displacement * 45 / 0.2  # Assuming shoulders span ~20% of screen width
                print(f"Center of mass stability: {metrics['center_of_mass_stability']:.2f} cm displacement")
        
        # 4. Knee Bend
        if data['swing_phases']['impact'] is not None:
            # Get knee angle at impact
            impact = data['swing_phases']['impact']
            
            knee_angles = [angle for frame, angle in 
                          data['metric_values_over_time']['knee_bend']
                          if frame == impact]
            
            if knee_angles:
                metrics['knee_bend'] = knee_angles[0]
                print(f"Knee bend angle: {metrics['knee_bend']:.2f} degrees")
        
        # 5. Vertical Wrist Displacement
        if (data['swing_phases']['backswing_peak'] is not None and 
            data['swing_phases']['impact'] is not None):
            # Calculate vertical distance traveled by wrist from backswing to impact
            backswing = data['swing_phases']['backswing_peak']
            impact = data['swing_phases']['impact']
            
            wrist_positions = data['metric_values_over_time']['wrist_position']
            backswing_pos = next((y for frame, _, y in wrist_positions if frame == backswing), None)
            impact_pos = next((y for frame, _, y in wrist_positions if frame == impact), None)
            
            if backswing_pos and impact_pos:
                # Calculate vertical displacement in normalized units
                vert_displacement = abs(impact_pos - backswing_pos)
                
                # Convert to real-world units (approximate)
                screen_height = self.video_info[player_type]['height']
                metrics['vertical_wrist_displacement'] = vert_displacement * 170  # Assuming typical height ~170cm
                print(f"Vertical wrist displacement: {metrics['vertical_wrist_displacement']:.2f} cm")
        
        # 6. Follow-through Length
        if data['swing_phases']['impact'] is not None and data['swing_phases']['follow_through_peak'] is not None:
            # Calculate distance from impact to follow-through peak
            impact = data['swing_phases']['impact']
            follow_through = data['swing_phases']['follow_through_peak']
            
            wrist_positions = data['metric_values_over_time']['wrist_position']
            impact_pos = next(((x, y) for frame, x, y in wrist_positions if frame == impact), None)
            follow_pos = next(((x, y) for frame, x, y in wrist_positions if frame == follow_through), None)
            
            if impact_pos and follow_pos:
                # Calculate distance in normalized units
                follow_length = np.sqrt((follow_pos[0] - impact_pos[0])**2 + (follow_pos[1] - impact_pos[1])**2)
                
                # Convert to real-world units (approximate)
                screen_width = self.video_info[player_type]['width']
                metrics['follow_through_length'] = follow_length * 170  # Assuming typical height ~170cm
                print(f"Follow-through length: {metrics['follow_through_length']:.2f} cm")
        
        # 7/8. Contact Position (horizontal/height) - Need ball tracking, approximate with wrist position
        # This is an approximation assuming the ball is near the wrist at contact
        if data['swing_phases']['impact'] is not None:
            impact = data['swing_phases']['impact']
            
            # Find wrist and hip positions at impact
            wrist_pos = next(((x, y) for frame, x, y in data['metric_values_over_time']['wrist_position'] 
                             if frame == impact), None)
            
            if wrist_pos and impact < len(data['landmarks']):
                landmarks = data['landmarks'][data['frames'].index(impact)]
                
                # Horizontal: relation to front hip
                right_hip_idx = self.key_joints['right_hip']
                if right_hip_idx < len(landmarks):
                    hip_x = landmarks[right_hip_idx].x
                    metrics['contact_position_horizontal'] = wrist_pos[0] - hip_x
                    print(f"Contact position horizontal: {metrics['contact_position_horizontal']:.3f} (normalized)")
                
                # Height: relation to hip height
                right_hip_idx = self.key_joints['right_hip']
                if right_hip_idx < len(landmarks):
                    hip_y = landmarks[right_hip_idx].y
                    metrics['contact_height'] = hip_y - wrist_pos[1]  # Positive is above hip
                    print(f"Contact height: {metrics['contact_height']:.3f} (normalized)")
        
        # 9. Elbow Angle at Contact
        if data['swing_phases']['impact'] is not None:
            impact = data['swing_phases']['impact']
            
            elbow_angles = [angle for frame, angle in 
                           data['metric_values_over_time']['elbow_angle']
                           if frame == impact]
            
            if elbow_angles:
                metrics['elbow_angle_at_contact'] = elbow_angles[0]
                print(f"Elbow angle at contact: {metrics['elbow_angle_at_contact']:.2f} degrees")
        
        # 10. Wrist Stability at Contact (approximation - need racket detection)
        # For now, we'll look at wrist angle change right before and after impact
        if data['swing_phases']['impact'] is not None:
            impact = data['swing_phases']['impact']
            impact_idx = data['frames'].index(impact)
            
            if (0 < impact_idx < len(data['landmarks']) - 1 and
                impact_idx - 1 < len(data['landmarks']) and
                impact_idx + 1 < len(data['landmarks'])):
                
                # Get wrist, elbow, and hand/finger positions
                landmarks_before = data['landmarks'][impact_idx - 1]
                landmarks_at = data['landmarks'][impact_idx]
                landmarks_after = data['landmarks'][impact_idx + 1]
                
                # Approximate wrist stability by looking at wrist angle change
                elbow_idx = self.key_joints['right_elbow']
                wrist_idx = self.key_joints['right_wrist']
                
                if (elbow_idx < len(landmarks_before) and wrist_idx < len(landmarks_before) and
                    elbow_idx < len(landmarks_at) and wrist_idx < len(landmarks_at) and
                    elbow_idx < len(landmarks_after) and wrist_idx < len(landmarks_after)):
                    
                    # We'll use the index finger as a proxy for racket direction (imperfect)
                    hand_idx = 20  # Index fingertip for racket direction approximation
                    
                    if hand_idx < len(landmarks_before) and hand_idx < len(landmarks_at) and hand_idx < len(landmarks_after):
                        # Calculate wrist angles before, at, and after impact
                        angles = []
                        for lm in [landmarks_before, landmarks_at, landmarks_after]:
                            elbow = np.array([lm[elbow_idx].x, lm[elbow_idx].y])
                            wrist = np.array([lm[wrist_idx].x, lm[wrist_idx].y])
                            hand = np.array([lm[hand_idx].x, lm[hand_idx].y])
                            
                            forearm_vec = wrist - elbow
                            hand_vec = hand - wrist
                            
                            angle = self.calculate_angle_between_vectors(forearm_vec, hand_vec)
                            angles.append(angle)
                        
                        # Calculate stability as the inverse of angle change
                        angle_change = max(abs(angles[1] - angles[0]), abs(angles[2] - angles[1]))
                        metrics['wrist_stability_at_contact'] = 180 - angle_change  # Higher is more stable
                        print(f"Wrist stability at contact: {metrics['wrist_stability_at_contact']:.2f} degrees")
        
        # 11. Hip Rotation
        if data['swing_phases']['impact'] is not None:
            impact = data['swing_phases']['impact']
            
            hip_angles = [angle for frame, angle in 
                         data['metric_values_over_time']['hip_rotation']
                         if frame == impact]
            
            if hip_angles:
                metrics['hip_rotation'] = hip_angles[0]
                print(f"Hip rotation at contact: {metrics['hip_rotation']:.2f} degrees")
        
        # 12. Post-Contact Shoulder & Hip Rotation
        if data['swing_phases']['impact'] is not None and data['swing_phases']['follow_through_peak'] is not None:
            impact = data['swing_phases']['impact']
            follow_through = data['swing_phases']['follow_through_peak']
            
            # Shoulder rotation after impact
            shoulder_at_impact = next((angle for frame, angle in 
                                     data['metric_values_over_time']['shoulder_rotation_angle']
                                     if frame == impact), None)
            
            shoulder_max_after = max([angle for frame, angle in 
                                     data['metric_values_over_time']['shoulder_rotation_angle']
                                     if impact < frame <= follow_through] or [0])
            
            if shoulder_at_impact is not None:
                metrics['post_contact_rotation'] = shoulder_max_after - shoulder_at_impact
                print(f"Post-contact rotation: {metrics['post_contact_rotation']:.2f} degrees")
        
        # 13. Return to Neutral Stance Time
        if data['swing_phases']['follow_through_peak'] is not None and data['swing_phases']['recovery_complete'] is not None:
            follow_through = data['swing_phases']['follow_through_peak']
            recovery = data['swing_phases']['recovery_complete']
            
            # Calculate time between follow-through and recovery
            recovery_time_frames = recovery - follow_through
            metrics['recovery_time'] = recovery_time_frames / self.video_info[player_type]['fps']
            print(f"Recovery time: {metrics['recovery_time']:.2f} seconds")

    def generate_enhanced_report(self, pro_data, amateur_data, output_path):
        """Generate an enhanced comprehensive report with practical insights."""
        # Create report header
        report_lines = []
        report_lines.append("# Tennis Forehand Metrics Analysis")
        report_lines.append(f"\n*Generated on {datetime.now().strftime('%Y-%m-%d')}*")
        
        # Executive summary
        report_lines.append("\n## Executive Summary")
        report_lines.append("\nThis analysis compares professional and amateur tennis forehand technique using biomechanical analysis. Key metrics have been extracted from video footage and analyzed to identify technical differences and opportunities for improvement.")
        
        # Key findings section
        report_lines.append("\n### Key Findings")
        
        # Analyze metrics for significant differences
        key_differences = []
        optimal_deviations = []
        
        # Determine which metrics have valid data for both players
        valid_metrics = {}
        for key in pro_data['metrics']:
            pro_value = pro_data['metrics'][key]
            amateur_value = amateur_data['metrics'][key]
            if pro_value is not None and amateur_value is not None:
                valid_metrics[key] = {
                    'name': key.replace('_', ' ').title(),
                    'pro': pro_value,
                    'amateur': amateur_value,
                    'diff': pro_value - amateur_value,
                    'diff_pct': abs((pro_value - amateur_value) / (pro_value if pro_value != 0 else 1)) * 100
                }
        
        # Sort by percentage difference
        sorted_metrics = sorted(valid_metrics.values(), key=lambda x: x['diff_pct'], reverse=True)
        
        # Get top differences
        for metric in sorted_metrics[:3]:
            if metric['diff_pct'] > 20:  # Only include significant differences (>20%)
                if metric['diff'] > 0:
                    key_differences.append(f"- **{metric['name']}**: The professional's value ({metric['pro']:.1f}) is {metric['diff_pct']:.0f}% higher than the amateur's ({metric['amateur']:.1f})")
                else:
                    key_differences.append(f"- **{metric['name']}**: The amateur's value ({metric['amateur']:.1f}) is {metric['diff_pct']:.0f}% higher than the professional's ({metric['pro']:.1f})")
        
        # Check optimal ranges for key metrics
        metrics_to_check = ['knee_bend', 'shoulder_rotation_angle', 'elbow_angle_at_contact', 'hip_rotation']
        optimal_ranges = {
            'knee_bend': (120, 140),
            'shoulder_rotation_angle': (70, 90),
            'elbow_angle_at_contact': (130, 150),
            'hip_rotation': (70, 90)
        }
        
        for key in metrics_to_check:
            if key in valid_metrics:
                metric = valid_metrics[key]
                if key in optimal_ranges:
                    min_val, max_val = optimal_ranges[key]
                    pro_in_range = min_val <= metric['pro'] <= max_val
                    amateur_in_range = min_val <= metric['amateur'] <= max_val
                    
                    if pro_in_range and not amateur_in_range:
                        if metric['amateur'] < min_val:
                            optimal_deviations.append(f"- **{metric['name']}**: The amateur's value ({metric['amateur']:.1f}) is below the optimal range ({min_val}-{max_val})")
                        else:
                            optimal_deviations.append(f"- **{metric['name']}**: The amateur's value ({metric['amateur']:.1f}) exceeds the optimal range ({min_val}-{max_val})")
        
        # Add findings to report
        if key_differences:
            report_lines.append("\n#### Major Technical Differences:")
            for item in key_differences:
                report_lines.append(item)
        
        if optimal_deviations:
            report_lines.append("\n#### Deviations from Optimal Range:")
            for item in optimal_deviations:
                report_lines.append(item)
        
        # Swing phase analysis
        report_lines.append("\n### Swing Phase Analysis")
        
        # Compare timing of key phases
        if (pro_data['swing_phases']['preparation_start'] is not None and 
            pro_data['swing_phases']['impact'] is not None and
            amateur_data['swing_phases']['preparation_start'] is not None and
            amateur_data['swing_phases']['impact'] is not None):
            
            # Calculate total swing time
            pro_prep_to_impact = (pro_data['swing_phases']['impact'] - pro_data['swing_phases']['preparation_start']) / self.video_info['pro']['fps']
            amateur_prep_to_impact = (amateur_data['swing_phases']['impact'] - amateur_data['swing_phases']['preparation_start']) / self.video_info['amateur']['fps']
            
            report_lines.append(f"\n- **Total Preparation to Impact Time**: Professional: {pro_prep_to_impact:.2f}s, Amateur: {amateur_prep_to_impact:.2f}s")
            
            # Compare backswing to impact timing
            if (pro_data['swing_phases']['backswing_peak'] is not None and
                amateur_data['swing_phases']['backswing_peak'] is not None):
                pro_backswing_to_impact = (pro_data['swing_phases']['impact'] - pro_data['swing_phases']['backswing_peak']) / self.video_info['pro']['fps']
                amateur_backswing_to_impact = (amateur_data['swing_phases']['impact'] - amateur_data['swing_phases']['backswing_peak']) / self.video_info['amateur']['fps']
                
                report_lines.append(f"- **Backswing to Impact Time**: Professional: {pro_backswing_to_impact:.2f}s, Amateur: {amateur_backswing_to_impact:.2f}s")
                
                # Add insight on timing
                if amateur_backswing_to_impact < pro_backswing_to_impact * 0.7:
                    report_lines.append("  - The amateur's forward swing is significantly faster than the professional's, which may indicate rushing through the swing.")
                elif amateur_backswing_to_impact > pro_backswing_to_impact * 1.3:
                    report_lines.append("  - The amateur's forward swing is significantly slower than the professional's, which may reduce power generation.")
        
        # Detailed metrics analysis
        report_lines.append("\n## Detailed Metrics Analysis")
        
        # Group metrics by category
        categories = self.metrics_df['Category'].unique()
        
        for category in categories:
            report_lines.append(f"\n### {category}")
            
            # Filter metrics for this category
            category_metrics = self.metrics_df[self.metrics_df['Category'] == category]
            
            for _, row in category_metrics.iterrows():
                metric_name = row['Metric']
                calculation = row['How to Calculate (AI method)']
                optimal_range = row['Optimal Range']
                issue = row['Issue if Outside Optimal Range']
                
                # Convert metric name to dictionary key format
                key = metric_name.lower().replace(' ', '_')
                
                # Get pro and amateur values
                pro_value = pro_data['metrics'].get(key)
                amateur_value = amateur_data['metrics'].get(key)
                
                # Only include metrics where we have data
                if pro_value is not None or amateur_value is not None:
                    report_lines.append(f"\n#### {metric_name}")
                    report_lines.append(f"*{calculation}*")
                    report_lines.append(f"\n**Optimal Range**: {optimal_range}")
                    
                    # Add units based on metric type
                    unit = "°" if "angle" in key or "rotation" in key else " sec" if "time" in key else " cm"
                    
                    if pro_value is not None:
                        report_lines.append(f"**Professional**: {pro_value:.2f}{unit}")
                    else:
                        report_lines.append(f"**Professional**: Not measured")
                        
                    if amateur_value is not None:
                        report_lines.append(f"**Amateur**: {amateur_value:.2f}{unit}")
                    else:
                        report_lines.append(f"**Amateur**: Not measured")
                    
                    # Add analysis if we have both values
                    if pro_value is not None and amateur_value is not None:
                        report_lines.append("\n**Analysis:**")
                        
                        # Calculate difference and percentage
                        diff = pro_value - amateur_value
                        diff_pct = abs(diff / (pro_value if pro_value != 0 else 1)) * 100
                        
                        # Parse optimal range if possible
                        optimal_min, optimal_max = None, None
                        if '–' in optimal_range:
                            try:
                                parts = optimal_range.replace('°', '').replace('cm', '').split('–')
                                optimal_min = float(parts[0])
                                optimal_max = float(parts[1])
                            except:
                                pass
                        
                        # Compare to optimal range
                        if optimal_min is not None and optimal_max is not None:
                            pro_in_range = optimal_min <= pro_value <= optimal_max
                            amateur_in_range = optimal_min <= amateur_value <= optimal_max
                            
                            if pro_in_range and not amateur_in_range:
                                if amateur_value < optimal_min:
                                    report_lines.append(f"The professional's value is within the optimal range, while the amateur's value is too low. {issue}")
                                    report_lines.append(f"**Recommendation**: Increase {metric_name.lower()} by approximately {(optimal_min - amateur_value):.1f}{unit}.")
                                else:
                                    report_lines.append(f"The professional's value is within the optimal range, while the amateur's value is too high. {issue}")
                                    report_lines.append(f"**Recommendation**: Decrease {metric_name.lower()} by approximately {(amateur_value - optimal_max):.1f}{unit}.")
                            elif not pro_in_range and amateur_in_range:
                                report_lines.append(f"Interestingly, the amateur's value is within the optimal range, while the professional's is not. This may reflect an individual style or technique adaptation.")
                            elif pro_in_range and amateur_in_range:
                                report_lines.append(f"Both players have values within the optimal range, indicating good technique for this aspect.")
                            else:
                                report_lines.append(f"Neither player has values within the optimal range. {issue}")
                        
                        # Compare values directly
                        if diff_pct > 20:  # Significant difference threshold
                            if diff > 0:
                                report_lines.append(f"The professional player's value is {diff_pct:.0f}% higher than the amateur's.")
                            else:
                                report_lines.append(f"The amateur player's value is {diff_pct:.0f}% higher than the professional's.")
                        else:
                            report_lines.append(f"Both players show similar values for this metric (difference: {diff_pct:.0f}%).")
        
        # Add practical recommendations section
        report_lines.append("\n## Practical Recommendations")
        report_lines.append("\nBased on the analysis, here are specific drills and exercises to improve your forehand technique:")
        
        # Generate specific recommendations based on metrics
        recommendations = []
        
        # Check knee bend
        if 'knee_bend' in valid_metrics and valid_metrics['knee_bend']['amateur'] > 160:
            recommendations.append("\n### 1. Improve Lower Body Stability and Knee Bend")
            recommendations.append("- **Issue**: Your knee angle is too straight (measured at " + f"{valid_metrics['knee_bend']['amateur']:.1f}°)")
            recommendations.append("- **Target**: Aim for knee angle between 120°-140° at impact")
            recommendations.append("- **Exercise**: Practice drop step and split step with exaggerated knee bend")
            recommendations.append("- **Drill**: Shadow swings focusing on holding a deeper knee bend throughout the stroke")
            recommendations.append("- **Cue**: \"Sit in a chair\" feeling during preparation and impact")
        
        # Check shoulder rotation
        if 'shoulder_rotation_angle' in valid_metrics:
            if abs(valid_metrics['shoulder_rotation_angle']['diff']) > 15:
                if valid_metrics['shoulder_rotation_angle']['diff'] > 0:
                    recommendations.append("\n### 2. Improve Shoulder Rotation")
                    recommendations.append("- **Issue**: Insufficient shoulder rotation during backswing")
                    recommendations.append("- **Target**: Increase shoulder rotation to 80°-90° during preparation")
                    recommendations.append("- **Exercise**: Medicine ball rotation throws with exaggerated wind-up")
                    recommendations.append("- **Drill**: Shadow swings with pause at maximum shoulder turn")
                    recommendations.append("- **Cue**: \"Show your back to the net\" during the preparation phase")
        
        # Check wrist path
        if 'vertical_wrist_displacement' in valid_metrics and valid_metrics['vertical_wrist_displacement']['amateur'] < 15:
            recommendations.append("\n### 3. Develop Proper Swing Path")
            recommendations.append("- **Issue**: Insufficient vertical wrist movement (too flat)")
            recommendations.append("- **Target**: Increase vertical wrist displacement to 35-50cm")
            recommendations.append("- **Exercise**: Shadow swings with exaggerated low-to-high path")
            recommendations.append("- **Drill**: Ball drops with focus on brushing up the back of the ball")
            recommendations.append("- **Cue**: \"Swing from pocket to shoulder\" to create proper swing arc")
        
        # Check follow-through
        if 'follow_through_length' in valid_metrics and valid_metrics['follow_through_length']['diff'] < -10:
            recommendations.append("\n### 4. Complete Follow-Through")
            recommendations.append("- **Issue**: While your follow-through distance is good, ensure it wraps around your body")
            recommendations.append("- **Target**: Maintain follow-through with proper shoulder rotation")
            recommendations.append("- **Exercise**: Shadow swings with full follow-through and freeze finish")
            recommendations.append("- **Drill**: Hit with focus on follow-through position (racket ending over opposite shoulder)")
            recommendations.append("- **Cue**: \"Finish with racket by your left ear\" (for right-handed player)")
        
        # Check elbow angle
        if 'elbow_angle_at_contact' in valid_metrics:
            if valid_metrics['elbow_angle_at_contact']['amateur'] < 120:
                recommendations.append("\n### 5. Improve Arm Extension")
                recommendations.append("- **Issue**: Elbow too bent at contact")
                recommendations.append("- **Target**: Increase elbow angle to 130°-150° at impact")
                recommendations.append("- **Exercise**: Wall drills focusing on contact point with extended arm")
                recommendations.append("- **Drill**: Shadow swings focusing on extension through the hitting zone")
                recommendations.append("- **Cue**: \"Extend through the ball\" at contact")
        
        # If no specific recommendations, add general ones
        if not recommendations:
            recommendations.append("\n### General Technique Improvements")
            recommendations.append("1. **Kinetic Chain Development**: Practice sequencing from ground up (legs → hips → torso → shoulder → elbow → wrist)")
            recommendations.append("2. **Balance Drills**: Practice forehands while standing on one leg to improve stability")
            recommendations.append("3. **Video Analysis**: Continue recording and comparing your technique with professional players")
        
        # Add all recommendations to report
        for rec in recommendations:
            report_lines.append(rec)
        
        # Practice plan section
        report_lines.append("\n## Weekly Practice Plan")
        report_lines.append("\nIntegrate these focus areas into your training with this progressive practice plan:")
        
        report_lines.append("\n### Week 1: Technical Foundation")
        report_lines.append("- Monday: 20 minutes shadow swings focusing on preparation position and knee bend")
        report_lines.append("- Wednesday: 30 minutes drop-feed practice focusing on swing path")
        report_lines.append("- Friday: 20 minutes video review and shadow swing corrections")
        
        report_lines.append("\n### Week 2: Integration")
        report_lines.append("- Monday: 30 minutes cooperative rally focusing on one technical element at a time")
        report_lines.append("- Wednesday: 30 minutes directional control with new technique")
        report_lines.append("- Friday: 20 minutes high-repetition drill with coach feedback")
        
        report_lines.append("\n### Week 3: Performance")
        report_lines.append("- Monday: 30 minutes live ball drilling with defensive to offensive scenarios")
        report_lines.append("- Wednesday: 30 minutes point play focusing on implementation")
        report_lines.append("- Friday: 30 minutes video analysis and comparison to baseline")
        
        # Final notes
        report_lines.append("\n## Final Notes")
        report_lines.append("\nThe analysis provides valuable insights, but remember that technique is individual. Focus on progressive implementation of changes rather than trying to change everything at once.")
        report_lines.append("\nConsult with your coach when implementing these recommendations to ensure they integrate well with your overall playing style and physical capabilities.")
        
        # Footer
        report_lines.append("\n---")
        report_lines.append("\n*Analysis generated by TennisMetricsEnhanced AI Analyzer*")
        
        # Write report to file
        with open(output_path, 'w') as f:
            for line in report_lines:
                f.write(line + '\n')
        
        print(f"Enhanced metrics report generated and saved to {output_path}")

    def generate_phase_normalized_visuals(self, pro_data, amateur_data, output_dir):
        """Create normalized visualizations comparing metrics across swing phases."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Check if we have normalized data
        if not pro_data.get('normalized_metrics') or not amateur_data.get('normalized_metrics'):
            print("Cannot generate phase-normalized visuals: missing normalized data")
            return
        
        # Metrics to visualize
        metrics_to_plot = [
            ('shoulder_rotation_angle', 'Shoulder Rotation Angle'),
            ('knee_bend', 'Knee Angle'),
            ('elbow_angle', 'Elbow Angle'),
            ('hip_rotation', 'Hip Rotation Angle')
        ]
        
        # Plot each metric normalized by swing phase
        for metric_key, metric_title in metrics_to_plot:
            if (metric_key in pro_data['normalized_metrics'] and 
                metric_key in amateur_data['normalized_metrics']):
                
                plt.figure(figsize=(12, 8))
                
                # Get data
                pro_data_points = pro_data['normalized_metrics'][metric_key]
                amateur_data_points = amateur_data['normalized_metrics'][metric_key]
                
                if pro_data_points and amateur_data_points:
                    # Extract data points
                    pro_x = [p[0] for p in pro_data_points]
                    pro_y = [p[1] for p in pro_data_points]
                    amateur_x = [p[0] for p in amateur_data_points]
                    amateur_y = [p[1] for p in amateur_data_points]
                    
                    # Plot data
                    plt.plot(pro_x, pro_y, 'g-', label='Professional', linewidth=2)
                    plt.plot(amateur_x, amateur_y, 'r-', label='Amateur', linewidth=2)
                    
                    # Add phase markers
                    plt.axvline(x=20, color='black', linestyle='--', alpha=0.5)
                    plt.axvline(x=60, color='black', linestyle='--', alpha=0.5)
                    
                    # Add phase labels
                    plt.text(10, plt.ylim()[0] + 0.1 * (plt.ylim()[1] - plt.ylim()[0]), 
                             "Preparation", ha='center')
                    plt.text(40, plt.ylim()[0] + 0.1 * (plt.ylim()[1] - plt.ylim()[0]), 
                             "Forward Swing", ha='center')
                    plt.text(80, plt.ylim()[0] + 0.1 * (plt.ylim()[1] - plt.ylim()[0]), 
                             "Follow Through", ha='center')
                    
                    # Set title and labels
                    plt.title(f"{metric_title} Throughout Swing (Normalized by Phase)")
                    plt.xlabel("Swing Progression (%)")
                    plt.ylabel("Angle (degrees)" if "angle" in metric_key or "rotation" in metric_key else "Value")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save figure
                    output_file = output_dir / f"{metric_key}_phase_normalized.png"
                    plt.savefig(str(output_file))
                    plt.close()
                    print(f"Saved phase-normalized {metric_title} to {output_file}")
        
        # Create wrist path visualization
        if ('wrist_position' in pro_data['normalized_metrics'] and 
            'wrist_position' in amateur_data['normalized_metrics']):
            
            plt.figure(figsize=(10, 8))
            
            # Get data
            pro_data_points = pro_data['normalized_metrics']['wrist_position']
            amateur_data_points = amateur_data['normalized_metrics']['wrist_position']
            
            if pro_data_points and amateur_data_points:
                # Extract coordinates
                pro_x = [p[1] for p in pro_data_points]
                pro_y = [p[2] for p in pro_data_points]
                amateur_x = [p[1] for p in amateur_data_points]
                amateur_y = [p[2] for p in amateur_data_points]
                
                # Plot paths
                plt.plot(pro_x, pro_y, 'g-', label='Professional', linewidth=2)
                plt.scatter(pro_x, pro_y, c='green', s=30, alpha=0.7)
                
                plt.plot(amateur_x, amateur_y, 'r-', label='Amateur', linewidth=2)
                plt.scatter(amateur_x, amateur_y, c='red', s=30, alpha=0.7)
                
                # Mark key points (20% = backswing peak, 60% = impact)
                pro_backswing_idx = next((i for i, p in enumerate(pro_data_points) if p[0] >= 20), 0)
                pro_impact_idx = next((i for i, p in enumerate(pro_data_points) if p[0] >= 60), 0)
                
                amateur_backswing_idx = next((i for i, p in enumerate(amateur_data_points) if p[0] >= 20), 0)
                amateur_impact_idx = next((i for i, p in enumerate(amateur_data_points) if p[0] >= 60), 0)
                
                # Add markers for backswing and impact
                if pro_backswing_idx < len(pro_x):
                    plt.scatter(pro_x[pro_backswing_idx], pro_y[pro_backswing_idx], 
                                c='green', s=100, marker='*')
                    plt.annotate('Backswing', 
                                xy=(pro_x[pro_backswing_idx], pro_y[pro_backswing_idx]),
                                color='g')
                
                if pro_impact_idx < len(pro_x):
                    plt.scatter(pro_x[pro_impact_idx], pro_y[pro_impact_idx], 
                                c='green', s=100, marker='o')
                    plt.annotate('Impact', 
                                xy=(pro_x[pro_impact_idx], pro_y[pro_impact_idx]),
                                color='g')
                
                if amateur_backswing_idx < len(amateur_x):
                    plt.scatter(amateur_x[amateur_backswing_idx], amateur_y[amateur_backswing_idx], 
                                c='red', s=100, marker='*')
                    plt.annotate('Backswing', 
                                xy=(amateur_x[amateur_backswing_idx], amateur_y[amateur_backswing_idx]),
                                color='r')
                
                if amateur_impact_idx < len(amateur_x):
                    plt.scatter(amateur_x[amateur_impact_idx], amateur_y[amateur_impact_idx], 
                                c='red', s=100, marker='o')
                    plt.annotate('Impact', 
                                xy=(amateur_x[amateur_impact_idx], amateur_y[amateur_impact_idx]),
                                color='r')
                
                # Set title and labels
                plt.title("Wrist Path Comparison (Normalized by Swing Phase)")
                plt.xlabel("X position (normalized)")
                plt.ylabel("Y position (normalized)")
                plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
                plt.grid(True)
                plt.legend()
                
                # Save figure
                output_file = output_dir / "wrist_path_phase_normalized.png"
                plt.savefig(str(output_file))
                plt.close()
                print(f"Saved phase-normalized wrist path to {output_file}")

    def create_enhanced_comparison_video(self, pro_frames, amateur_frames, output_path, output_fps=15):
        """Create an enhanced comparison video with metrics and phase information."""
        if not pro_frames or not amateur_frames:
            print("Error: No frames to compare")
            return
        
        print(f"Creating enhanced comparison video...")
        
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
        num_output_frames = min(len(pro_frames), len(amateur_frames))
        
        # Create indices for sampling frames
        pro_indices = np.linspace(0, len(pro_frames) - 1, num_output_frames, dtype=int)
        amateur_indices = np.linspace(0, len(amateur_frames) - 1, num_output_frames, dtype=int)
        
        print(f"Creating {num_output_frames} synchronized frames")
        
        # Get phase information
        pro_phases = self.pro_data['swing_phases']
        amateur_phases = self.amateur_data['swing_phases']
        
        # Process each frame
        for i in range(num_output_frames):
            # Get frames from each video
            pro_idx = pro_indices[i]
            amateur_idx = amateur_indices[i]
            
            pro_frame = pro_frames[pro_idx]
            amateur_frame = amateur_frames[amateur_idx]
            
            # Current frame numbers
            pro_frame_num = self.pro_data['frames'][pro_idx] if pro_idx < len(self.pro_data['frames']) else 0
            amateur_frame_num = self.amateur_data['frames'][amateur_idx] if amateur_idx < len(self.amateur_data['frames']) else 0
            
            # Calculate swing progress percentage for each player
            pro_progress = self.calculate_swing_progress(pro_frame_num, pro_phases)
            amateur_progress = self.calculate_swing_progress(amateur_frame_num, amateur_phases)
            
            # Add metrics and phase info to frames
            pro_frame_with_metrics = self.add_enhanced_metrics_to_frame(
                pro_frame, self.pro_data, pro_frame_num, pro_progress, self.pro_color
            )
            amateur_frame_with_metrics = self.add_enhanced_metrics_to_frame(
                amateur_frame, self.amateur_data, amateur_frame_num, amateur_progress, self.amateur_color
            )
            
            # Resize frames preserving aspect ratio
            pro_frame_resized = cv2.resize(pro_frame_with_metrics, (pro_new_width, target_height))
            amateur_frame_resized = cv2.resize(amateur_frame_with_metrics, (amateur_new_width, target_height))
            
            # Create combined frame
            combined_frame = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
            combined_frame[0:target_height, 0:pro_new_width] = pro_frame_resized
            combined_frame[0:target_height, pro_new_width:combined_width] = amateur_frame_resized
            
            # Add synchronized progress bar
            avg_progress = (pro_progress + amateur_progress) / 2
            progress_width = int(combined_width * (avg_progress / 100))
            cv2.rectangle(combined_frame, (0, target_height - 10), (progress_width, target_height), (0, 255, 255), -1)
            
            # Add frame counter and title
            cv2.putText(combined_frame, "Tennis Forehand Analysis", 
                       (combined_width // 2 - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add comparison titles for specific metrics
            if 'elbow_angle_at_contact' in self.pro_data['metrics'] and 'elbow_angle_at_contact' in self.amateur_data['metrics']:
                pro_elbow = self.pro_data['metrics']['elbow_angle_at_contact']
                amateur_elbow = self.amateur_data['metrics']['elbow_angle_at_contact']
                if pro_elbow is not None and amateur_elbow is not None:
                    cv2.putText(combined_frame, f"Elbow at Impact: Pro {pro_elbow:.1f}° vs Amateur {amateur_elbow:.1f}°", 
                               (10, target_height - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output
            out.write(combined_frame)
            
            # Show progress
            if i % 20 == 0:
                print(f"Processed frame {i}/{num_output_frames}")
        
        # Release resources
        out.release()
        print(f"Enhanced comparison video saved to {output_path}")

    def calculate_swing_progress(self, frame_num, phases):
        """Calculate swing progress as a percentage (0-100%)."""
        if phases['preparation_start'] is None or phases['follow_through_peak'] is None:
            return 0
        
        prep_start = phases['preparation_start']
        backswing = phases['backswing_peak']
        impact = phases['impact']
        follow_through = phases['follow_through_peak']
        
        # Define phase boundaries
        if frame_num < prep_start:
            return 0
        elif frame_num >= follow_through:
            return 100
        elif frame_num <= backswing:
            # Preparation to backswing (0-20%)
            return (frame_num - prep_start) / (backswing - prep_start) * 20
        elif frame_num <= impact:
            # Backswing to impact (20-60%)
            return 20 + (frame_num - backswing) / (impact - backswing) * 40
        else:
            # Impact to follow-through (60-100%)
            return 60 + (frame_num - impact) / (follow_through - impact) * 40

    def add_enhanced_metrics_to_frame(self, frame, data, frame_num, progress_pct, color):
        """Add enhanced metrics and phase information to a frame."""
        frame_with_metrics = frame.copy()
        height, width = frame.shape[:2]
        
        # Get current metric values for this frame
        metrics = {}
        
        for metric_key in data['metric_values_over_time']:
            for frame_data in data['metric_values_over_time'][metric_key]:
                if frame_data[0] == frame_num:
                    metrics[metric_key] = frame_data[1] if len(frame_data) > 1 else None
                    break
        
        # Add progress bar at top
        progress_width = int(width * (progress_pct / 100))
        cv2.rectangle(frame_with_metrics, (0, 5), (progress_width, 15), color, -1)
        
        # Add swing phase label
        if progress_pct < 20:
            phase_text = "Preparation"
        elif progress_pct < 60:
            phase_text = "Forward Swing"
        else:
            phase_text = "Follow Through"
        
        cv2.putText(frame_with_metrics, f"Phase: {phase_text}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show progress percentage
        cv2.putText(frame_with_metrics, f"{progress_pct:.0f}%", 
                   (width - 70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add metrics table
        text_y = 80
        text_spacing = 30
        
        # Show important metrics
        metric_display = {
            'shoulder_rotation_angle': 'Shoulder Rot:',
            'knee_bend': 'Knee Angle:',
            'elbow_angle': 'Elbow Angle:',
            'hip_rotation': 'Hip Rotation:'
        }
        
        # Draw semi-transparent background for metrics
        metrics_bg_height = len(metric_display) * text_spacing + 10
        metrics_bg_width = 200
        overlay = frame_with_metrics.copy()
        cv2.rectangle(overlay, (5, text_y - 25), (metrics_bg_width, text_y + metrics_bg_height), 
                     (0, 0, 0), -1)
        # Apply transparency
        cv2.addWeighted(overlay, 0.6, frame_with_metrics, 0.4, 0, frame_with_metrics)
        
        for metric_key, display_name in metric_display.items():
            if metric_key in metrics and metrics[metric_key] is not None:
                cv2.putText(frame_with_metrics, f"{display_name} {metrics[metric_key]:.1f}°", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                text_y += text_spacing
        
        # Highlight optimal ranges for current phase
        if 'knee_bend' in metrics and progress_pct >= 50 and progress_pct <= 70:
            knee_value = metrics['knee_bend']
            if knee_value < 120:
                cv2.putText(frame_with_metrics, "KNEE TOO BENT", 
                           (width//2 - 70, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif knee_value > 150:
                cv2.putText(frame_with_metrics, "KNEES TOO STRAIGHT", 
                           (width//2 - 90, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame_with_metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced tennis forehand metrics analysis')
    parser.add_argument('--pro_video', default="jannik-sinner-forehands.mp4", help='Path to professional player video')
    parser.add_argument('--amateur_video', default="amateur-player-forehands.mov", help='Path to amateur player video')
    parser.add_argument('--output_dir', default="tennis_analysis_output", help='Output directory')
    parser.add_argument('--skip_frames', type=int, default=2, help='Process every nth frame (higher values = faster)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer
    analyzer = TennisMetricsEnhanced()
    
    # Extract pose data from videos
    print("Extracting pose data from videos...")
    pro_frames = analyzer.extract_pose_data(args.pro_video, 'pro', skip_frames=args.skip_frames)
    amateur_frames = analyzer.extract_pose_data(args.amateur_video, 'amateur', skip_frames=args.skip_frames)
    
    # Generate enhanced report
    print("Generating enhanced metrics report...")
    analyzer.generate_enhanced_report(
        analyzer.pro_data, 
        analyzer.amateur_data,
        str(output_dir / "enhanced_metrics_report.md")
    )
    
    # Create phase-normalized visualizations
    print("Creating phase-normalized visualizations...")
    analyzer.generate_phase_normalized_visuals(
        analyzer.pro_data,
        analyzer.amateur_data,
        output_dir
    )
    
    # Create enhanced comparison video
    print("Creating enhanced comparison video...")
    analyzer.create_enhanced_comparison_video(
        pro_frames, 
        amateur_frames,
        str(output_dir / "enhanced_metrics_comparison.mp4")
    )
    
    print("Enhanced analysis complete!")
    print(f"Output files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()