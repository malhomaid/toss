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

class TennisMetricsAnalyzer:
    def __init__(self, model_path='pose_landmarker.task'):
        """Initialize the tennis metrics analyzer with MediaPipe pose detection."""
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
            }
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

        # Find direction changes and extremes in the wrist path

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

    def generate_metrics_report(self, pro_data, amateur_data, output_path):
        """Generate comprehensive metrics report comparing pro and amateur players."""
        report_lines = []
        report_lines.append("# Tennis Forehand Metrics Comparison")
        report_lines.append("\n## Key Metrics Analysis\n")

        # Organize metrics by category
        categories = self.metrics_df['Category'].unique()

        for category in categories:
            report_lines.append(f"### {category}\n")

            # Filter metrics for this category
            category_metrics = self.metrics_df[self.metrics_df['Category'] == category]

            for _, row in category_metrics.iterrows():
                metric_name = row['Metric']
                optimal_range = row['Optimal Range']
                issue = row['Issue if Outside Optimal Range']

                # Convert metric name to dictionary key format
                key = metric_name.lower().replace(' ', '_')

                # Get pro and amateur values
                pro_value = pro_data['metrics'].get(key)
                amateur_value = amateur_data['metrics'].get(key)

                report_lines.append(f"#### {metric_name}")
                report_lines.append(f"* Optimal Range: {optimal_range}")

                if pro_value is not None:
                    report_lines.append(f"* Professional: {pro_value:.2f}")
                else:
                    report_lines.append(f"* Professional: Not measured")

                if amateur_value is not None:
                    report_lines.append(f"* Amateur: {amateur_value:.2f}")
                else:
                    report_lines.append(f"* Amateur: Not measured")

                # Generate insight if both values are available
                if pro_value is not None and amateur_value is not None:
                    report_lines.append("\n**Analysis:**")

                    # Parse optimal range if possible
                    optimal_min, optimal_max = None, None
                    if '–' in optimal_range:
                        try:
                            parts = optimal_range.replace('°', '').replace('cm', '').split('–')
                            optimal_min = float(parts[0])
                            optimal_max = float(parts[1])
                        except:
                            pass

                    # Compare to optimal range if available
                    if optimal_min is not None and optimal_max is not None:
                        pro_in_range = optimal_min <= pro_value <= optimal_max
                        amateur_in_range = optimal_min <= amateur_value <= optimal_max

                        if pro_in_range and not amateur_in_range:
                            report_lines.append(f"The professional player's value ({pro_value:.2f}) is within the optimal range, "
                                              f"while the amateur's ({amateur_value:.2f}) is not. {issue}")
                        elif not pro_in_range and amateur_in_range:
                            report_lines.append(f"Interestingly, the amateur player's value ({amateur_value:.2f}) is within the optimal range, "
                                              f"while the professional's ({pro_value:.2f}) is not.")
                        elif pro_in_range and amateur_in_range:
                            report_lines.append(f"Both players have values within the optimal range.")
                        else:
                            report_lines.append(f"Neither player has values within the optimal range. {issue}")

                    # Compare pro vs amateur directly
                    diff = pro_value - amateur_value
                    threshold = (optimal_max - optimal_min) * 0.2 if (optimal_min is not None and optimal_max is not None) else 0.1
                    if abs(diff) > threshold:
                        if diff > 0:
                            report_lines.append(f"The professional player shows a significantly higher value than the amateur.")
                        else:
                            report_lines.append(f"The amateur player shows a higher value than the professional.")
                    else:
                        report_lines.append(f"Both players show similar values for this metric.")

                report_lines.append("\n")

        # Overall recommendations
        report_lines.append("## Overall Recommendations for Improvement\n")

        # Generate specific recommendations based on metrics
        recommendations = []

        # Check key metrics and generate specific recommendations
        if (amateur_data['metrics'].get('shoulder_rotation_angle') is not None and
            pro_data['metrics'].get('shoulder_rotation_angle') is not None):
            if amateur_data['metrics']['shoulder_rotation_angle'] < pro_data['metrics']['shoulder_rotation_angle'] - 10:
                recommendations.append("1. Increase shoulder rotation during preparation. Try exaggerating the turning of your shoulders away from the target during backswing.")

        if (amateur_data['metrics'].get('knee_bend') is not None and
            pro_data['metrics'].get('knee_bend') is not None):
            if abs(amateur_data['metrics']['knee_bend'] - 135) > 15:  # Optimal around 120-140 degrees
                if amateur_data['metrics']['knee_bend'] < 120:
                    recommendations.append("2. Reduce excessive knee bend. Focus on maintaining a balanced, athletic stance with moderate knee flexion.")
                else:
                    recommendations.append("2. Increase knee bend for better stability and power generation. Practicing drop-steps and split-steps can improve your stance.")

        if (amateur_data['metrics'].get('hip_rotation') is not None and
            pro_data['metrics'].get('hip_rotation') is not None):
            if amateur_data['metrics']['hip_rotation'] < pro_data['metrics']['hip_rotation'] - 15:
                recommendations.append("3. Improve hip rotation during the swing for better power generation. Practice exercises that promote hip-shoulder separation.")

        if (amateur_data['metrics'].get('follow_through_length') is not None and
            pro_data['metrics'].get('follow_through_length') is not None):
            if amateur_data['metrics']['follow_through_length'] < pro_data['metrics']['follow_through_length'] * 0.8:
                recommendations.append("4. Extend your follow-through to complete the swing properly. Allow the racket to continue its path well after contact.")

        # Add generic recommendations if we don't have many specific ones
        if len(recommendations) < 3:
            recommendations.append("5. Practice proper weight transfer from back to front foot during the swing.")
            recommendations.append("6. Focus on the correct sequence of movement: legs → hips → torso → shoulder → elbow → wrist → racket.")
            recommendations.append("7. Record your swing regularly and compare to professional players to track improvement.")

        # Add recommendations to report
        for recommendation in recommendations:
            report_lines.append(recommendation)

        # Write report to file
        with open(output_path, 'w') as f:
            for line in report_lines:
                f.write(line + '\n')

        print(f"Metrics report generated and saved to {output_path}")

    def create_visualizations(self, pro_data, amateur_data, output_dir):
        """Create visualizations for key metrics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Plot Metrics Over Time
        metrics_to_plot = [
            ('shoulder_rotation_angle', 'Shoulder Rotation Angle'),
            ('knee_bend', 'Knee Bend Angle'),
            ('elbow_angle', 'Elbow Angle'),
            ('hip_rotation', 'Hip Rotation Angle')
        ]

        for metric_key, metric_title in metrics_to_plot:
            plt.figure(figsize=(12, 6))

            # Plot pro data
            if pro_data['metric_values_over_time'][metric_key]:
                frames, values = zip(*pro_data['metric_values_over_time'][metric_key])
                plt.plot(frames, values, 'g-', label='Professional', linewidth=2)

            # Plot amateur data
            if amateur_data['metric_values_over_time'][metric_key]:
                frames, values = zip(*amateur_data['metric_values_over_time'][metric_key])
                plt.plot(frames, values, 'r-', label='Amateur', linewidth=2)

            # Mark swing phases for pro
            for phase, frame in pro_data['swing_phases'].items():
                if frame is not None:
                    plt.axvline(x=frame, color='g', linestyle='--', alpha=0.5)
                    # Add phase label only for key moments
                    if phase in ['backswing_peak', 'impact', 'follow_through_peak']:
                        plt.annotate(phase.replace('_', ' ').title(),
                                    xy=(frame, plt.ylim()[0] + 5),
                                    color='g', rotation=90, fontsize=8)

            # Mark swing phases for amateur
            for phase, frame in amateur_data['swing_phases'].items():
                if frame is not None:
                    plt.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
                    # Add phase label only for key moments
                    if phase in ['backswing_peak', 'impact', 'follow_through_peak']:
                        plt.annotate(phase.replace('_', ' ').title(),
                                    xy=(frame, plt.ylim()[1] - 5),
                                    color='r', rotation=90, fontsize=8, va='top')

            plt.title(f"{metric_title} Throughout Swing")
            plt.xlabel("Frame Number")
            plt.ylabel("Angle (degrees)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save figure
            plt.savefig(str(output_dir / f"{metric_key}_comparison.png"))
            plt.close()
            print(f"Saved {metric_title} comparison to {output_dir / f'{metric_key}_comparison.png'}")

        # 2. Plot Wrist Path
        plt.figure(figsize=(10, 8))

        # Pro wrist path
        if pro_data['metric_values_over_time']['wrist_position']:
            _, x_coords, y_coords = zip(*pro_data['metric_values_over_time']['wrist_position'])
            plt.plot(x_coords, y_coords, 'g-', label='Professional')
            plt.scatter(x_coords, y_coords, c='green', s=30, alpha=0.7)

            # Mark key phases
            if pro_data['swing_phases']['backswing_peak'] is not None:
                idx = pro_data['frames'].index(pro_data['swing_phases']['backswing_peak'])
                if idx < len(x_coords):
                    plt.scatter(x_coords[idx], y_coords[idx], c='green', s=100, marker='*')
                    plt.annotate('Backswing', xy=(x_coords[idx], y_coords[idx]), color='g')

            if pro_data['swing_phases']['impact'] is not None:
                idx = pro_data['frames'].index(pro_data['swing_phases']['impact'])
                if idx < len(x_coords):
                    plt.scatter(x_coords[idx], y_coords[idx], c='green', s=100, marker='o')
                    plt.annotate('Impact', xy=(x_coords[idx], y_coords[idx]), color='g')

        # Amateur wrist path
        if amateur_data['metric_values_over_time']['wrist_position']:
            _, x_coords, y_coords = zip(*amateur_data['metric_values_over_time']['wrist_position'])
            plt.plot(x_coords, y_coords, 'r-', label='Amateur')
            plt.scatter(x_coords, y_coords, c='red', s=30, alpha=0.7)

            # Mark key phases
            if amateur_data['swing_phases']['backswing_peak'] is not None:
                idx = amateur_data['frames'].index(amateur_data['swing_phases']['backswing_peak'])
                if idx < len(x_coords):
                    plt.scatter(x_coords[idx], y_coords[idx], c='red', s=100, marker='*')
                    plt.annotate('Backswing', xy=(x_coords[idx], y_coords[idx]), color='r')

            if amateur_data['swing_phases']['impact'] is not None:
                idx = amateur_data['frames'].index(amateur_data['swing_phases']['impact'])
                if idx < len(x_coords):
                    plt.scatter(x_coords[idx], y_coords[idx], c='red', s=100, marker='o')
                    plt.annotate('Impact', xy=(x_coords[idx], y_coords[idx]), color='r')

        plt.title("Wrist Path Comparison")
        plt.xlabel("X position (normalized)")
        plt.ylabel("Y position (normalized)")
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        plt.grid(True)
        plt.legend()

        # Save figure
        plt.savefig(str(output_dir / "wrist_path_comparison.png"))
        plt.close()
        print(f"Saved wrist path comparison to {output_dir / 'wrist_path_comparison.png'}")

    def create_metrics_comparison_video(self, pro_frames, amateur_frames, output_path, output_fps=15):
        """Create comparison video with metrics overlaid."""
        if not pro_frames or not amateur_frames:
            print("Error: No frames to compare")
            return

        print(f"Creating metrics comparison video...")

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

            # Add metrics text to frames
            pro_frame_with_metrics = self.add_metrics_to_frame(pro_frame, self.pro_data, pro_frame_num, self.pro_color)
            amateur_frame_with_metrics = self.add_metrics_to_frame(amateur_frame, self.amateur_data, amateur_frame_num, self.amateur_color)

            # Resize frames preserving aspect ratio
            pro_frame_resized = cv2.resize(pro_frame_with_metrics, (pro_new_width, target_height))
            amateur_frame_resized = cv2.resize(amateur_frame_with_metrics, (amateur_new_width, target_height))

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
                print(f"Processed frame {i}/{num_output_frames}")

        # Release resources
        out.release()
        print(f"Metrics comparison video saved to {output_path}")

    def add_metrics_to_frame(self, frame, data, frame_num, color):
        """Add real-time metrics to a frame."""
        frame_with_metrics = frame.copy()
        height, width = frame.shape[:2]

        # Get current metric values for this frame
        metrics = {}

        for metric_key in data['metric_values_over_time']:
            for frame_data in data['metric_values_over_time'][metric_key]:
                if frame_data[0] == frame_num:
                    metrics[metric_key] = frame_data[1] if len(frame_data) > 1 else None
                    break

        # Add text showing current metric values
        text_y = 30
        text_spacing = 30

        # Show current phase
        current_phase = "Unknown"
        for phase, phase_frame in data['swing_phases'].items():
            if phase_frame is not None and phase_frame <= frame_num:
                current_phase = phase.replace('_', ' ').title()

        cv2.putText(frame_with_metrics, f"Phase: {current_phase}",
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        text_y += text_spacing

        # Show metrics
        metric_display = {
            'shoulder_rotation_angle': 'Shoulder Rot:',
            'knee_bend': 'Knee Angle:',
            'elbow_angle': 'Elbow Angle:',
            'hip_rotation': 'Hip Rotation:'
        }

        for metric_key, display_name in metric_display.items():
            if metric_key in metrics and metrics[metric_key] is not None:
                cv2.putText(frame_with_metrics, f"{display_name} {metrics[metric_key]:.1f}°",
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                text_y += text_spacing

        return frame_with_metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tennis forehand metrics analysis')
    parser.add_argument('--pro_video', default="jannik-sinner-forehands.mp4", help='Path to professional player video')
    parser.add_argument('--amateur_video', default="amateur-player-forehands.mov", help='Path to amateur player video')
    parser.add_argument('--output_dir', default="tennis_analysis_output", help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create analyzer
    analyzer = TennisMetricsAnalyzer()

    # Extract pose data from videos
    print("Extracting pose data from videos...")
    pro_frames = analyzer.extract_pose_data(args.pro_video, 'pro', skip_frames=2)
    amateur_frames = analyzer.extract_pose_data(args.amateur_video, 'amateur', skip_frames=2)

    # Create metrics report
    print("Generating metrics report...")
    analyzer.generate_metrics_report(
        analyzer.pro_data,
        analyzer.amateur_data,
        str(output_dir / "metrics_report.md")
    )

    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations(
        analyzer.pro_data,
        analyzer.amateur_data,
        output_dir
    )

    # Create metrics comparison video
    print("Creating metrics comparison video...")
    analyzer.create_metrics_comparison_video(
        pro_frames,
        amateur_frames,
        str(output_dir / "metrics_comparison.mp4")
    )

    print("Analysis complete!")
    print(f"Output files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
