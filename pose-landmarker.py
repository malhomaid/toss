import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Function to draw landmarks on each frame
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

        # Highlight knee landmarks specifically (larger circles and different color)
        # MediaPipe Pose model indices: 25-left knee, 26-right knee
        knee_indices = [25, 26]  # Left and right knee indices
        height, width, _ = annotated_image.shape

        for knee_idx in knee_indices:
            if len(pose_landmarks) > knee_idx:
                knee = pose_landmarks[knee_idx]
                knee_x = int(knee.x * width)
                knee_y = int(knee.y * height)
                # Draw a larger circle around the knee (red color)
                cv2.circle(annotated_image, (knee_x, knee_y), 15, (0, 0, 255), -1)

    return annotated_image

video_path = "IMG_2278 2.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create video writer for output
output_path = "tennis_forehand_analyzed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Set up the PoseLandmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Create detector
detector = vision.PoseLandmarker.create_from_options(options)

# Process video frame by frame
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    # Process every frame (you can adjust to skip frames for speed)

    # Convert the frame to RGB (MediaPipe uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect pose landmarks
    detection_result = detector.detect(mp_image)

    # Draw landmarks and additional visualizations
    annotated_frame = draw_landmarks_on_image(frame_rgb, detection_result)

    # Convert back to BGR for OpenCV output
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Add frame counter
    cv2.putText(annotated_frame_bgr, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write to output video
    out.write(annotated_frame_bgr)

    # Display the frame (comment out if running in headless environment)
    cv2.imshow('Tennis Pose Analysis', annotated_frame_bgr)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Analysis complete. Output saved to {output_path}")
