import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose and Face Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Replace the webcam initialization with video file path
video_path = "rr.webm"  # Specify the path to your .webm file
cap = cv2.VideoCapture(video_path)

# Lists to store data
stability_values = []
distance_values = []
expression_changes = []

last_expression = None
expression_stable_count = 0

# Helper function to detect facial expression
def detect_expression(face_landmarks):
    if face_landmarks is None:
        return "Neutral"

    # Use mouth landmarks to detect smile
    try:
        upper_lip = face_landmarks.landmark[13].y
        lower_lip = face_landmarks.landmark[14].y
        mouth_gap = lower_lip - upper_lip

        # Basic threshold for smile detection
        if mouth_gap > 0.05:
            return "Smiling"
        else:
            return "Neutral"
    except:
        return "Neutral"

frame_count = 0  # To keep track of the number of frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    pose_results = pose.process(image)

    # Detect face
    face_results = mp_face_mesh.process(image)

    if pose_results.pose_landmarks and face_results.multi_face_landmarks:
        # Calculate stability by checking movement of a reference landmark (e.g., nose)
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        stability_value = np.abs(nose.x - 0.5) * 100  # Stability based on deviation from center
        stability_values.append(stability_value)

        # Estimate distance from the camera based on nose z-coordinate
        distance_value = max(0, nose.z * -100)  # Scaling z to a meaningful distance
        distance_values.append(distance_value)

        # Detect facial expression
        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        expression = detect_expression(face_landmarks)

        if last_expression is not None and expression != last_expression:
            expression_changes.append(1)  # Count change as 1
        else:
            expression_changes.append(0)  # No change

        last_expression = expression

# Release the video file
cap.release()

# Calculate averages and summarize the results
avg_stability = np.mean(stability_values)
avg_distance = np.mean(distance_values)
expression_stability_ratio = sum(expression_changes) / len(expression_changes) if expression_changes else 0

# Convert stability to percentage (lower stability value is better)
stability_percentage = 100 - avg_stability if avg_stability <= 100 else 0

# Use average distance directly (now in a more realistic scale)
distance_percentage = max(0, min(100, (avg_distance - 20) / (100 - 20) * 100))

# Stability analysis
stability_description = "Stable" if stability_percentage > 80 else "Unstable"

# Distance analysis
distance_description = "Good" if 40 <= distance_percentage <= 60 else ("Too Close" if distance_percentage < 40 else "Too Far")

# Facial expression analysis
expression_description = "Facial expressions were mostly stable." if expression_stability_ratio > 0.8 else "Facial expressions changed frequently."

# Print the summary
print(f"Posture Stability: {stability_description} (Stability: {stability_percentage:.2f}%)")
print(f"Distance from Camera: {distance_description} (Distance: {avg_distance:.2f} cm)")
print(f"Facial Expression Analysis: {expression_description}")

# Plot line graphs
plt.figure(figsize=(15, 7))

# Stability Line Graph
plt.subplot(3, 1, 1)
plt.plot(stability_values, label='Stability', color='blue')
plt.xlabel('Frame Number')
plt.ylabel('Stability (%)')
plt.title('Posture Stability Over Time')
plt.legend()

# Distance Line Graph
plt.subplot(3, 1, 2)
plt.plot(distance_values, label='Distance', color='green')
plt.xlabel('Frame Number')
plt.ylabel('Distance (cm)')
plt.title('Distance from Camera Over Time')
plt.legend()

# Facial Expression Changes Line Graph
plt.subplot(3, 1, 3)
plt.plot(expression_changes, label='Expression Changes', color='orange')
plt.xlabel('Frame Number')
plt.ylabel('Expression Change (1=Change, 0=No Change)')
plt.title('Facial Expression Changes Over Time')
plt.legend()

plt.tight_layout()
plt.show()
