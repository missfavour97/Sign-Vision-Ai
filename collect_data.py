import csv
import os
import cv2
import mediapipe as mp
import numpy as np

# Setup directories
DATA_DIR = "dataset"
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

# Define the sign you want to record right now (e.g., "A", "Hello", "Thanks")
TARGET_SIGN = input("Enter the name of the sign you are recording: ")
file_path = os.path.join(DATA_DIR, f"{TARGET_SIGN}.csv")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
print(f"\nRecording data for: '{TARGET_SIGN}'")
print(
    "Press 's' to start/pause saving samples. Press 'q' to quit and save"
    " dataset."
)

collecting = False
sample_count = 0

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(rgb_frame)

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
      )

      # If recording is active, extract 21 landmarks (x, y, z for each)
      if collecting:
        landmarks = []
        for lm in hand_landmarks.landmark:
          landmarks.extend([lm.x, lm.y, lm.z])

        # Append label at the end of the row
        landmarks.append(TARGET_SIGN)

        # Save to CSV file
        with open(file_path, mode="a", newline="") as f:
          writer = csv.writer(f)
          writer.writerow(landmarks)

        sample_count += 1
        print(
            f"Saved sample #{sample_count} for '{TARGET_SIGN}'", end="\r"
        )

  # Display status on screen
  status_text = (
      f"Recording: {'ON' if collecting else 'OFF'} | Samples: {sample_count}"
  )
  color = (0, 255, 0) if collecting else (0, 0, 255)
  cv2.putText(
      frame,
      status_text,
      (10, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      color,
      2,
      cv2.LINE_AA,
  )

  cv2.imshow("Sign Data Collector", frame)

  key = cv2.waitKey(1) & 0xFF
  if key == ord("s"):  # Press 's' to toggle recording
    collecting = not collecting
    print(
        f"\n{'Started' if collecting else 'Paused'} recording for"
        f" {TARGET_SIGN}"
    )
  elif key == ord("q"):  # Press 'q' to quit
    break

cap.release()
cv2.destroyAllWindows()
print(f"\nSaved {sample_count} samples to {file_path}")