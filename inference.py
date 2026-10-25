import cv2
import mediapipe as mp
import numpy as np
import joblib

# Force load MediaPipe solutions correctly
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions as mp_solutions
    mp_hands = mp_solutions.hands
    mp_drawing = mp_solutions.drawing_utils

# 1. Load your trained model
model = joblib.load('sign_language_model.pkl')

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2. Open Webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a selfie-view display, and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the video frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 (x, y, z) coordinates to match the 63features the model was trained on
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Reshape for the model prediction
            features = np.array(landmarks).reshape(1, -1)

            try:

                 # Predict the sign digit
                 prediction = model.predict(features)[0]
                 confidence_score = np.max(model.predict_proba(features))

                 # Display prediction text on screen
                 text = f"Sign: {prediction} ({confidence_score * 100:.1f}%)"
                 cv2.putText(
                     frame, text, (50, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
                ) 
            except Exception as e:
                print(f"Error during prediction: {e}")

    # Show the live feed window
    cv2.imshow('Sign Language Translator', frame)

    # Break loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()