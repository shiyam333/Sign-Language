import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# 1. Load model and labels
model = tf.keras.models.load_model("../models/asl_mlp.h5")
label_classes = np.load("../models/label_classes.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sentence = ""          # accumulated text
pred_history = []      # last N predictions
HISTORY_SIZE = 15
CONF_THRESH = 0.8

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: fix output size so it nicely fills half / full screen
        frame = cv2.resize(frame, (960, 540))   # width, height

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_pred = None
        current_prob = 0.0

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Build input vector: x0,y0,z0,...,x20,y20,z20
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            x_input = np.array(coords, dtype=np.float32).reshape(1, -1)

            # Predict
            probs = model.predict(x_input, verbose=0)[0]
            idx = np.argmax(probs)
            current_prob = probs[idx]
            current_pred = label_classes[idx]

            # Show current prediction (top-left)
            cv2.putText(
                frame,
                f"{current_pred} ({current_prob:.2f})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if current_prob > CONF_THRESH else (0, 255, 255),
                2,
            )

            # Update history only if above some minimal prob
            if current_prob > 0.6:
                pred_history.append(current_pred)
                if len(pred_history) > HISTORY_SIZE:
                    pred_history.pop(0)

                # Check if stable
                if len(pred_history) == HISTORY_SIZE:
                    most_common = max(set(pred_history), key=pred_history.count)
                    freq = pred_history.count(most_common)

                    # Require stability
                    if freq > HISTORY_SIZE * 0.7:
                        if most_common == "space":
                            sentence += " "
                        elif most_common == "del":
                            sentence = sentence[:-1]
                        else:
                            sentence += most_common

                        pred_history.clear()

        # Get frame height for positioning text
        h, w, _ = frame.shape

        # Display sentence just above bottom
        cv2.putText(
            frame,
            sentence,
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Quit hint at very bottom
        cv2.putText(
            frame,
            "Press q to quit",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Show only the frame (no extra grey canvas)
        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
