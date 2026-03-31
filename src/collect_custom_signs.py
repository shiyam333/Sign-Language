import cv2
import mediapipe as mp
import csv
import os

# Where your main CSV lives
CSV_PATH = "../data/asl_landmarks.csv"   # adjust if needed

LABEL = "G"       # change this to any word, e.g. "STOP", "LIKE"
NUM_SAMPLES = 150  # how many frames to record
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def main():
    cap = cv2.VideoCapture(0)

    # Open CSV in append mode
    file_exists = os.path.isfile(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        count = 0
        print(f"Show the '{LABEL}' gesture. Press 's' to start recording.")

        recording = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                if recording and count < NUM_SAMPLES:
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])

                    coords.append(LABEL)
                    writer.writerow(coords)
                    count += 1

            status_text = f"Label: {LABEL} | Samples: {count}/{NUM_SAMPLES}"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, "Press s = start, q = quit",
                        (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Collect custom sign", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                recording = True

            if recording and count >= NUM_SAMPLES:
                print("Finished recording samples.")
                break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
