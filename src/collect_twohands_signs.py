import cv2
import mediapipe as mp
import csv
import os

# Path to new two-hand CSV
CSV_PATH = "../data/twohand_signs.csv"

LABEL = "HI"      # change this for each word
NUM_SAMPLES = 200      # frames to record
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)

    # Create file if not exists, with header
    file_exists = os.path.isfile(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        # 21 points per hand * 3 coords * 2 hands = 126 + label
        header = []
        for hand in ["L", "R"]:
            for i in range(21):
                header += [f"x{i}_{hand}", f"y{i}_{hand}", f"z{i}_{hand}"]
        header.append("label")
        writer.writerow(header)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        count = 0
        print(f"Show the '{LABEL}' gesture with TWO HANDS. Press 's' to start recording.")

        recording = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w, _ = frame.shape

            if result.multi_hand_landmarks:
                # Draw all hands
                for hand_lms in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS
                    )

            if recording and result.multi_hand_landmarks:
                # Collect up to 2 hands, sorted left -> right
                hands_lms = result.multi_hand_landmarks
                hands_lms = sorted(
                    hands_lms,
                    key=lambda h_lm: h_lm.landmark[0].x
                )

                coords = []

                # LEFT hand (first)
                for lm in hands_lms[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                # RIGHT hand (second or zeros)
                if len(hands_lms) > 1:
                    for lm in hands_lms[1].landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                else:
                    coords.extend([0.0] * 63)

                coords.append(LABEL)
                writer.writerow(coords)
                count += 1

            status_text = f"Label: {LABEL} | Samples: {count}/{NUM_SAMPLES}"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, "Press s=start, q=quit",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Collect TWO-HAND sign", frame)

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
