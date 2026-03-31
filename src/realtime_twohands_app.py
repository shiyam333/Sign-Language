import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading
from collections import deque
import queue
import subprocess
import os


# =========================================================
# ===================== WINDOWS TTS (Non-blocking) =======
# =========================================================
class WindowsTTS:
    def __init__(self):
        self.tts_queue = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._worker_thread, daemon=True)
        self.worker.start()

    def _worker_thread(self):
        """Play TTS using Windows PowerShell (non-blocking)"""
        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text and len(text.strip()) > 0:
                    # Use Windows PowerShell to speak (non-blocking)
                    ps_command = f'Add-Type -AssemblyName System.speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
                    subprocess.Popen(
                        ["powershell", "-Command", ps_command],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except queue.Empty:
                pass
            except Exception as e:
                print(f"TTS Error: {e}")

    def speak(self, text):
        """Queue text to speak"""
        if text and len(text.strip()) > 0:
            self.tts_queue.put(text)

    def stop(self):
        self.running = False


tts = WindowsTTS()


# =========================================================
# ======================= LOAD MODEL ======================
# =========================================================
model = tf.keras.models.load_model("../models/twohand_mlp.h5")
labels = np.load("../models/twohand_label_classes.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


# =========================================================
# ==================== STATE VARIABLES ====================
# =========================================================
frame_window = deque(maxlen=10)
sentence = ""

CONF_THRESHOLD = 0.75
STABILITY_RATIO = 0.65
last_spoken = None


# =========================================================
# ===================== UI COLORS =========================
# =========================================================
BLUE_TOP = (40, 90, 160)
BLUE_BOTTOM = (20, 40, 90)
WHITE = (255, 255, 255)
CYAN = (120, 220, 255)
GREEN = (50, 255, 100)
RED = (50, 100, 255)


def draw_gradient(panel):
    h = panel.shape[0]
    for i in range(h):
        alpha = i / h
        color = (
            int(BLUE_TOP[0] * (1 - alpha) + BLUE_BOTTOM[0] * alpha),
            int(BLUE_TOP[1] * (1 - alpha) + BLUE_BOTTOM[1] * alpha),
            int(BLUE_TOP[2] * (1 - alpha) + BLUE_BOTTOM[2] * alpha),
        )
        panel[i, :] = color


# =========================================================
# ======================= MAIN LOOP =======================
# =========================================================
print("Starting Sign Language Translator...")
print("Press Q to quit, C to clear, S to speak all\n")

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(cv2.resize(frame, (760, 720)), 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_word = "None"
        current_conf = 0.0

        if result.multi_hand_landmarks:
            for h in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, h, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(100, 255, 100), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(100, 200, 100), thickness=2)
                )

            hands_sorted = sorted(
                result.multi_hand_landmarks,
                key=lambda h: h.landmark[0].x
            )

            coords = []
            for lm in hands_sorted[0].landmark:
                coords.extend([lm.x, lm.y, lm.z])

            if len(hands_sorted) > 1:
                for lm in hands_sorted[1].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
            else:
                coords.extend([0.0] * 63)

            x = np.array(coords, dtype=np.float32).reshape(1, -1)
            probs = model.predict(x, verbose=0)[0]

            idx = np.argmax(probs)
            current_conf = float(probs[idx])
            current_word = str(labels[idx])

            # ----- CORE LOGIC -----
            if current_conf > CONF_THRESHOLD:
                frame_window.append(current_word)

                if len(frame_window) == frame_window.maxlen:
                    most_common = max(set(frame_window), key=frame_window.count)
                    freq = frame_window.count(most_common)
                    stability = freq / len(frame_window)

                    # Speak if stable AND new word
                    if stability >= STABILITY_RATIO and most_common != last_spoken:
                        if most_common == "space":
                            sentence += " "
                            print(f"✓ [SPACE]")
                        elif most_common == "del":
                            sentence = sentence[:-1] if sentence else ""
                            print(f"✓ [DELETE]")
                        else:
                            sentence += most_common + " "
                            # SPEAK IMMEDIATELY (non-blocking)
                            tts.speak(most_common)
                            print(f"✓ [SPEAKING] '{most_common}' → {sentence.strip()}")

                        last_spoken = most_common

        else:
            frame_window.clear()
            last_spoken = None

        # ================= UI PANEL =================
        panel = np.zeros((720, 500, 3), dtype=np.uint8)
        draw_gradient(panel)

        cv2.putText(
            panel,
            "SIGN LANGUAGE TRANSLATOR",
            (15, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            WHITE,
            2,
        )

        conf_color = GREEN if current_conf > CONF_THRESHOLD else RED
        cv2.rectangle(panel, (15, 70), (485, 150), conf_color, 2)
        cv2.putText(
            panel,
            "Current Gesture",
            (25, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            1,
        )
        cv2.putText(
            panel,
            current_word,
            (25, 130),
            cv2.FONT_HERSHEY_DUPLEX,
            1.3,
            WHITE,
            2,
        )
        cv2.putText(
            panel,
            f"Conf: {current_conf:.2f}",
            (320, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            1,
        )

        cv2.rectangle(panel, (15, 170), (485, 650), CYAN, 2)
        cv2.putText(
            panel,
            "Recognized Text:",
            (25, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            1,
        )

        # Sentence display
        y = 240
        words = sentence.split() if sentence.strip() else ["[Waiting for gestures...]"]
        line = ""

        for word in words:
            test_line = (line + " " + word).strip() if line else word
            width = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]

            if width > 450 and line:
                cv2.putText(panel, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)
                y += 35
                line = word
            else:
                line = test_line

        if line:
            cv2.putText(panel, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

        queue_size = tts.tts_queue.qsize()
        status_color = GREEN if queue_size == 0 else CYAN
        cv2.putText(
            panel,
            f"Speaking: {queue_size} words in queue",
            (15, 690),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            1,
        )

        cv2.putText(
            panel,
            "C=Clear | S=Speak | Q=Quit",
            (280, 690),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            CYAN,
            1,
        )

        combined = np.hstack((frame, panel))
        cv2.imshow("Real-Time Sign Language Translator", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""
            last_spoken = None
            frame_window.clear()
        elif key == ord("s"):
            tts.speak(sentence)

cap.release()
cv2.destroyAllWindows()
tts.stop()
