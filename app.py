"""
SignBridge - app.py  (SQLite edition)
======================================
Uses Python's built-in sqlite3 — no MySQL, no Workbench, no config.
The database file  signbridge.db  is created automatically in this folder.

SETUP:
  pip install flask flask-cors opencv-python "numpy<2" mediapipe tensorflow deep-translator gtts
  python app.py
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import threading
import queue
import os
import sqlite3
import tempfile
import platform
import hashlib
import secrets
import random
from datetime import datetime

# ── Optional deps ──────────────────────────────────────────────
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("INFO: gTTS not installed — TTS disabled.")

try:
    from deep_translator import GoogleTranslator
    TRANSLATE_AVAILABLE = True
except ImportError:
    TRANSLATE_AVAILABLE = False
    print("INFO: deep-translator not installed — translation disabled.")

# ── GPU config ─────────────────────────────────────────────────
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# ── Flask ───────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# ─────────────────────────────────────────────────────────────
# SQLite  — file lives next to this script, created on first run
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "signbridge.db")


def get_db():
    """Open a connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables on first run. Safe to call every startup."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            avatar_color  TEXT DEFAULT '#2dd4bf',
            created_at    TEXT DEFAULT (datetime('now')),
            last_login    TEXT
        );

        CREATE TABLE IF NOT EXISTS user_progress (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            lesson_id      TEXT    NOT NULL,
            sign_word      TEXT    NOT NULL,
            attempts       INTEGER DEFAULT 0,
            correct_count  INTEGER DEFAULT 0,
            accuracy       REAL    DEFAULT 0.0,
            completed      INTEGER DEFAULT 0,
            last_practiced TEXT,
            UNIQUE (user_id, lesson_id, sign_word),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS achievements (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            badge_name  TEXT NOT NULL,
            badge_icon  TEXT DEFAULT '🏅',
            earned_at   TEXT DEFAULT (datetime('now')),
            UNIQUE (user_id, badge_name),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()
    print(f"Database ready  →  {DB_PATH}")


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
MODEL_LOADED = False
model  = None
labels = np.array([
    "Hello", "Thank You", "Yes",  "No",    "Please",
    "Help",  "Sorry",     "Good", "Bad",   "Love",
    "Happy", "Sad",       "Stop", "More",  "Again",
])

print("Loading sign language model...")
try:
    _labels     = np.load("models/twohand_label_classes.npy", allow_pickle=True)
    num_classes = len(_labels)
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(126,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    _model.load_weights("models/twohand_mlp.h5", by_name=True, skip_mismatch=True)
    _model.predict(np.random.rand(1, 126).astype(np.float32), verbose=0)
    model = _model
    labels = _labels
    MODEL_LOADED = True
    print("Model loaded and warmed up.")
except Exception as e:
    print(f"Model load failed (demo mode): {e}")

# ── MediaPipe ───────────────────────────────────────────────────
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────────────────────────
# LANGUAGES
# ─────────────────────────────────────────────────────────────
LANGUAGES = {
    'en': {'name': 'English',   'lang_code': 'en'},
    'ta': {'name': 'Tamil',     'lang_code': 'ta'},
    'hi': {'name': 'Hindi',     'lang_code': 'hi'},
    'te': {'name': 'Telugu',    'lang_code': 'te'},
    'ml': {'name': 'Malayalam', 'lang_code': 'ml'},
    'kn': {'name': 'Kannada',   'lang_code': 'kn'},
    'mr': {'name': 'Marathi',   'lang_code': 'mr'},
    'gu': {'name': 'Gujarati',  'lang_code': 'gu'},
    'bn': {'name': 'Bengali',   'lang_code': 'bn'},
    'pa': {'name': 'Punjabi',   'lang_code': 'pa'},
}

# ─────────────────────────────────────────────────────────────
# SIGN DICTIONARY
# ─────────────────────────────────────────────────────────────
SIGN_DICTIONARY = {
    "Greetings": [
        {"word": "Hello",     "emoji": "👋", "difficulty": "beginner",
         "description": "Wave your open hand side to side at shoulder height.",
         "tip": "Keep palm facing out and swing from the wrist."},
        {"word": "Goodbye",   "emoji": "👐", "difficulty": "beginner",
         "description": "Wave your hand outward away from your body.",
         "tip": "Open hand, sweep away smoothly."},
        {"word": "Thank You", "emoji": "🙏", "difficulty": "beginner",
         "description": "Touch flat hand to chin and move forward.",
         "tip": "Touch fingertips to chin, extend outward."},
        {"word": "Please",    "emoji": "🫶", "difficulty": "beginner",
         "description": "Rub flat hand in circles on your chest.",
         "tip": "Flat hand, circular motion on upper chest."},
        {"word": "Sorry",     "emoji": "😔", "difficulty": "beginner",
         "description": "Circle a closed fist on your chest.",
         "tip": "Closed fist, rub clockwise circles on chest."},
        {"word": "Welcome",   "emoji": "🤗", "difficulty": "intermediate",
         "description": "Sweep your open arm inward toward your body.",
         "tip": "Arm outstretched, sweep inward like an invitation."},
    ],
    "Basics": [
        {"word": "Yes",   "emoji": "✅", "difficulty": "beginner",
         "description": "Nod a closed fist up and down.",
         "tip": "Make a fist and bob it up and down like nodding."},
        {"word": "No",    "emoji": "❌", "difficulty": "beginner",
         "description": "Tap index and middle finger against thumb.",
         "tip": "Extend two fingers and thumb, tap repeatedly."},
        {"word": "Help",  "emoji": "🆘", "difficulty": "intermediate",
         "description": "Lift a thumbs-up fist with the other hand.",
         "tip": "One hand flat, lift the thumbs-up on top of it."},
        {"word": "Stop",  "emoji": "🛑", "difficulty": "beginner",
         "description": "Chop horizontal hand into vertical palm.",
         "tip": "Flat horizontal hand chops down into palm."},
        {"word": "More",  "emoji": "➕", "difficulty": "beginner",
         "description": "Tap pinched fingertips of both hands.",
         "tip": "Both hands pinch O-shape, tap together twice."},
        {"word": "Again", "emoji": "🔄", "difficulty": "intermediate",
         "description": "Arc a curved hand down into flat palm.",
         "tip": "Curved hand arcs over and taps flat palm."},
    ],
    "Emotions": [
        {"word": "Happy",   "emoji": "😊", "difficulty": "beginner",
         "description": "Brush both hands upward on chest.",
         "tip": "Both open hands, brush up on chest two times."},
        {"word": "Sad",     "emoji": "😢", "difficulty": "beginner",
         "description": "Slide both open hands down your face.",
         "tip": "Open hands at eyes, slide slowly downward."},
        {"word": "Love",    "emoji": "❤️", "difficulty": "beginner",
         "description": "Cross both arms over your heart.",
         "tip": "Cross wrists, press closed fists to chest."},
        {"word": "Angry",   "emoji": "😡", "difficulty": "intermediate",
         "description": "Pull claw-shaped hands away from your face.",
         "tip": "Claw hands at face, pull outward and tense up."},
        {"word": "Scared",  "emoji": "😨", "difficulty": "intermediate",
         "description": "Both hands jump to chest suddenly.",
         "tip": "Hands out to sides, snap quickly to chest."},
        {"word": "Excited", "emoji": "🤩", "difficulty": "intermediate",
         "description": "Alternate middle fingers brush up on chest.",
         "tip": "Alternate hands, middle fingers brush upward."},
    ],
    "Numbers": [
        {"word": "One",   "emoji": "1️⃣", "difficulty": "beginner",
         "description": "Hold up one index finger.",
         "tip": "Only index finger extended, others closed."},
        {"word": "Two",   "emoji": "2️⃣", "difficulty": "beginner",
         "description": "Hold up index and middle fingers.",
         "tip": "Peace sign, palm facing outward."},
        {"word": "Three", "emoji": "3️⃣", "difficulty": "beginner",
         "description": "Hold up thumb, index, and middle.",
         "tip": "Three fingers spread, palm forward."},
        {"word": "Four",  "emoji": "4️⃣", "difficulty": "beginner",
         "description": "Hold up four fingers, thumb folded.",
         "tip": "All except thumb extended, palm out."},
        {"word": "Five",  "emoji": "5️⃣", "difficulty": "beginner",
         "description": "Open hand with all five fingers spread.",
         "tip": "All five fingers spread wide, palm forward."},
        {"word": "Ten",   "emoji": "🔟", "difficulty": "intermediate",
         "description": "Shake a thumbs-up hand side to side.",
         "tip": "Fist with thumb up, wiggle side to side."},
    ],
    "Family": [
        {"word": "Mother", "emoji": "👩", "difficulty": "beginner",
         "description": "Touch thumb of open hand to chin.",
         "tip": "A-handshape, tap thumb on chin twice."},
        {"word": "Father", "emoji": "👨", "difficulty": "beginner",
         "description": "Touch thumb of open hand to forehead.",
         "tip": "A-handshape, tap thumb on forehead twice."},
        {"word": "Baby",   "emoji": "👶", "difficulty": "beginner",
         "description": "Cradle arms and rock side to side.",
         "tip": "Fold arms, rock like holding a baby."},
        {"word": "Friend", "emoji": "🤝", "difficulty": "intermediate",
         "description": "Hook index fingers and flip hands back-forth.",
         "tip": "Interlock bent index fingers, switch orientation."},
    ],
}

# ─────────────────────────────────────────────────────────────
# LESSONS
# ─────────────────────────────────────────────────────────────
LESSONS = [
    {
        "id": "lesson_1", "title": "First Steps",
        "subtitle": "Essential greetings and responses",
        "level": 1, "icon": "🌟", "color": "#2dd4bf",
        "signs": ["Hello", "Goodbye", "Thank You", "Please", "Yes", "No"],
        "xp_reward": 100, "prerequisite": None,
    },
    {
        "id": "lesson_2", "title": "Express Yourself",
        "subtitle": "Emotions and feelings",
        "level": 2, "icon": "💫", "color": "#f59e0b",
        "signs": ["Happy", "Sad", "Love", "Angry", "Scared", "Excited"],
        "xp_reward": 150, "prerequisite": "lesson_1",
    },
    {
        "id": "lesson_3", "title": "Count It Out",
        "subtitle": "Numbers 1 through 10",
        "level": 3, "icon": "🔢", "color": "#8b5cf6",
        "signs": ["One", "Two", "Three", "Four", "Five", "Ten"],
        "xp_reward": 150, "prerequisite": "lesson_1",
    },
    {
        "id": "lesson_4", "title": "Daily Basics",
        "subtitle": "Common everyday signs",
        "level": 4, "icon": "📚", "color": "#ec4899",
        "signs": ["Sorry", "Help", "Stop", "More", "Again", "Welcome"],
        "xp_reward": 200, "prerequisite": "lesson_2",
    },
    {
        "id": "lesson_5", "title": "Family Bonds",
        "subtitle": "Family and relationships",
        "level": 5, "icon": "👨‍👩‍👧", "color": "#f97316",
        "signs": ["Mother", "Father", "Baby", "Friend", "Love", "Happy"],
        "xp_reward": 250, "prerequisite": "lesson_3",
    },
]

# ─────────────────────────────────────────────────────────────
# TTS QUEUE
# ─────────────────────────────────────────────────────────────
tts_queue = queue.Queue(maxsize=50)

AVATAR_COLORS = ['#2dd4bf', '#f59e0b', '#8b5cf6', '#ec4899', '#f97316', '#06b6d4']


def translate_text(text, target_lang):
    if not text or target_lang == 'en' or not TRANSLATE_AVAILABLE:
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def play_tts(text, lang_code):
    if not GTTS_AVAILABLE or not text:
        return
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        path = tmp.name
        tmp.close()
        gTTS(text=text, lang=lang_code, slow=False).save(path)
        sys = platform.system()
        if sys == 'Windows':
            os.startfile(path)
        elif sys == 'Darwin':
            os.system(f'afplay "{path}" &')
        else:
            os.system(f'mpg123 -q "{path}" 2>/dev/null &')
        def _clean():
            import time; time.sleep(10)
            try: os.remove(path)
            except: pass
        threading.Thread(target=_clean, daemon=True).start()
    except Exception as e:
        print(f"TTS error: {e}")


def tts_worker():
    while True:
        try:
            data = tts_queue.get(timeout=1)
            if data:
                text = str(data.get('text', '')).strip()
                lang = data.get('lang', 'en')
                if text and text.lower() != 'none':
                    lang_code = LANGUAGES.get(lang, {}).get('lang_code', 'en')
                    play_tts(text, lang_code)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS worker error: {e}")


threading.Thread(target=tts_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def row_to_dict(row):
    """Convert sqlite3.Row to plain dict."""
    return dict(row) if row else None


def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    try:
        conn = get_db()
        row  = conn.execute(
            "SELECT id, username, email, avatar_color, created_at FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        user = row_to_dict(row)
        if user.get('created_at'):
            try:
                user['created_at'] = datetime.strptime(user['created_at'][:19], '%Y-%m-%d %H:%M:%S')
            except Exception:
                user['created_at'] = None
        return user
    except Exception as e:
        print(f"get_current_user error: {e}")
        return None


def get_user_stats(user_id):
    blank = {'total_signs': 0, 'avg_accuracy': 0.0,
             'total_attempts': 0, 'completed_lessons': 0, 'badges': []}
    try:
        conn = get_db()
        cur  = conn.cursor()

        row = cur.execute("""
            SELECT
                COUNT(DISTINCT sign_word)           AS total_signs,
                COALESCE(AVG(accuracy),    0.0)     AS avg_accuracy,
                COALESCE(SUM(attempts),    0)       AS total_attempts,
                COALESCE(SUM(correct_count), 0)     AS total_correct
            FROM user_progress WHERE user_id = ?
        """, (user_id,)).fetchone()

        stats = row_to_dict(row) or blank.copy()

        completed = 0
        for lesson in LESSONS:
            r = cur.execute("""
                SELECT COUNT(*) AS cnt FROM user_progress
                WHERE user_id = ? AND lesson_id = ? AND accuracy >= 70
            """, (user_id, lesson['id'])).fetchone()
            if r and r['cnt'] >= len(lesson['signs']):
                completed += 1
        stats['completed_lessons'] = completed

        badge_rows = cur.execute("""
            SELECT badge_name, badge_icon, earned_at
            FROM achievements WHERE user_id = ? ORDER BY earned_at DESC
        """, (user_id,)).fetchall()

        stats['badges'] = []
        for b in badge_rows:
            bd = row_to_dict(b)
            if bd.get('earned_at'):
                try:
                    bd['earned_at'] = datetime.strptime(bd['earned_at'][:19], '%Y-%m-%d %H:%M:%S')
                except Exception:
                    bd['earned_at'] = None
            stats['badges'].append(bd)

        conn.close()
        return stats
    except Exception as e:
        print(f"get_user_stats error: {e}")
        return blank


# ─────────────────────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", user=get_current_user())


@app.route("/translator")
def translator():
    return render_template("translator.html", user=get_current_user())


@app.route("/dictionary")
def dictionary():
    return render_template("dictionary.html", user=get_current_user(),
                           sign_dictionary=SIGN_DICTIONARY)


@app.route("/learn")
def learn():
    user = get_current_user()
    if not user:
        return redirect(url_for('login_page'))

    progress = {}
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT lesson_id, sign_word, accuracy, completed FROM user_progress WHERE user_id = ?",
            (user['id'],)
        ).fetchall()
        conn.close()
        for row in rows:
            lid = row['lesson_id']
            progress.setdefault(lid, []).append(row_to_dict(row))
    except Exception as e:
        print(f"learn progress error: {e}")

    stats = get_user_stats(user['id'])
    return render_template("learn.html", user=user, lessons=LESSONS,
                           progress=progress, stats=stats)


@app.route("/lesson/<lesson_id>")
def lesson_detail(lesson_id):
    user = get_current_user()
    if not user:
        return redirect(url_for('login_page'))

    lesson = next((l for l in LESSONS if l['id'] == lesson_id), None)
    if not lesson:
        return redirect(url_for('learn'))

    progress = {}
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT sign_word, attempts, correct_count, accuracy, completed
            FROM user_progress WHERE user_id = ? AND lesson_id = ?
        """, (user['id'], lesson_id)).fetchall()
        conn.close()
        for row in rows:
            progress[row['sign_word']] = row_to_dict(row)
    except Exception as e:
        print(f"lesson_detail error: {e}")

    return render_template("lesson.html", user=user, lesson=lesson,
                           progress=progress, sign_dictionary=SIGN_DICTIONARY)


@app.route("/about")
def about():
    return render_template("about.html", user=get_current_user())


@app.route("/login", methods=["GET"])
def login_page():
    if get_current_user():
        return redirect(url_for('learn'))
    return render_template("login.html")


@app.route("/register", methods=["GET"])
def register_page():
    if get_current_user():
        return redirect(url_for('learn'))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route("/profile")
def profile():
    user = get_current_user()
    if not user:
        return redirect(url_for('login_page'))
    stats = get_user_stats(user['id'])
    return render_template("profile.html", user=user, stats=stats, lessons=LESSONS)


# ─────────────────────────────────────────────────────────────
# ROUTES — AUTH API
# ─────────────────────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def api_login():
    data       = request.get_json(silent=True) or {}
    identifier = str(data.get("identifier", "")).strip()
    password   = str(data.get("password",   "")).strip()

    if not identifier or not password:
        return jsonify({"success": False, "error": "All fields are required."}), 400

    try:
        conn = get_db()
        row  = conn.execute(
            "SELECT * FROM users WHERE email = ? OR username = ?",
            (identifier, identifier)
        ).fetchone()

        if row and row['password_hash'] == hash_password(password):
            conn.execute(
                "UPDATE users SET last_login = datetime('now') WHERE id = ?",
                (row['id'],)
            )
            conn.commit()
            conn.close()
            session['user_id'] = row['id']
            return jsonify({"success": True, "redirect": "/learn"})

        conn.close()
        return jsonify({"success": False, "error": "Invalid email/username or password."}), 401

    except Exception as e:
        print(f"login error: {e}")
        return jsonify({"success": False, "error": "Server error."}), 500


@app.route("/api/register", methods=["POST"])
def api_register():
    data     = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    email    = str(data.get("email",    "")).strip()
    password = str(data.get("password", "")).strip()

    if not username or not email or not password:
        return jsonify({"success": False, "error": "All fields are required."}), 400
    if len(username) < 3:
        return jsonify({"success": False, "error": "Username must be at least 3 characters."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400
    if '@' not in email:
        return jsonify({"success": False, "error": "Please enter a valid email."}), 400

    try:
        conn  = get_db()
        color = random.choice(AVATAR_COLORS)
        conn.execute(
            "INSERT INTO users (username, email, password_hash, avatar_color) VALUES (?, ?, ?, ?)",
            (username, email, hash_password(password), color)
        )
        conn.commit()
        user_id = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()['id']
        conn.close()
        session['user_id'] = user_id
        return jsonify({"success": True, "redirect": "/learn"})

    except sqlite3.IntegrityError as e:
        err = str(e).lower()
        if 'username' in err:
            msg = "Username is already taken."
        elif 'email' in err:
            msg = "Email is already registered."
        else:
            msg = "Registration failed. Please try again."
        return jsonify({"success": False, "error": msg}), 400
    except Exception as e:
        print(f"register error: {e}")
        return jsonify({"success": False, "error": "Server error."}), 500


# ─────────────────────────────────────────────────────────────
# ROUTES — TRANSLATOR API
# ─────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data       = request.get_json(silent=True) or {}
        frame_data = data.get("frame", "")
        if not frame_data or ',' not in frame_data:
            return jsonify({"success": False, "error": "No frame data."})

        img_bytes = base64.b64decode(frame_data.split(",")[1])
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Invalid image."})

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_word = "None"
        current_conf = 0.0

        with mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
            model_complexity=0,
        ) as det:
            result = det.process(rgb)

        if result.multi_hand_landmarks:
            sorted_hands = sorted(result.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
            coords = []
            for lm in sorted_hands[0].landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if len(sorted_hands) > 1:
                for lm in sorted_hands[1].landmark:
                    coords.extend([lm.x, lm.y, lm.z])
            else:
                coords.extend([0.0] * 63)

            if MODEL_LOADED and model is not None:
                x     = np.array(coords, dtype=np.float32).reshape(1, -1)
                probs = model.predict(x, verbose=0)[0]
                idx   = int(np.argmax(probs))
                current_conf = float(probs[idx])
                if current_conf > 0.6:
                    current_word = str(labels[idx])

            for hand_lm in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        return jsonify({"success": True, "gesture": current_word,
                        "confidence": round(current_conf, 3), "frame": frame_b64})

    except Exception as e:
        print(f"predict error: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    lang = str(data.get("lang", "en")).strip()
    if not text or lang == "en" or lang not in LANGUAGES:
        return jsonify({"translated": text})
    return jsonify({"translated": translate_text(text, lang)})


@app.route("/api/translate-word", methods=["POST"])
def api_translate_word():
    data = request.get_json(silent=True) or {}
    word = str(data.get("word", "")).strip()
    lang = str(data.get("lang", "en")).strip()
    if not word or word.lower() == "none" or lang not in LANGUAGES:
        return jsonify({"translated": word})
    return jsonify({"original": word, "translated": translate_text(word, lang), "lang": lang})


@app.route("/api/speak", methods=["POST"])
def api_speak():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    lang = str(data.get("lang", "en")).strip()
    if text and text.lower() != "none":
        try:
            tts_queue.put_nowait({"text": text, "lang": lang})
        except queue.Full:
            pass
    return jsonify({"success": True})


@app.route("/api/languages", methods=["GET"])
def api_languages():
    return jsonify({"languages": {k: v['name'] for k, v in LANGUAGES.items()}})


# ─────────────────────────────────────────────────────────────
# ROUTES — LESSON / PROGRESS API
# ─────────────────────────────────────────────────────────────

@app.route("/api/lesson/progress", methods=["POST"])
def api_lesson_progress():
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "error": "Not logged in."}), 401

    data      = request.get_json(silent=True) or {}
    lesson_id = str(data.get("lesson_id", "")).strip()
    sign_word = str(data.get("sign_word", "")).strip()
    correct   = bool(data.get("correct", False))

    if not lesson_id or not sign_word:
        return jsonify({"success": False, "error": "Missing fields."}), 400

    try:
        conn = get_db()
        cur  = conn.cursor()

        existing = cur.execute("""
            SELECT attempts, correct_count FROM user_progress
            WHERE user_id = ? AND lesson_id = ? AND sign_word = ?
        """, (user['id'], lesson_id, sign_word)).fetchone()

        if existing:
            new_att  = existing['attempts'] + 1
            new_cor  = existing['correct_count'] + (1 if correct else 0)
            new_acc  = (new_cor / new_att) * 100.0
            cur.execute("""
                UPDATE user_progress
                SET attempts = ?, correct_count = ?, accuracy = ?,
                    last_practiced = datetime('now')
                WHERE user_id = ? AND lesson_id = ? AND sign_word = ?
            """, (new_att, new_cor, new_acc, user['id'], lesson_id, sign_word))
        else:
            new_acc = 100.0 if correct else 0.0
            cur.execute("""
                INSERT INTO user_progress
                    (user_id, lesson_id, sign_word, attempts, correct_count, accuracy, last_practiced)
                VALUES (?, ?, ?, 1, ?, ?, datetime('now'))
            """, (user['id'], lesson_id, sign_word, 1 if correct else 0, new_acc))

        # Check lesson completion
        lesson = next((l for l in LESSONS if l['id'] == lesson_id), None)
        lesson_complete = False
        if lesson:
            r = cur.execute("""
                SELECT COUNT(*) AS cnt FROM user_progress
                WHERE user_id = ? AND lesson_id = ? AND accuracy >= 70
            """, (user['id'], lesson_id)).fetchone()
            if r and r['cnt'] >= len(lesson['signs']):
                cur.execute("""
                    UPDATE user_progress SET completed = 1
                    WHERE user_id = ? AND lesson_id = ?
                """, (user['id'], lesson_id))
                cur.execute("""
                    INSERT OR IGNORE INTO achievements (user_id, badge_name, badge_icon)
                    VALUES (?, ?, ?)
                """, (user['id'], f"Completed: {lesson['title']}", "🏅"))
                lesson_complete = True

        conn.commit()
        conn.close()
        return jsonify({"success": True, "lesson_complete": lesson_complete})

    except Exception as e:
        print(f"lesson progress error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/leaderboard", methods=["GET"])
def api_leaderboard():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT
                u.username,
                u.avatar_color,
                COUNT(DISTINCT up.sign_word)              AS signs_learned,
                COALESCE(ROUND(AVG(up.accuracy), 1), 0.0) AS avg_accuracy,
                COUNT(DISTINCT a.id)                      AS badges
            FROM users u
            LEFT JOIN user_progress up ON u.id = up.user_id
            LEFT JOIN achievements   a  ON u.id = a.user_id
            GROUP BY u.id, u.username, u.avatar_color
            ORDER BY signs_learned DESC, avg_accuracy DESC
            LIMIT 10
        """).fetchall()
        conn.close()
        return jsonify({"leaderboard": [row_to_dict(r) for r in rows]})
    except Exception as e:
        print(f"leaderboard error: {e}")
        return jsonify({"leaderboard": []})


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SignBridge — AI Sign Language Translator")
    print("  Database: SQLite (built-in, no setup needed)")
    print("=" * 60)
    print(f"\n  DB file  : {DB_PATH}")
    print("  Open     : http://127.0.0.1:5000\n")
    init_db()
    import webbrowser, time
    def _open():
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:5000")
    threading.Thread(target=_open, daemon=True).start()
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)