"""
FocusGuard – Real-Time Attention Detection & Audio Alert System
================================================================
Tracks face orientation & eye gaze via webcam.
Detects loss of focus (head down / eyes down for 2+ seconds).
Plays alternating alarm audio files until focus is regained.

Controls:
  q  – Quit
"""

import os
import sys
import time
import math
import numpy as np
import cv2
import mediapipe as mp
import pygame

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Head pose thresholds
HEAD_DOWN_PITCH_THRESHOLD = -15  # degrees – head tilted forward
HEAD_UP_PITCH_THRESHOLD = -5     # degrees – must come back above this to count as up

# Eye gaze thresholds
EYE_DOWN_GAZE_RATIO = 0.62       # iris below this fraction of the eye = looking down
EYE_UP_GAZE_RATIO = 0.55         # must come back above this to count as forward

# Eye aspect ratio for sleep detection
EAR_CLOSED_THRESHOLD = 0.18      # below this = eyes closed
SLEEP_SECONDS = 3.0              # eyes closed this long = sleeping

# Focus logic
DISTRACTION_DELAY_SECONDS = 2.0  # seconds before alarm triggers

# Audio files (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_A = os.path.join(SCRIPT_DIR, "abey-uth-jaa-saale.mp3")
AUDIO_B = os.path.join(SCRIPT_DIR, "uth-jaa-no-censor.mp3")

# Colours (BGR)
COL_GREEN = (0, 220, 80)
COL_YELLOW = (0, 220, 255)
COL_RED = (0, 60, 255)
COL_WHITE = (255, 255, 255)
COL_BLACK = (0, 0, 0)
COL_CYAN = (255, 220, 0)
COL_OVERLAY_BG = (30, 30, 30)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 – FACE MESH INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # enables iris landmarks 468-477
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 – HEAD POSE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
# 3D model points of a canonical face (in arbitrary units)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -63.6, -12.5),       # Chin
    (-43.3, 32.7, -26.0),      # Left eye left corner
    (43.3, 32.7, -26.0),       # Right eye right corner
    (-28.9, -28.9, -24.1),     # Left mouth corner
    (28.9, -28.9, -24.1),      # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe FaceMesh landmark indices
POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

def get_camera_matrix(frame_w, frame_h):
    """Approximate camera intrinsic matrix."""
    focal_length = frame_w
    center = (frame_w / 2.0, frame_h / 2.0)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

def estimate_head_pitch(landmarks, frame_w, frame_h):
    """Return pitch angle in degrees (negative = head tilted forward/down)."""
    image_points = np.array([
        (landmarks[idx].x * frame_w, landmarks[idx].y * frame_h)
        for idx in POSE_LANDMARK_IDS
    ], dtype=np.float64)

    cam_matrix = get_camera_matrix(frame_w, frame_h)
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)
    # Decompose rotation matrix to Euler angles
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-rmat[2, 0], sy)
    else:
        pitch = math.atan2(-rmat[2, 0], sy)
    return math.degrees(pitch)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 – EYE GAZE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Landmark indices for eye boundaries (upper/lower) and iris centre
LEFT_EYE_TOP = [159, 160, 161]
LEFT_EYE_BOTTOM = [144, 145, 153]
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_IRIS_CENTER = 468          # refined iris landmark

RIGHT_EYE_TOP = [386, 387, 388]
RIGHT_EYE_BOTTOM = [373, 374, 380]
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_IRIS_CENTER = 473         # refined iris landmark

# EAR landmarks
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # P1–P6
RIGHT_EYE_EAR = [362, 387, 385, 263, 380, 373]  # P1–P6

def _eye_aspect_ratio(landmarks, indices, w, h):
    """Compute Eye Aspect Ratio for one eye."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    # Vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    # Horizontal distance
    h1 = math.dist(pts[0], pts[3])
    if h1 < 1e-6:
        return 0.3
    return (v1 + v2) / (2.0 * h1)

def compute_ear(landmarks, w, h):
    """Average EAR across both eyes."""
    left = _eye_aspect_ratio(landmarks, LEFT_EYE_EAR, w, h)
    right = _eye_aspect_ratio(landmarks, RIGHT_EYE_EAR, w, h)
    return (left + right) / 2.0

def _vertical_gaze_ratio(landmarks, top_ids, bottom_ids, iris_id, w, h):
    """0.0 = looking up, 1.0 = looking down."""
    top_y = np.mean([landmarks[i].y * h for i in top_ids])
    bottom_y = np.mean([landmarks[i].y * h for i in bottom_ids])
    iris_y = landmarks[iris_id].y * h
    span = bottom_y - top_y
    if span < 1e-6:
        return 0.5
    return (iris_y - top_y) / span

def compute_gaze_ratio(landmarks, w, h):
    """Average vertical gaze ratio across both eyes."""
    left = _vertical_gaze_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                                 LEFT_IRIS_CENTER, w, h)
    right = _vertical_gaze_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                  RIGHT_IRIS_CENTER, w, h)
    return (left + right) / 2.0

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 – FOCUS LOGIC ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class FocusState:
    FOCUSED = "FOCUSED"
    WARNING = "WARNING"
    DISTRACTED = "DISTRACTED"

class FocusEngine:
    def __init__(self, delay=DISTRACTION_DELAY_SECONDS):
        self.state = FocusState.FOCUSED
        self.delay = delay
        self.warning_start = None
        self.total_focus_time = 0.0
        self.focus_start = time.time()
        self.distraction_count = 0
        self._last_update = time.time()

    def update(self, head_down: bool, eyes_down: bool, eyes_closed: bool):
        """Call every frame. Returns current state string."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        distracted_signal = head_down or eyes_down or eyes_closed

        if self.state == FocusState.FOCUSED:
            self.total_focus_time += dt
            if distracted_signal:
                self.state = FocusState.WARNING
                self.warning_start = now

        elif self.state == FocusState.WARNING:
            if not distracted_signal:
                # False alarm – return to focused
                self.state = FocusState.FOCUSED
                self.warning_start = None
            elif (now - self.warning_start) >= self.delay:
                # Persisted long enough → distracted
                self.state = FocusState.DISTRACTED
                self.distraction_count += 1
                self.warning_start = None

        elif self.state == FocusState.DISTRACTED:
            if not distracted_signal:
                # Regained focus
                self.state = FocusState.FOCUSED
                self.focus_start = now

        return self.state

    @property
    def focus_minutes(self):
        return self.total_focus_time / 60.0

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 – AUDIO CONTROL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AudioEngine:
    def __init__(self, audio_a: str, audio_b: str):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

        self.tracks = [audio_a, audio_b]
        self.current_index = 0
        self.alarm_active = False
        self._playing = False  # True while a track is actively playing

        # Validate files
        for f in self.tracks:
            if not os.path.isfile(f):
                print(f"[AudioEngine] WARNING: audio file not found: {f}")

    def start_alarm(self):
        """Begin alternating audio loop from the first track."""
        if self.alarm_active:
            return
        self.alarm_active = True
        self.current_index = 0
        self._play_current()

    def stop_alarm(self):
        """Immediately stop playback."""
        if not self.alarm_active:
            return
        self.alarm_active = False
        self._playing = False
        pygame.mixer.music.stop()

    def pump(self):
        """Must be called every frame to detect track end and queue next."""
        if not self.alarm_active:
            return
        # If we were playing but the mixer is now idle, the track finished
        if self._playing and not pygame.mixer.music.get_busy():
            self.current_index = (self.current_index + 1) % len(self.tracks)
            self._play_current()

    def _play_current(self):
        try:
            pygame.mixer.music.load(self.tracks[self.current_index])
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()
            self._playing = True
        except Exception as e:
            print(f"[AudioEngine] Error playing audio: {e}")
            self._playing = False

    def cleanup(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()

# ─────────────────────────────────────────────────────────────────────────────
# HUD DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, state, pitch, gaze_ratio, ear, focus_engine, fps):
    """Draw a translucent status overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), COL_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Status badge
    if state == FocusState.FOCUSED:
        label = "FOCUSED"
        col = COL_GREEN
        icon = "[OK]"
    elif state == FocusState.WARNING:
        label = "WARNING"
        col = COL_YELLOW
        icon = "[!!]"
    else:
        label = "DISTRACTED"
        col = COL_RED
        icon = "[!!]"

    cv2.putText(frame, f"{icon} {label}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)

    # Metrics line 1
    pitch_str = f"Pitch: {pitch:+.1f} deg"
    gaze_str = f"Gaze: {gaze_ratio:.2f}"
    ear_str = f"EAR: {ear:.2f}"
    cv2.putText(frame, f"{pitch_str}  |  {gaze_str}  |  {ear_str}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)

    # Metrics line 2
    focus_time = focus_engine.total_focus_time
    mins = int(focus_time // 60)
    secs = int(focus_time % 60)
    dist_count = focus_engine.distraction_count
    cv2.putText(frame,
                f"Focus: {mins:02d}:{secs:02d}  |  Distractions: {dist_count}  |  FPS: {fps:.0f}",
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1, cv2.LINE_AA)

    # Pulsing border when distracted
    if state == FocusState.DISTRACTED:
        pulse = int(127 + 127 * math.sin(time.time() * 6))
        border_col = (0, 0, pulse)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_col, 4)
    elif state == FocusState.WARNING:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COL_YELLOW, 2)

def draw_face_contours(frame, landmarks, w, h):
    """Draw minimal face mesh contour lines for visual feedback."""
    # Face oval
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    pts = []
    for idx in FACE_OVAL:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], (100, 200, 100), 1, cv2.LINE_AA)

    # Eyes
    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157,
                        158, 159, 160, 161, 246, 33]
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                         388, 387, 386, 385, 384, 398, 362]
    for contour in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR]:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in contour]
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (200, 200, 100), 1, cv2.LINE_AA)

    # Iris circles
    for iris_center in [468, 473]:
        cx = int(landmarks[iris_center].x * w)
        cy = int(landmarks[iris_center].y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHING HELPER
# ─────────────────────────────────────────────────────────────────────────────
class Smoother:
    """Exponential moving average smoother."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, raw):
        if self.value is None:
            self.value = raw
        else:
            self.value = self.alpha * raw + (1 - self.alpha) * self.value
        return self.value

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  FocusGuard – Attention Detection System")
    print("=" * 50)
    print("  Press 'q' to quit")
    print()

    # Initialise camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    # Initialise engines
    focus_engine = FocusEngine()
    audio_engine = AudioEngine(AUDIO_A, AUDIO_B)

    # Smoothers
    pitch_smoother = Smoother(alpha=0.25)
    gaze_smoother = Smoother(alpha=0.3)
    ear_smoother = Smoother(alpha=0.3)

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_start = time.time()

    # Sleep detection state
    eyes_closed_start = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)  # mirror
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Face mesh detection ──
            results = face_mesh.process(rgb)

            pitch = 0.0
            gaze_ratio = 0.5
            ear = 0.3
            head_down = False
            eyes_down = False
            eyes_closed = False

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Draw face contours
                draw_face_contours(frame, landmarks, w, h)

                # ── Head pose ──
                raw_pitch = estimate_head_pitch(landmarks, w, h)
                pitch = pitch_smoother.update(raw_pitch)
                head_down = pitch < HEAD_DOWN_PITCH_THRESHOLD

                # ── Eye gaze ──
                raw_gaze = compute_gaze_ratio(landmarks, w, h)
                gaze_ratio = gaze_smoother.update(raw_gaze)
                eyes_down = gaze_ratio > EYE_DOWN_GAZE_RATIO

                # ── EAR / sleep ──
                raw_ear = compute_ear(landmarks, w, h)
                ear = ear_smoother.update(raw_ear)
                if ear < EAR_CLOSED_THRESHOLD:
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                    elif (time.time() - eyes_closed_start) >= SLEEP_SECONDS:
                        eyes_closed = True
                else:
                    eyes_closed_start = None

            else:
                # No face – treat as distracted
                head_down = True

            # ── Focus logic ──
            state = focus_engine.update(head_down, eyes_down, eyes_closed)

            # ── Audio control ──
            if state == FocusState.DISTRACTED:
                audio_engine.start_alarm()
            else:
                audio_engine.stop_alarm()
            audio_engine.pump()

            # ── FPS ──
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # ── Draw HUD ──
            draw_hud(frame, state, pitch, gaze_ratio, ear, focus_engine, fps)

            # ── Show ──
            cv2.imshow("FocusGuard", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        audio_engine.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print()
        print("─" * 50)
        mins = int(focus_engine.total_focus_time // 60)
        secs = int(focus_engine.total_focus_time % 60)
        print(f"  Session Summary")
        print(f"  Total Focus Time : {mins:02d}:{secs:02d}")
        print(f"  Distractions     : {focus_engine.distraction_count}")
        print("─" * 50)


if __name__ == "__main__":
    main()
