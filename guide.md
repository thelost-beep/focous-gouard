# FocusGuard Architecture Guide

Here is a breakdown of how the FocusGuard project is built in both Python and Web versions and the specific camera and AI tools used.

## The Core AI Model
Both the Python and Web versions of the project rely on the exact same core AI engine to track your focus: **Google's MediaPipe Face Mesh**. It precisely maps 468 3D facial landmarks (plus refined iris landmarks for your eye gaze) to detect if your head is tilted down, your eyes are looking away, or if your eyes are closed.

## 1. The Python Version (`focusguard.py`)
This is a native desktop script running entirely on your local machine.

*   **Camera Extension/Library used**: **OpenCV (`cv2`)**. 
    *   It uses `cv2.VideoCapture(0)` to directly plug into your computer's built-in webcam. OpenCV handles all the heavy lifting of reading the video stream frame-by-frame.
*   **MediaPipe Integration**: It uses the official `mediapipe` Python package (`mp.solutions.face_mesh`) to process those OpenCV frames and find where you are looking.
*   **Audio Alerts**: It uses the `pygame.mixer` to stream the `mp3` audio alerts natively when a distraction is detected.

## 2. The Web Version (`index.html` & `app.js`)
We rebuilt the logic so that it can run entirely inside a browser, making it incredibly accessible with zero installation required. It doesn't send your video to any backend server; it runs 100% locally in your browser to protect your privacy.

*   **Camera Extension/Library used**: **HTML5 and `@mediapipe/camera_utils`**.
    *   Instead of OpenCV, it simply uses the browser's built-in `<video id="input-video" playsinline>` element to get your webcam feed. 
    *   We use the `@mediapipe/camera_utils` helper script (loaded via CDN) to easily pipe the webcam feed from the HTML5 video player into the Face Mesh model.
*   **MediaPipe Integration**: It uses the JavaScript version of the Face Mesh model (`@mediapipe/face_mesh/face_mesh.js`), loaded via CDN, to map your facial landmarks in JavaScript. We wrote custom functions in `app.js` (`calculatePitch`, `calculateGazeRatio`, `calculateEAR`) to mimic the math we were doing in Python.
*   **Audio Alerts**: Instead of Pygame, it utilizes standard HTML5 `<audio>` elements controlled by JavaScript (`currentAudio.play()`) to queue and play the warning sounds.

**In summary:** We ported the exact same math—calculating Head Pitch, Eye Aspect Ratio (EAR), and Vertical Gaze Ratio—from Python to JavaScript. While the Python version uses **OpenCV (`cv2`)** to grab your camera feed, the web version uses standard **HTML5 Video elements (`<video>`)** managed by MediaPipe's own `camera_utils.js`.
