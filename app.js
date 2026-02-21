/**
 * FocusGuard Web App Logic
 * -------------------------
 * Uses MediaPipe Face Mesh to track head pose and gaze.
 */

const videoElement = document.getElementById('input-video');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusBadge = document.getElementById('status-badge');
const alertOverlay = document.getElementById('alert-overlay');
const focusTimeEl = document.getElementById('focus-time');
const distractionCountEl = document.getElementById('distraction-count');

const audioA = document.getElementById('audio-a');
const audioB = document.getElementById('audio-b');

// --- Configuration ---
const THRESHOLDS = {
    PITCH_DOWN: -12,      // head tilted forward (degrees)
    GAZE_DOWN: 0.65,     // iris below threshold
    EAR_CLOSED: 0.18,    // eye aspect ratio
    DELAY_MS: 1500,      // time before alarm (ms)
    SLEEP_DELAY_MS: 2000 // eyes closed time before alarm (ms)
};

// --- State ---
let isGuarding = false;
let focusState = 'FOCUSED'; // FOCUSED, WARNING, DISTRACTED
let distractionCount = 0;
let focusStartTime = Date.now();
let lastUpdateTime = Date.now();
let totalFocusMs = 0;
let warningStartTime = null;
let eyesClosedStartTime = null;
let currentAudio = null;

// --- Helper Functions ---

function calculatePitch(landmarks) {
    // MediaPipe face mesh landmarks are normalized (0 to 1)
    // We'll use a simplified pitch estimation based on Y-difference
    // between nose tip and relative eyes/mouth center.
    const nose = landmarks[1];
    const chin = landmarks[152];
    const top = landmarks[10];
    
    // Simple ratio-based pitch (approximate but effective for relative motion)
    const midY = (top.y + chin.y) / 2;
    const verticalSpan = chin.y - top.y;
    const noseOffset = nose.y - midY;
    
    // Convert to "degrees" relative to neutral (approximate)
    return (noseOffset / verticalSpan) * -90; 
}

function calculateGazeRatio(landmarks) {
    // Average vertical position of irises relative to eye lids
    const leftIris = landmarks[468];
    const rightIris = landmarks[473];
    
    const leftTop = landmarks[159].y;
    const leftBottom = landmarks[145].y;
    const rightTop = landmarks[386].y;
    const rightBottom = landmarks[374].y;
    
    const leftRatio = (leftIris.y - leftTop) / (leftBottom - leftTop);
    const rightRatio = (rightIris.y - rightTop) / (rightBottom - rightTop);
    
    return (leftRatio + rightRatio) / 2;
}

function calculateEAR(landmarks) {
    // Eye Aspect Ratio for blink/sleep detection
    function eyeEAR(p1, p2, p3, p4, p5, p6) {
        const v1 = Math.hypot(landmarks[p2].x - landmarks[p6].x, landmarks[p2].y - landmarks[p6].y);
        const v2 = Math.hypot(landmarks[p3].x - landmarks[p5].x, landmarks[p3].y - landmarks[p5].y);
        const h = Math.hypot(landmarks[p1].x - landmarks[p4].x, landmarks[p1].y - landmarks[p4].y);
        return (v1 + v2) / (2.0 * h);
    }
    
    const leftEAR = eyeEAR(33, 160, 158, 133, 153, 144);
    const rightEAR = eyeEAR(362, 387, 385, 263, 380, 373);
    
    return (leftEAR + rightEAR) / 2.0;
}

function updateStats() {
    if (!isGuarding) return;
    
    const now = Date.now();
    const dt = now - lastUpdateTime;
    lastUpdateTime = now;
    
    if (focusState === 'FOCUSED') {
        totalFocusMs += dt;
    }
    
    const totalSecs = Math.floor(totalFocusMs / 1000);
    const mins = Math.floor(totalSecs / 60);
    const secs = totalSecs % 60;
    focusTimeEl.innerText = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    distractionCountEl.innerText = distractionCount;
}

function playAlarm() {
    if (!currentAudio || currentAudio.paused) {
        currentAudio = Math.random() > 0.5 ? audioA : audioB;
        currentAudio.play().catch(e => console.log("Audio play failed, user interaction needed."));
        currentAudio.onended = playAlarm; // loop alternatingly
    }
}

function stopAlarm() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
}

// --- Face Mesh Process ---

function onResults(results) {
    updateStats();
    
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        
        // Draw mesh for feedback
        drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
        drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {color: '#00e0ff'});
        drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {color: '#00e0ff'});
        drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {color: '#00e0ff'});
        
        // Metrics
        const pitch = calculatePitch(landmarks);
        const gaze = calculateGazeRatio(landmarks);
        const ear = calculateEAR(landmarks);
        
        const isHeadDown = pitch < THRESHOLDS.PITCH_DOWN;
        const isGazeDown = gaze > THRESHOLDS.GAZE_DOWN;
        const isClosed = ear < THRESHOLDS.EAR_CLOSED;
        
        const now = Date.now();
        
        // --- Focus Logic ---
        const distractedSignal = isHeadDown || isGazeDown || isClosed;
        
        if (focusState === 'FOCUSED') {
            if (distractedSignal) {
                focusState = 'WARNING';
                warningStartTime = now;
            }
        } else if (focusState === 'WARNING') {
            if (!distractedSignal) {
                focusState = 'FOCUSED';
                warningStartTime = null;
            } else if (now - warningStartTime > THRESHOLDS.DELAY_MS) {
                focusState = 'DISTRACTED';
                distractionCount++;
                playAlarm();
            }
        } else if (focusState === 'DISTRACTED') {
            if (!distractedSignal) {
                focusState = 'FOCUSED';
                stopAlarm();
            }
        }
    } else {
        // No face detected - treat as warning/distraction
        if (isGuarding) {
            if (focusState === 'FOCUSED') {
                focusState = 'WARNING';
                warningStartTime = Date.now();
            } else if (focusState === 'WARNING' && (Date.now() - warningStartTime > THRESHOLDS.DELAY_MS)) {
                focusState = 'DISTRACTED';
                distractionCount++;
                playAlarm();
            }
        }
    }
    
    // --- Update UI ---
    statusBadge.className = `badge ${focusState.toLowerCase()}`;
    statusBadge.innerText = focusState;
    
    if (focusState === 'DISTRACTED') {
        alertOverlay.classList.remove('hidden');
    } else {
        alertOverlay.classList.add('hidden');
    }
    
    canvasCtx.restore();
}

// --- Init MediaPipe ---

const faceMesh = new FaceMesh({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
}});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

faceMesh.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        if (isGuarding) {
            await faceMesh.send({image: videoElement});
        }
    },
    width: 640,
    height: 480
});

// --- Controls ---

startBtn.addEventListener('click', () => {
    isGuarding = true;
    startBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    
    focusStartTime = Date.now();
    lastUpdateTime = Date.now();
    
    // Force audio context start (required by modern browsers)
    audioA.play().catch(() => {});
    audioA.pause();
    
    camera.start();
});

stopBtn.addEventListener('click', () => {
    isGuarding = false;
    startBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    
    stopAlarm();
    camera.stop();
    
    focusState = 'FOCUSED';
    statusBadge.className = 'badge focused';
    statusBadge.innerText = 'FOCUSED';
    alertOverlay.classList.add('hidden');
});

// Handle resize
function handleResize() {
    canvasElement.width = videoElement.clientWidth;
    canvasElement.height = videoElement.clientHeight;
}
window.addEventListener('resize', handleResize);
setTimeout(handleResize, 1000);
