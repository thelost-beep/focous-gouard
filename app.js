/**
 * FocusGuard Web App Logic
 * -------------------------
 * Uses MediaPipe Face Mesh to track head pose and gaze.
 * Uses TensorFlow.js COCO-SSD to detect cell phones.
 */

const videoElement = document.getElementById('input-video');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusBadge = document.getElementById('status-badge');
const alertOverlay = document.getElementById('alert-overlay');
const alertText = document.getElementById('alert-text');
const focusTimeEl = document.getElementById('focus-time');
const dailyTimeEl = document.getElementById('daily-time');
const bestTimeEl = document.getElementById('best-time');
const distractionCountEl = document.getElementById('distraction-count');
const completedSessionsEl = document.getElementById('completed-sessions');
const globalVisitorsEl = document.getElementById('global-visitors');

const audioA = document.getElementById('audio-a');
const audioB = document.getElementById('audio-b');

const valPitch = document.getElementById('val-pitch');
const valYaw = document.getElementById('val-yaw');
const valGaze = document.getElementById('val-gaze');

// Guide Modal bindings
const guideBtn = document.getElementById('guide-btn');
const guideModal = document.getElementById('guide-modal');
const closeBtn = document.querySelector('.close-btn');

if (guideBtn && guideModal && closeBtn) {
    guideBtn.addEventListener('click', () => {
        guideModal.classList.remove('hidden');
    });
    closeBtn.addEventListener('click', () => {
        guideModal.classList.add('hidden');
    });
    guideModal.addEventListener('click', (e) => {
        if (e.target === guideModal) guideModal.classList.add('hidden');
    });
}

// --- Configuration ---
const THRESHOLDS = {
    PITCH_MAX_DOWN: -20,  // slumping to sleep or phone buried in lap (DISTRACTED)
    PITCH_MIN_DOWN: -2,   // very slight bow / looking mostly straight (FOCUSED)
    PITCH_MAX_UP: 15,     // looking up at ceiling (DISTRACTED)
    YAW_DISTRACTED: 45,   // head turned left/right (degrees)
    GAZE_DOWN: 0.58,      // iris below threshold
    EAR_CLOSED: 0.20,     // eye aspect ratio (slightly reduced to prevent false-positive sleep lock)
    DELAY_MS: 1500,       // 1.5 seconds grace period to allow momentary glances before blasting audio
    SLEEP_DELAY_MS: 1000, // eyes closed time before alarm
    MISSING_DELAY_MS: 4000 // 4 seconds before screaming if user leaves camera
};

// --- State ---
let isGuarding = false;
let isHardcore = false; // Prevents stopping the timer early
let focusState = 'FOCUSED'; // FOCUSED, READING, WARNING, DISTRACTED
let distractionCount = 0;
let focusStartTime = Date.now();
let totalFocusMs = 0;
let remainingMs = 0; // Pomodoro countdown
let lastUpdateTime = Date.now();
let warningStartTime = null;
let eyesClosedStartTime = null;
let missingStartTime = null; // Track when user completely drops from frame
let currentAudio = null;

// Object Detection State
let objectDetector = null;
let phoneDetected = false;
let lastDetectTime = 0;

// Local Storage State
let dailyTotalMs = parseInt(localStorage.getItem('fg_dailyTotalMs')) || 0;
let bestTotalMs = parseInt(localStorage.getItem('fg_bestTotalMs')) || 0;
let completedSessions = parseInt(localStorage.getItem('fg_completedSessions')) || 0;
let lastSavedDate = localStorage.getItem('fg_lastSavedDate') || new Date().toDateString();

// Rolling 7-day Array
let weeklyHistory = JSON.parse(localStorage.getItem('fg_weeklyHistory')) || [];

// --- Initialization ---

// Auto-show guide for new users & Track Visitors
window.addEventListener('DOMContentLoaded', () => {
    const today = new Date().toDateString();

    // Check if it's a new day to reset daily total
    if (today !== lastSavedDate) {
        // Push yesterday's total into the history array before resetting
        const existingEntry = weeklyHistory.find(entry => entry.date === lastSavedDate);
        if (!existingEntry) {
            weeklyHistory.push({ date: lastSavedDate, focusMs: dailyTotalMs });
        } else {
            existingEntry.focusMs = Math.max(existingEntry.focusMs, dailyTotalMs);
        }

        // Keep only last 7 days
        if (weeklyHistory.length > 7) weeklyHistory = weeklyHistory.slice(weeklyHistory.length - 7);
        localStorage.setItem('fg_weeklyHistory', JSON.stringify(weeklyHistory));

        dailyTotalMs = 0;
        completedSessions = 0;
        lastSavedDate = today;
        localStorage.setItem('fg_dailyTotalMs', 0);
        localStorage.setItem('fg_completedSessions', 0);
        localStorage.setItem('fg_lastSavedDate', today);
    }

    // Update initial UI with saved stats
    updateStatsUI(0);

    if (!localStorage.getItem('fg_hasVisited')) {
        if (guideModal) guideModal.classList.remove('hidden');
        localStorage.setItem('fg_hasVisited', 'true');
    }

    // Global Visitor Counter via CountAPI/JSONBlob
    const BLOB_URL = "https://jsonblob.com/api/jsonBlob/1346559385614974976";
    // Using a simple JSONBlob to store { "visitors": X }
    fetch(BLOB_URL)
        .then(res => res.json())
        .then(data => {
            let currentCount = data.visitors || 0;

            // Increment if they haven't visited this session
            if (!sessionStorage.getItem('fg_counted_visit')) {
                currentCount++;
                sessionStorage.setItem('fg_counted_visit', 'true');

                // Save back to blob
                fetch(BLOB_URL, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ visitors: currentCount })
                }).catch(e => console.warn("Failed to update visitor count"));
            }
            if (globalVisitorsEl) globalVisitorsEl.innerText = currentCount;
        })
        .catch(e => {
            console.warn("Could not load visitor count.");
            if (globalVisitorsEl) globalVisitorsEl.innerText = "-";
        });

    // Theme Loading
    const savedTheme = localStorage.getItem('fg_theme');
    if (savedTheme === 'light') {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
    }
});

// Load COCO-SSD
if (typeof cocoSsd !== 'undefined') {
    cocoSsd.load().then(model => {
        objectDetector = model;
        console.log("COCO-SSD loaded for perfect phone detection");
    });
}

// --- Helper Functions ---

function calculatePitch(landmarks) {
    const nose = landmarks[1];
    const chin = landmarks[152];
    const top = landmarks[10];

    const midY = (top.y + chin.y) / 2;
    const verticalSpan = chin.y - top.y;
    const noseOffset = nose.y - midY;

    return (noseOffset / verticalSpan) * -90;
}

function calculateYaw(landmarks) {
    const nose = landmarks[1];
    const leftEyeOuter = landmarks[33];
    const rightEyeOuter = landmarks[263];

    const leftDist = Math.abs(nose.x - leftEyeOuter.x);
    const rightDist = Math.abs(nose.x - rightEyeOuter.x);

    const faceSpan = Math.abs(leftEyeOuter.x - rightEyeOuter.x);
    if (faceSpan === 0) return 0;

    const diff = leftDist - rightDist;
    return (diff / faceSpan) * 90;
}

function calculateGazeRatio(landmarks) {
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

function formatTime(ms) {
    const totalSecs = Math.floor(ms / 1000);
    const mins = Math.floor(totalSecs / 60);
    const secs = totalSecs % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function updateStatsUI(dtMs) {
    if (dtMs > 0) {
        totalFocusMs += dtMs;
        dailyTotalMs += dtMs;

        // Countdown remaining commit time
        remainingMs -= dtMs;
        if (remainingMs <= 0) {
            triggerBreak();
            remainingMs = 0;
            dtMs = 0; // Prevent negatives
        }

        // Check for new best
        if (dailyTotalMs > bestTotalMs) {
            bestTotalMs = dailyTotalMs;
            localStorage.setItem('fg_bestTotalMs', bestTotalMs);
        }

        // Save daily progress periodically
        localStorage.setItem('fg_dailyTotalMs', dailyTotalMs);
        localStorage.setItem('fg_lastSavedDate', new Date().toDateString());

        // Update today's entry in the live array but don't commit it to storage yet (to save ops)
        const today = new Date().toDateString();
        const existingEntry = weeklyHistory.find(entry => entry.date === today);
        if (existingEntry) {
            existingEntry.focusMs = dailyTotalMs;
        } else {
            weeklyHistory.push({ date: today, focusMs: dailyTotalMs });
            if (weeklyHistory.length > 7) weeklyHistory.shift();
        }
        localStorage.setItem('fg_weeklyHistory', JSON.stringify(weeklyHistory));
    }

    // Header tracks current session total (counting UP)
    focusTimeEl.innerText = formatTime(totalFocusMs);
    dailyTimeEl.innerText = formatTime(dailyTotalMs);
    bestTimeEl.innerText = formatTime(bestTotalMs);
    distractionCountEl.innerText = distractionCount;
    completedSessionsEl.innerText = completedSessions;

    // Show remaining time prominently on the badge if focused
    if (focusState === 'FOCUSED' || focusState === 'READING') {
        statusBadge.innerText = `${focusState} ( ${formatTime(remainingMs)} left )`;
    } else {
        statusBadge.innerText = focusState;
    }
}

function updateStats() {
    if (!isGuarding) return;

    const now = Date.now();
    const dt = now - lastUpdateTime;
    lastUpdateTime = now;

    let accrual = 0;
    if (focusState === 'FOCUSED' || focusState === 'READING') {
        accrual = dt;
    }

    updateStatsUI(accrual);
}

function playAlarm() {
    if (!currentAudio || currentAudio.paused) {
        currentAudio = Math.random() > 0.5 ? audioA : audioB;
        currentAudio.play().catch(e => console.log("Audio play failed, user interaction needed."));
        currentAudio.onended = playAlarm;
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

        // Draw minimal mesh (Outline instead of messy web)
        drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, { color: '#00b3cc', lineWidth: 1.5 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_IRIS, { color: '#00ffa3', lineWidth: 1 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_IRIS, { color: '#00ffa3', lineWidth: 1 });

        const pitch = calculatePitch(landmarks);
        const yaw = calculateYaw(landmarks);
        const gaze = calculateGazeRatio(landmarks);
        const ear = calculateEAR(landmarks);

        valPitch.innerText = pitch.toFixed(1);
        valYaw.innerText = Math.abs(yaw).toFixed(1);
        valGaze.innerText = gaze.toFixed(2);

        const isSlumpingDown = pitch < THRESHOLDS.PITCH_MAX_DOWN;
        const isLookingUp = pitch > THRESHOLDS.PITCH_MAX_UP;
        const isBowingToRead = pitch < THRESHOLDS.PITCH_MIN_DOWN && pitch >= THRESHOLDS.PITCH_MAX_DOWN;
        const isHeadTurned = Math.abs(yaw) > THRESHOLDS.YAW_DISTRACTED;
        const isGazeDown = gaze > THRESHOLDS.GAZE_DOWN;
        const isClosed = ear < THRESHOLDS.EAR_CLOSED;

        const now = Date.now();
        missingStartTime = null; // Reset missing timer immediately

        // Reading/writing logic MUST be calculated first
        const isReading = isBowingToRead && !isHeadTurned && !isClosed;

        // General distraction
        const distractedSignal = isHeadTurned || isSlumpingDown || isLookingUp || (isBowingToRead && !isReading && !isGazeDown);

        // Track eyes closed
        if (isClosed) {
            if (!eyesClosedStartTime) eyesClosedStartTime = now;
        } else {
            eyesClosedStartTime = null;
        }

        const isSleeping = eyesClosedStartTime && (now - eyesClosedStartTime > THRESHOLDS.SLEEP_DELAY_MS);

        if (phoneDetected || isSleeping) {
            if (focusState !== 'DISTRACTED') {
                focusState = 'DISTRACTED';
                distractionCount++;
                alertText.innerText = phoneDetected ? "PUT THE PHONE DOWN!" : "WAKE UP!";
            }
            playAlarm();
            warningStartTime = null;
        } else if (distractedSignal) {
            // Keep the general distraction logic fast
            if (focusState === 'FOCUSED' || focusState === 'READING') {
                focusState = 'WARNING';
                warningStartTime = now;
            } else if (focusState === 'WARNING' && warningStartTime) {
                if (now - warningStartTime > THRESHOLDS.DELAY_MS) {
                    focusState = 'DISTRACTED';
                    distractionCount++;
                    alertText.innerText = "HEY! GET BACK TO WORK!";
                    playAlarm();
                }
            } else if (focusState === 'DISTRACTED') {
                // If it was just a warning distraction, keep alarming
                // Only if it wasn't a sleep state that specifically cleared
                if ((alertText.innerText === "WAKE UP!" && !isSleeping) || (alertText.innerText === "PUT THE PHONE DOWN!" && !phoneDetected)) {
                    // Force instant recovery from specific locks
                    focusState = isReading ? 'READING' : 'FOCUSED';
                    warningStartTime = null;
                    eyesClosedStartTime = null;
                    stopAlarm();
                    alertOverlay.classList.add('hidden'); // Guarantee instant visual clear
                } else {
                    playAlarm();
                }
            }
        } else {
            // Fully Recovered (No phone, eyes open, head straight, not distracted)
            focusState = isReading ? 'READING' : 'FOCUSED';
            warningStartTime = null;
            eyesClosedStartTime = null; // Instantly reset sleep timer just in case
            stopAlarm();
            alertOverlay.classList.add('hidden'); // Guarantee instant visual clear
        }

    } else {
        // No face detected
        if (isGuarding) {
            focusState = 'NO USER DETECTED';
            warningStartTime = null;
            eyesClosedStartTime = null;

            if (phoneDetected) {
                alertText.innerText = "PUT THE PHONE DOWN!";
                alertOverlay.classList.remove('hidden');
                playAlarm();
            } else {
                const now = Date.now();
                if (!missingStartTime) missingStartTime = now;

                // Track how long caller has been missing
                if (now - missingStartTime > THRESHOLDS.MISSING_DELAY_MS) {
                    alertText.innerText = "WHERE ARE YOU? GET BACK TO WORK!";
                    alertOverlay.classList.remove('hidden');
                    playAlarm();
                } else {
                    stopAlarm();
                    alertOverlay.classList.add('hidden');
                }
            }
        }
    }
    // --- Update UI ---
    statusBadge.className = `badge ${focusState.replace(/ /g, '-').toLowerCase()}`;
    // statusBadge.innerText = focusState; // Moved to updateStatsUI for dynamic text

    if (focusState === 'DISTRACTED') {
        alertOverlay.classList.remove('hidden');
    } else if (focusState !== 'NO USER DETECTED' || phoneDetected) {
        // The no user detected handles its own UI logic above, but ensure it doesn't accidentally reveal the big red overlay unless phone is found
    }

    canvasCtx.restore();
}

// --- Init MediaPipe ---

const faceMesh = new FaceMesh({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }
});

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
            await faceMesh.send({ image: videoElement });

            // Run Object Detection explicitly at 1 FPS to prevent extreme lagging
            const now = Date.now();
            if (objectDetector && now - lastDetectTime > 1000) {
                lastDetectTime = now;

                // Do not block faceMesh with await if possible, run async
                objectDetector.detect(videoElement).then(predictions => {
                    phoneDetected = predictions.some(p => p.class === 'cell phone' && p.score > 0.35);
                });
            }
        }
    },
    width: 640,
    height: 480
});

// --- Controls ---

// Handle resize
function handleResize() {
    canvasElement.width = videoElement.clientWidth;
    canvasElement.height = videoElement.clientHeight;
}
window.addEventListener('resize', handleResize);
// Periodically check resize just in case the video flexes in the DOM
setInterval(handleResize, 1000);

// --- Event Listeners ---

const pomodoroSetupOverlay = document.getElementById('pomodoro-setup-overlay');
const timeBtns = document.querySelectorAll('.time-btn');
const customTimeInput = document.getElementById('commit-mins-custom');
const hardcoreToggle = document.getElementById('hardcore-toggle');
const commitStartBtn = document.getElementById('commit-start-btn');
const cancelSetupBtn = document.getElementById('cancel-setup-btn');

const breakOverlay = document.getElementById('break-overlay');
const resumeBtn = document.getElementById('resume-btn');

let selectedCommitmentMins = 25;

// Pomodoro Selection Logic
timeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        timeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedCommitmentMins = parseInt(btn.dataset.time);
        customTimeInput.value = ''; // clear custom
    });
});

customTimeInput.addEventListener('input', () => {
    timeBtns.forEach(b => b.classList.remove('active'));
    selectedCommitmentMins = parseInt(customTimeInput.value) || 25;
});

// Setup Modal Triggers
cancelSetupBtn.addEventListener('click', () => {
    pomodoroSetupOverlay.classList.add('hidden');
});

startBtn.addEventListener('click', () => {
    pomodoroSetupOverlay.classList.remove('hidden');
});

function triggerBreak() {
    isGuarding = false;
    stopAlarm();
    camera.stop(); // Stop camera feed
    alertOverlay.classList.add('hidden');

    // Log Completed Session
    completedSessions++;
    localStorage.setItem('fg_completedSessions', completedSessions);
    completedSessionsEl.innerText = completedSessions;

    // UI Updates
    startBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    statusBadge.innerText = "BREAK TIME";
    statusBadge.className = 'badge';
    videoElement.style.border = "4px solid var(--accent-gray)";

    // Show Break Modal
    breakOverlay.classList.remove('hidden');
}

resumeBtn.addEventListener('click', () => {
    breakOverlay.classList.add('hidden');
    pomodoroSetupOverlay.classList.remove('hidden'); // Ask for commit again
});

function startGuarding() {
    remainingMs = selectedCommitmentMins * 60 * 1000;
    isHardcore = hardcoreToggle.checked;

    isGuarding = true;
    totalFocusMs = 0;
    distractionCount = 0;
    focusTimeEl.innerText = "00:00";
    distractionCountEl.innerText = "0";

    pomodoroSetupOverlay.classList.add('hidden');
    startBtn.classList.add('hidden');

    if (!isHardcore) {
        stopBtn.classList.remove('hidden');
    }

    lastUpdateTime = Date.now();
    focusState = 'FOCUSED'; // Reset state
    statusBadge.innerText = "FOCUSED";
    statusBadge.className = 'badge focused';
    videoElement.style.border = "4px solid var(--neon-cyan)";
    breakOverlay.classList.add('hidden');
    missingStartTime = null;

    // Force audio context start
    audioA.play().catch(() => { });
    audioA.pause();

    camera.start();
}

commitStartBtn.addEventListener('click', startGuarding);

stopBtn.addEventListener('click', () => {
    isGuarding = false;
    isHardcore = false;
    startBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    statusBadge.innerText = "STANDBY";
    statusBadge.className = 'badge';
    videoElement.style.border = "4px solid var(--accent-gray)";
    stopAlarm();
    camera.stop(); // Stop camera feed
    alertOverlay.classList.add('hidden');
    breakOverlay.classList.add('hidden');
    pomodoroSetupOverlay.classList.add('hidden');
    missingStartTime = null;
    phoneDetected = false;
});

// Tab/Window Guarding (Anti-Cheating)
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isGuarding) {
        // Punish immediately if they change tabs
        focusState = 'DISTRACTED';
        statusBadge.className = `badge distracted`;
        alertText.innerText = "NO CHEATING! COME BACK TO THIS TAB!";
        alertOverlay.classList.remove('hidden');
        playAlarm();
    }
});

// --- Stats Modal & Charting ---

const statsBtn = document.getElementById('stats-btn');
const statsModal = document.getElementById('stats-modal');
const closeStatsBtn = document.getElementById('close-stats-btn');
let chartInstance = null;

statsBtn.addEventListener('click', () => {
    statsModal.classList.remove('hidden');
    renderChart();
});

closeStatsBtn.addEventListener('click', () => {
    statsModal.classList.add('hidden');
});

// --- Support Modal ---

const supportBtn = document.getElementById('support-btn');
const supportModal = document.getElementById('support-modal');
const closeSupportBtn = document.getElementById('close-support-btn');

supportBtn.addEventListener('click', () => {
    supportModal.classList.remove('hidden');
});

closeSupportBtn.addEventListener('click', () => {
    supportModal.classList.add('hidden');
});

// --- Theme Toggle ---
const themeToggleBtn = document.getElementById('theme-toggle-btn');
themeToggleBtn.addEventListener('click', () => {
    if (document.body.classList.contains('dark-theme')) {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        localStorage.setItem('fg_theme', 'light');
    } else {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        localStorage.setItem('fg_theme', 'dark');
    }
    // Rerender chart to match new theme if open
    if (!statsModal.classList.contains('hidden')) renderChart();
});

function renderChart() {
    const ctx = document.getElementById('weeklyStatsChart').getContext('2d');

    // Ensure today's live data is in the array for the chart
    const today = new Date().toDateString();
    let displayHistory = [...weeklyHistory];

    if (!displayHistory.find(e => e.date === today)) {
        displayHistory.push({ date: today, focusMs: dailyTotalMs });
    }
    if (displayHistory.length > 7) displayHistory = displayHistory.slice(-7);

    const labels = displayHistory.map(entry => {
        // Format to simple day/month (e.g. "Mar 04")
        const d = new Date(entry.date);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    });

    const dataPoints = displayHistory.map(entry => parseFloat((entry.focusMs / (60 * 1000)).toFixed(2))); // converted to minutes

    if (chartInstance) {
        chartInstance.destroy(); // Clear old chart
    }

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Focus Time (Minutes)',
                data: dataPoints,
                backgroundColor: 'rgba(0, 224, 255, 0.7)',
                borderColor: '#00e0ff',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(255,255,255,0.6)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: 'rgba(255,255,255,0.6)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#fff' } }
            }
        }
    });
}
