import numpy as np
import cv2
import math
from collections import deque, Counter
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 130)  # Speech rate

# Start video capture from webcam
capture = cv2.VideoCapture(0)

# Gesture stability settings
history = deque(maxlen=10)
last_spoken = None  # Track last spoken gesture

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    dynamic_crop_used = False  # Flag to switch between dynamic and fallback ROI
    crop_image = None

    # Preprocess full frame for contour detection
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros_like(frame)
    gesture_text = ""

    try:
        # Use largest contour (likely hand)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 5000:
            raise Exception("Too small to be a hand")

        x, y, w, h = cv2.boundingRect(contour)
        padding = 30
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, frame_width)
        y2 = min(y + h + padding, frame_height)
        crop_image = frame[y1:y2, x1:x2]
        dynamic_crop_used = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Redetect inside cropped image for accuracy
        blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.dist(end, start)
            b = math.dist(far, start)
            c = math.dist(end, far)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * (180 / math.pi)

            if angle <= 90 and d > 10000:
                count_defects += 1
                cv2.circle(crop_image, far, 5, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        history.append(count_defects)
        stable_defects = Counter(history).most_common(1)[0][0]

        gesture_map = {
            0: "One",
            1: "Two",
            2: "Three",
            3: "Four",
            4: "Five"
        }
        gesture_text = gesture_map.get(stable_defects, "")

        if gesture_text and gesture_text != last_spoken:
            engine.say(gesture_text)
            engine.runAndWait()
            last_spoken = gesture_text

    except:
        # Fallback ROI if no hand is detected
        x1, y1, x2, y2 = 100, 100, 400, 400
        crop_image = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display output
    if crop_image is not None:
        cv2.imshow("Crop", crop_image)

    cv2.putText(frame, gesture_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
