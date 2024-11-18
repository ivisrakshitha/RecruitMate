# -- coding: utf-8 --
"""
Real-Time Eye Tracking, Person, Phone, Mouth Opening, and Head Pose Detection with Webcam
"""

import cv2
import numpy as np
import time
import warnings
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

# Suppress OpenCV warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")

# YOLO model file paths for Person and Phone detection
weights_path = "models/yolov3.weights"
cfg_path = "models/yolov3.cfg"
names_path = "models/coco.names"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load face detection and landmark models for mouth and eye detection
face_model = get_face_detector()
landmark_model = get_landmark_model()

# Initialize face detector for head pose detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define mouth, eye, and head pose tracking variables
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0] * 5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0] * 3
left_eye_points = [36, 37, 38, 39, 40, 41]
right_eye_points = [42, 43, 44, 45, 46, 47]

# Initialize tracking and detection variables
person_detected_time = phone_detected_time = both_detected_time = no_detection_time = 0
open_start_time = close_start_time = None
total_open_time = total_close_time = 0
mouth_open = False
time_left, time_right, time_up, time_forward = 0, 0, 0, 0
last_position = 0
last_time = time.time()
head_up = head_down = head_left = head_right = 0
frame_count = 0


def get_camera_matrix(img):
    """Creates the camera matrix based on the image size."""
    if img is None:
        print("Error: The camera feed is empty!")
        return None
    size = img.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    return np.array([[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double")


def is_mouth_open(shape):
    cnt_outer = cnt_inner = 0
    for i, (p1, p2) in enumerate(outer_points):
        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
            cnt_outer += 1
    for i, (p1, p2) in enumerate(inner_points):
        if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
            cnt_inner += 1
    return cnt_outer > 3 and cnt_inner > 2


def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]


def find_eyeball_position(end_points, cx, cy):
    x_ratio = (end_points[0] - cx) / (cx - end_points[2])
    y_ratio = (cy - end_points[1]) / (end_points[3] - cy)
    if x_ratio > 3:
        return 1  # Looking left
    elif x_ratio < 0.33:
        return 2  # Looking right
    elif y_ratio < 0.33:
        return 3  # Looking up
    else:
        return 0  # Looking forward


def contouring(thresh, mid, img, end_points, right=False):
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass


def process_thresh(thresh):
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def track_eye(img, rects):
    global last_position, last_time, time_left, time_right, time_up, time_forward
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        if shape is None:
            continue

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left_eye_points, shape)
        mask, end_points_right = eye_on_mask(mask, right_eye_points, shape)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, 5)

        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)

        eyeball_pos_left = contouring(
            thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(
            thresh[:, mid:], mid, img, end_points_right, True)

        current_position = eyeball_pos_left if eyeball_pos_left == eyeball_pos_right else 0
        current_time = time.time()
        elapsed = current_time - last_time
        if last_position == 1:
            time_left += elapsed
        elif last_position == 2:
            time_right += elapsed
        elif last_position == 3:
            time_up += elapsed
        else:
            time_forward += elapsed

        last_position = current_position
        last_time = current_time


def detect_objects_and_mouth(frame, start_time):
    global person_detected_time, phone_detected_time, both_detected_time, no_detection_time
    global open_start_time, close_start_time, total_open_time, total_close_time, mouth_open

    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    person_detected = phone_detected = False
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                if class_id == 0:
                    person_detected = True
                elif class_id == 67:
                    phone_detected = True

    current_time = time.time() - start_time
    if person_detected and phone_detected:
        both_detected_time += current_time
    elif person_detected:
        person_detected_time += current_time
    elif phone_detected:
        phone_detected_time += current_time
    else:
        no_detection_time += current_time
    start_time = time.time()

    rects = find_faces(frame, face_model)
    for rect in rects:
        shape = detect_marks(frame, landmark_model, rect)
        if shape is None:
            continue

        if is_mouth_open(shape):
            if not mouth_open:
                if close_start_time:
                    total_close_time += time.time() - close_start_time
                open_start_time = time.time()
                mouth_open = True
        else:
            if mouth_open:
                if open_start_time:
                    total_open_time += time.time() - open_start_time
                close_start_time = time.time()
                mouth_open = False

    return start_time


cap = cv2.VideoCapture(0)
start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            start_time = detect_objects_and_mouth(frame, start_time)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            track_eye(frame, find_faces(frame, face_model))

        for (x, y, w, h) in faces:
            if y < frame.shape[0] / 2:
                head_up += 1
            else:
                head_down += 1
            if x < frame.shape[1] / 2:
                head_left += 1
            else:
                head_right += 1

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Person detected time:", person_detected_time)
    print("Phone detected time:", phone_detected_time)
    print("Both detected time:", both_detected_time)
    print("No detection time:", no_detection_time)
    print("Total mouth open time:", total_open_time)
    print("Total mouth closed time:", total_close_time)
    print("Time looking left:", time_left)
    print("Time looking right:", time_right)
    print("Time looking up:", time_up)
    print("Time looking forward:", time_forward)
    print("Head Up:", head_up)
    print("Head Down:", head_down)
    print("Head Left:", head_left)
    print("Head Right:", head_right)
    print("Total frames processed:", frame_count)
