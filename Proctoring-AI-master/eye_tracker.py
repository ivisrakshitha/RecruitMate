# -*- coding: utf-8 -*-
"""
Real-Time Eye Tracking and Position Detection with Webcam
"""

import cv2
import numpy as np
import time
import warnings
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

# Suppress OpenCV warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")


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


def track_eye():
    # Initialize variables for tracking time in each direction
    start_time = time.time()
    time_left, time_right, time_up, time_forward = 0, 0, 0, 0
    last_position = 0  # Track the last detected position
    last_time = start_time

    # Set up face and landmark detection
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    left_eye_points = [36, 37, 38, 39, 40, 41]
    right_eye_points = [42, 43, 44, 45, 46, 47]

    cap = cv2.VideoCapture(0)  # Open the default webcam

    try:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            rects = find_faces(img, face_model)

            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask, end_points_left = eye_on_mask(
                    mask, left_eye_points, shape)
                mask, end_points_right = eye_on_mask(
                    mask, right_eye_points, shape)
                kernel = np.ones((9, 9), np.uint8)
                mask = cv2.dilate(mask, kernel, 5)

                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(
                    eyes_gray, 75, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)

                # Detect eyeball position for each eye
                eyeball_pos_left = contouring(
                    thresh[:, 0:mid], mid, img, end_points_left)
                eyeball_pos_right = contouring(
                    thresh[:, mid:], mid, img, end_points_right, True)

                # Choose the detected position if both eyes agree
                if eyeball_pos_left == eyeball_pos_right:
                    current_position = eyeball_pos_left
                else:
                    current_position = 0  # Default to looking forward if uncertain

                # Update time for each direction
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

                # Update last position and time
                last_position = current_position
                last_time = current_time

            # Display the live camera feed only
            cv2.imshow('eyes', img)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping eye tracking...")

    finally:
        # Release and close
        cap.release()
        cv2.destroyAllWindows()

        # Convert seconds to minutes and seconds for each direction
        def convert_seconds_to_min_sec(seconds):
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes} minutes {seconds:.2f} seconds"

        # Print recorded time in console
        print(f"Time looking left: {convert_seconds_to_min_sec(time_left)}")
        print(f"Time looking right: {convert_seconds_to_min_sec(time_right)}")
        print(f"Time looking up: {convert_seconds_to_min_sec(time_up)}")
        print(
            f"Time looking forward: {convert_seconds_to_min_sec(time_forward)}")


# Run the eye tracking function
track_eye()
