# -*- coding: utf-8 -*-
"""
Real-Time Mouth Opening Detector with Webcam
"""

import cv2
import time
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

# Load the face detection and landmark models
face_model = get_face_detector()
landmark_model = get_landmark_model()

# Define outer and inner points of the mouth and initialize distances
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0] * 5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0] * 3

# Variables to store mouth open/close times
open_start_time = None
close_start_time = None
total_open_time = 0
total_close_time = 0
mouth_open = False  # To track current state of the mouth (open or closed)

# Function to check if the mouth is open


def is_mouth_open(shape):
    cnt_outer = cnt_inner = 0
    # Check if outer mouth points exceed calibrated distances
    for i, (p1, p2) in enumerate(outer_points):
        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
            cnt_outer += 1

    # Check if inner mouth points exceed calibrated distances
    for i, (p1, p2) in enumerate(inner_points):
        if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
            cnt_inner += 1

    # If thresholds are exceeded, mouth is open
    return cnt_outer > 3 and cnt_inner > 2


def mouth_opening_detector():
    global open_start_time, close_start_time, total_open_time, total_close_time, mouth_open

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Initial calibration loop to get average mouth distances
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        rects = find_faces(img, face_model)

        # Record distances automatically (without waiting for "r")
        for _ in range(100):  # Collect 100 frames for averaging
            ret, img = cap.read()
            if not ret:
                break
            rects = find_faces(img, face_model)

            # Update mouth distances for each frame
            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                for i, (p1, p2) in enumerate(outer_points):
                    d_outer[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    d_inner[i] += shape[p2][1] - shape[p1][1]
        break

    # Compute average distances
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]
    cv2.destroyAllWindows()

    # Real-time mouth open detection loop
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            # Detect faces in the frame
            rects = find_faces(img, face_model)

            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)

                # Check if the mouth is open based on the landmarks
                if is_mouth_open(shape):
                    if not mouth_open:  # If mouth was previously closed
                        if close_start_time:  # Calculate time for closing
                            total_close_time += time.time() - close_start_time
                        open_start_time = time.time()  # Start timing the open state
                        mouth_open = True
                else:
                    if mouth_open:  # If mouth was previously open
                        if open_start_time:  # Calculate time for opening
                            total_open_time += time.time() - open_start_time
                        close_start_time = time.time()  # Start timing the close state
                        mouth_open = False

            # Display the output (not needed for the final version, can remove)
            cv2.imshow("Mouth Opening Detector", img)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    # Print total time for mouth opening and closing
    print(
        f"Total mouth open time: {time.strftime('%H:%M:%S', time.gmtime(total_open_time))}")
    print(
        f"Total mouth close time: {time.strftime('%H:%M:%S', time.gmtime(total_close_time))}")

    cap.release()
    cv2.destroyAllWindows()


# Run the mouth opening detector
mouth_opening_detector()
