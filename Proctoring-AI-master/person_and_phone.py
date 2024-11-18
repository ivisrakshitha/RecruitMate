import cv2
import numpy as np
import time

# Ensure you have the correct file paths for the YOLO model files
weights_path = "models/yolov3.weights"
cfg_path = "models/yolov3.cfg"
names_path = "models/coco.names"  # This file contains the class names

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Variables to track detection times
start_time = time.time()
person_detected_time = 0
phone_detected_time = 0
both_detected_time = 0
no_detection_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Run detection and process outputs
        detections = net.forward(output_layers)

        # Variables to track detection status
        person_detected = False
        phone_detected = False

        # Process detections
        for out in detections:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter detections based on confidence threshold
                if confidence > 0.5:
                    if class_id == 0:  # Class ID 0 is 'person' in COCO dataset
                        person_detected = True
                    elif class_id == 67:  # Class ID 67 is 'cell phone' in COCO dataset
                        phone_detected = True

        # Update detection times based on the results
        current_time = time.time() - start_time
        if person_detected and phone_detected:
            both_detected_time += current_time
        elif person_detected and not phone_detected:
            person_detected_time += current_time
        elif not person_detected and phone_detected:
            phone_detected_time += current_time
        else:
            no_detection_time += current_time

        # Reset the start time for the next frame
        start_time = time.time()

        # Display the frame
        cv2.imshow("Detection", frame)

        # Wait for user to press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Exiting...")

finally:
    # Output the total times for each detection
    print(
        f"Total time with person detected: {time.strftime('%H:%M:%S', time.gmtime(person_detected_time))}")
    print(
        f"Total time with phone detected: {time.strftime('%H:%M:%S', time.gmtime(phone_detected_time))}")
    print(
        f"Total time with both detected: {time.strftime('%H:%M:%S', time.gmtime(both_detected_time))}")
    print(
        f"Total time with no detection: {time.strftime('%H:%M:%S', time.gmtime(no_detection_time))}")

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
