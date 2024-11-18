import cv2
import numpy as np
import time

# Initialize face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


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


def detect_head_pose():
    """Detect and display head pose from a webcam feed."""
    # Try accessing the webcam using index 0 (default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access any camera.")
        return

    head_up = head_down = head_left = head_right = 0
    frame_count = 0  # Count number of frames processed

    try:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Camera matrix based on image dimensions
            camera_matrix = get_camera_matrix(img)
            if camera_matrix is None:
                print("Error: Could not calculate the camera matrix")
                break

            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Update frame count
            frame_count += 1

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Draw rectangle around detected face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Track head pose based on face position
                    # Face is towards the top half (head up)
                    if y < img.shape[0] / 2:
                        head_up += 1
                    else:  # Face is towards the bottom half (head down)
                        head_down += 1

                    # Face is on the left half (head left)
                    if x < img.shape[1] / 2:
                        head_left += 1
                    else:  # Face is on the right half (head right)
                        head_right += 1

            # Display the resulting frame
            cv2.imshow('Head Pose Estimation - Camera Feed', img)

            # Add a small delay between frames to reduce CPU usage
            time.sleep(0.1)

            # Wait for the user to press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Handle graceful exit when keyboard interrupt occurs
        print("\nProgram interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Output the results after camera is closed
        print(f"Head Up: {head_up}")
        print(f"Head Down: {head_down}")
        print(f"Head Left: {head_left}")
        print(f"Head Right: {head_right}")
        print(f"Total frames processed: {frame_count}")


# Run head pose detection from the webcam
detect_head_pose()
