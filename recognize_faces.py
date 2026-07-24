import cv2
import face_recognition
import pickle
import time
import numpy as np
from database import mark_attendance

# Load known face encodings
with open("face_encodings.pkl", "rb") as f:
    known_faces, face_names = pickle.load(f)

cam = cv2.VideoCapture(0)

marked_attendance = set()
last_detected_time = time.time()  # Track the last time a face was detected
TIMEOUT = 10  # Exit after 10 seconds of no new detections
FACE_MATCH_THRESHOLD = 0.6  # Lower threshold means stricter matching
MIN_FACE_SIZE = 30  # Minimum face size in pixels

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    detected_now = False  # Track if at least one new face is detected

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Scale back up face locations since we scaled down the frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Skip if face is too small
        face_height = bottom - top
        face_width = right - left
        if face_height < MIN_FACE_SIZE or face_width < MIN_FACE_SIZE:
            continue

        # Calculate face distances to all known faces
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]
        
        name = "Unknown"
        if best_match_distance < FACE_MATCH_THRESHOLD:
            name = face_names[best_match_index]
            
            # Mark attendance if the student hasn't been marked yet
            if name not in marked_attendance:
                mark_attendance(name)
                marked_attendance.add(name)
                print(f"Attendance marked for {name} (confidence: {1 - best_match_distance:.2f})")
                detected_now = True

        # Draw rectangle & name with confidence
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{name} ({1 - best_match_distance:.2f})" if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Attendance System", frame)

    # Update last detected time if a new face was detected
    if detected_now:
        last_detected_time = time.time()

    # Exit after TIMEOUT seconds of inactivity
    if time.time() - last_detected_time > TIMEOUT:
        print("No new faces detected for 10 seconds. Exiting...")
        break

    # Allow manual exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manual exit triggered. Exiting...")
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
