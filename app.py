from flask import Flask, render_template, request, jsonify, Response
import cv2
import face_recognition
import pickle
import numpy as np
from database import mark_attendance, get_attendance_records, delete_attendance_record
import os
from datetime import datetime
import threading
import base64

app = Flask(__name__)

# Global variables for face recognition
known_faces = []
face_names = []
camera = None
is_running = False
marked_attendance = set()

def load_face_encodings():
    global known_faces, face_names
    try:
        with open("face_encodings.pkl", "rb") as f:
            known_faces, face_names = pickle.load(f)
    except:
        known_faces = []
        face_names = []

def save_face_encodings():
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump((known_faces, face_names), f)

def init_camera():
    global camera
    try:
        if camera is not None:
            camera.release()
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = cv2.VideoCapture(1)  # Try secondary camera
            if not camera.isOpened():
                raise Exception("No working camera found")
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
        return False

def detect_liveness(frame, face_location):
    """
    Detect if the face is from a real person or a photo
    Returns: (is_real, confidence)
    """
    top, right, bottom, left = face_location
    face = frame[top:bottom, left:right]
    
    if face.size == 0:
        return False, 0
    
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # 1. Calculate texture features using Local Binary Patterns
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    
    # 2. Calculate image gradients for depth perception
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = gradient_magnitude.mean()
    
    # 3. Calculate standard deviation of pixel intensities
    intensity_std = np.std(gray)
    
    # 4. Calculate face region size ratio
    face_height = bottom - top
    face_width = right - left
    size_ratio = face_height / frame.shape[0]
    
    # Combine multiple factors for liveness detection
    is_real = (
        laplacian_var > 50 and  # Reduced from 100
        gradient_mean > 10 and   # Reduced from 20
        intensity_std > 25 and   # Add intensity variation check
        0.1 < size_ratio < 0.9   # Face should be reasonable size in frame
    )
    
    # Calculate confidence score
    confidence = min(100, (
        (laplacian_var / 100) * 30 +  # Texture contribution
        (gradient_mean / 20) * 30 +    # Gradient contribution
        (intensity_std / 50) * 40      # Intensity variation contribution
    ))
    
    return is_real, confidence

def generate_frames():
    global is_running, marked_attendance, camera
    
    if not init_camera():
        print("Failed to initialize camera")
        yield b''
        return
    
    print("Starting video feed...")
    frame_count = 0
    liveness_counter = {}  # Track consecutive live detections
    
    while is_running:
        try:
            if camera is None or not camera.isOpened():
                print("Camera not available")
                break

            ret, frame = camera.read()
            if not ret or frame is None:
                print("Failed to read frame")
                continue

            # Process every 2nd frame instead of every 3rd
            process_this_frame = frame_count % 2 == 0
            frame_count += 1

            # Resize frame for display (keeping aspect ratio)
            height, width = frame.shape[:2]
            max_dimension = 640
            scale = max_dimension / max(height, width)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            if process_this_frame:
                # Convert to RGB for face recognition
                rgb_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
                
                # Find faces in the frame
                face_locations = face_recognition.face_locations(small_frame)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                        # Scale back the face locations
                        top *= 2
                        right *= 2
                        bottom *= 2
                        left *= 2
                        
                        # Perform liveness detection
                        is_real, liveness_confidence = detect_liveness(frame, (top, right, bottom, left))
                        
                        # Calculate face distances
                        if len(known_faces) > 0:
                            face_distances = face_recognition.face_distance(known_faces, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            best_match_distance = face_distances[best_match_index]
                            
                            name = "Unknown"
                            if best_match_distance < 0.6:
                                name = face_names[best_match_index]
                                
                                # Initialize liveness counter for new faces
                                if name not in liveness_counter:
                                    liveness_counter[name] = 0
                                
                                # Increment liveness counter for real faces
                                if is_real:
                                    liveness_counter[name] += 1
                                    print(f"Liveness counter for {name}: {liveness_counter[name]}")
                                else:
                                    # Decrease counter but don't let it go below 0
                                    liveness_counter[name] = max(0, liveness_counter[name] - 1)
                                
                                # Mark attendance after fewer consistent live detections
                                if liveness_counter[name] >= 5 and name not in marked_attendance:  # Reduced from 10
                                    mark_attendance(name)
                                    marked_attendance.add(name)
                                    print(f"Marked attendance for {name} (Live detection confirmed)")
                                
                                print(f"Recognized: {name} with confidence: {1 - best_match_distance:.2f}")
                                print(f"Liveness confidence: {liveness_confidence:.2f}%")

                            # Draw rectangle & name with liveness indicator
                            if is_real:
                                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                status = f"Live ({liveness_confidence:.0f}%)"
                            else:
                                color = (0, 0, 255)  # Red for non-live detection
                                status = f"Checking... ({liveness_confidence:.0f}%)"
                            
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            cv2.putText(frame, f"{name} - {status}", (left, top - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Convert frame to JPEG with reduced quality for faster transmission
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            break

    if camera is not None:
        camera.release()
        print("Camera released")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/record-attendance')
def record_attendance():
    return render_template('record_attendance.html')

@app.route('/add-student')
def add_student():
    return render_template('add_student.html')

@app.route('/attendance-log')
def attendance_log():
    return render_template('attendance_log.html')

@app.route('/video_feed')
def video_feed():
    global camera, is_running
    
    # Ensure camera is released if not running
    if not is_running and camera is not None:
        camera.release()
        camera = None
    
    if not is_running:
        # Return a blank frame when not running
        blank_frame = np.zeros((480, 640, 3), np.uint8)
        # Add text to indicate camera is stopped
        cv2.putText(blank_frame, "Click 'Start Recognition' to begin", 
                   (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        return Response(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition')
def start_recognition():
    global is_running, marked_attendance, camera
    try:
        if not init_camera():
            return jsonify({"status": "error", "message": "Failed to initialize camera"})
        
        is_running = True
        marked_attendance.clear()
        print("Face recognition started")
        return jsonify({"status": "success", "message": "Face recognition started"})
    except Exception as e:
        print(f"Error starting recognition: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_recognition')
def stop_recognition():
    global is_running, camera
    try:
        is_running = False
        if camera is not None:
            camera.release()
            camera = None
        print("Face recognition stopped")
        return jsonify({"status": "success", "message": "Face recognition stopped"})
    except Exception as e:
        print(f"Error stopping recognition: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/add_person', methods=['POST'])
def add_person():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"status": "error", "message": "No photo uploaded"})
        
        name = data.get('name')
        roll_number = data.get('roll_number')
        
        if not name or not roll_number:
            return jsonify({"status": "error", "message": "Name and roll number are required"})
        
        # Create faces directory if it doesn't exist
        if not os.path.exists('faces'):
            os.makedirs('faces')
        
        # Process the base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"status": "error", "message": "Invalid image data"})
        
        # Save the image
        filename = f"faces/{name}_{roll_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, image)
        
        # Get face encoding
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            os.remove(filename)
            return jsonify({"status": "error", "message": "No face detected in the image"})
        
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        # Add to known faces
        known_faces.append(face_encoding)
        face_names.append(name)
        
        # Save updated encodings
        save_face_encodings()
        
        return jsonify({"status": "success", "message": "Student added successfully"})
        
    except Exception as e:
        print(f"Error adding person: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_attendance')
def get_attendance():
    try:
        records = get_attendance_records()
        return jsonify({"records": records})  # records are already formatted in get_attendance_records()
    except Exception as e:
        print(f"Error getting attendance records: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_registered_people')
def get_registered_people():
    return jsonify({"people": list(set(face_names))})

@app.route('/manage-students')
def manage_students():
    return render_template('manage_students.html')

@app.route('/delete_student', methods=['POST'])
def delete_student():
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"status": "error", "message": "No student name provided"})
        
        name = data['name']
        
        # Find the index of the student in face_names
        if name not in face_names:
            return jsonify({"status": "error", "message": "Student not found"})
        
        # Get all indices where this name appears
        indices = [i for i, x in enumerate(face_names) if x == name]
        
        # Remove the face encodings and names in reverse order
        for index in sorted(indices, reverse=True):
            del known_faces[index]
            del face_names[index]
        
        # Save updated encodings
        save_face_encodings()
        
        # Delete the student's photos
        if os.path.exists('faces'):
            for filename in os.listdir('faces'):
                if filename.startswith(f"{name}_"):
                    try:
                        os.remove(os.path.join('faces', filename))
                    except Exception as e:
                        print(f"Error deleting file {filename}: {str(e)}")
        
        return jsonify({"status": "success", "message": "Student deleted successfully"})
        
    except Exception as e:
        print(f"Error deleting student: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/delete_attendance/<int:record_id>', methods=['DELETE'])
def delete_attendance(record_id):
    try:
        if delete_attendance_record(record_id):
            return jsonify({'success': True, 'message': 'Attendance record deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to delete record'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    load_face_encodings()
    app.run(debug=True)
