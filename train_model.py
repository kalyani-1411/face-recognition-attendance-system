import face_recognition
import os
import pickle

known_faces = []
face_names = []

# Loop through all images in faces folder
for file in os.listdir("faces"):
    img_path = os.path.join("faces", file)
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)

    if encodings:
        known_faces.append(encodings[0])
        face_names.append(file.split("_")[0])  # Extract name from filename

# Save encodings to file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump((known_faces, face_names), f)

print("Face model trained and saved.")
