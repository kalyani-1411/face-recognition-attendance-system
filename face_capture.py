import cv2
import os

# Create faces directory if not exists
if not os.path.exists("faces"):
    os.makedirs("faces")

# Start webcam
cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture Face")

name = input("Enter your name: ").strip()
count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)

    # Press 's' to save image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        img_path = f"faces/{name}_{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    # Stop capturing after 5 images
    if count >= 5:
        break

cam.release()
cv2.destroyAllWindows()
print("Face capture complete.")
