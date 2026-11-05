
import cv2

# Try camera index 0
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera 0 not found. Trying camera 1...")
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera not detected!")
    exit()

print("Camera connected successfully!")
print("Press Q to close the camera.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Unable to receive camera frame.")
        break

    # Mono camera: convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display original and grayscale camera feeds
    cv2.imshow("Mono Camera - Original", frame)
    cv2.imshow("Mono Camera - Grayscale", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
