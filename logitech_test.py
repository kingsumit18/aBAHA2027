
import cv2

print("Connecting to Logitech C922...")

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible!")
    exit()

print("Camera connected!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received!")
        break

    cv2.imshow("LOGITECH C922 LIVE CAMERA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
