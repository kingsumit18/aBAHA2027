
import cv2

print("Searching for USB cameras...")

camera = None

for i in range(10):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera detected at index: {i}")
            camera = cap
            break

        cap.release()

if camera is None:
    print("No working camera found!")
    exit()

print("USB CAMERA CONNECTED SUCCESSFULLY")
print("Press Q to exit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Frame capture failed")
        break

    cv2.imshow("LIVE USB MONO CAMERA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
