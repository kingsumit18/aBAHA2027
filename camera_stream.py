
import cv2

def find_camera(max_index=5):
    """Try camera indexes 0..max_index and return the first one that works."""
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"Camera found at index {i}")
                return cap, i
            cap.release()
    return None, None

def main():
    cap, index = find_camera()
    if cap is None:
        print("No camera found. Check the USB connection and close other apps using the camera.")
        return

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Streaming... Press 'q' to quit, 's' to save a snapshot.")
    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        cv2.imshow("USB Camera Live Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()