
import cv2
import time
import torch
import cpu_video_demo as lane

# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDICES = range(10)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

torch.set_num_threads(4)


# ==========================================
# SELECT USB CAMERA
# ==========================================

def select_camera():

    print("\nSearching for connected cameras...")

    for index in CAMERA_INDICES:

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Give camera time to initialize
        time.sleep(0.5)

        ret, frame = cap.read()

        if not ret:
            cap.release()
            continue

        print(f"\nCamera found at index {index}")

        while True:

            preview = frame.copy()

            cv2.putText(
                preview,
                f"CAMERA INDEX: {index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                preview,
                "S = Select | N = Next | Q = Quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("SELECT YOUR USB CAMERA", preview)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                cv2.destroyWindow("SELECT YOUR USB CAMERA")
                print(f"Selected camera index: {index}")
                return cap, index

            elif key == ord('n'):
                break

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None

            ret, frame = cap.read()

            if not ret:
                break

        cap.release()

    cv2.destroyAllWindows()
    return None, None


# ==========================================
# MAIN LIVE LANE DETECTION
# ==========================================

def main():

    print("===================================")
    print(" UFLD V2 LIVE LANE DETECTION")
    print("===================================")

    # Select Logitech C922 camera
    cap, camera_index = select_camera()

    if cap is None:
        print("No camera selected.")
        return

    # Load your existing trained UFLD model
    print("\nLoading existing UFLD V2 model...")

    model = lane.load_model()

    # Reset temporal smoothing
    lane._previous_center = None
    lane._previous_left = None
    lane._previous_right = None

    print("\nLIVE LANE DETECTION STARTED")
    print("Press Q to quit.")
    print("Press R to reset lane smoothing.")

    previous_time = time.time()
    fps_smooth = 0.0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Camera frame not received.")
                break

            # Run existing UFLD lane detection pipeline
            output = lane.process_frame(model, frame)

            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps_smooth = (
                    0.9 * fps_smooth +
                    0.1 * current_fps
                    if fps_smooth > 0
                    else current_fps
                )

            # Display information
            cv2.putText(
                output,
                f"Camera: {camera_index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"FPS: {fps_smooth:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("UFLD V2 - LIVE LANE DETECTION", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                lane._previous_center = None
                lane._previous_left = None
                lane._previous_right = None
                print("Lane smoothing reset.")

    except KeyboardInterrupt:
        print("\nStopping camera...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
