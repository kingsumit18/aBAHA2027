
import cv2
import math

CAMERA_INDEX = 1

points = []
captured_image = None


def mouse_callback(event, x, y, flags, param):
    global points, captured_image

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) >= 2:
            return

        points.append((x, y))

        print(f"Point {len(points)}: x={x}, y={y}")

        # Draw selected point
        cv2.circle(captured_image, (x, y), 6, (0, 0, 255), -1)

        if len(points) == 2:

            p1, p2 = points

            # Euclidean pixel distance
            distance = math.sqrt(
                (p2[0] - p1[0]) ** 2 +
                (p2[1] - p1[1]) ** 2
            )

            # Draw line between points
            cv2.line(
                captured_image,
                p1,
                p2,
                (0, 255, 0),
                2
            )

            midpoint = (
                (p1[0] + p2[0]) // 2,
                (p1[1] + p2[1]) // 2
            )

            cv2.putText(
                captured_image,
                f"{distance:.2f} pixels",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            print("\n-------------------------")
            print("Point 1:", p1)
            print("Point 2:", p2)
            print(f"Distance: {distance:.2f} pixels")
            print("-------------------------\n")

        cv2.imshow("Pixel Distance", captured_image)


def main():

    global captured_image, points

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera not found. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Live camera started")
    print("Press S to capture image")
    print("Press Q to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        cv2.imshow("Live Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # Capture image
        if key == ord('s'):

            captured_image = frame.copy()
            points = []

            cv2.namedWindow("Pixel Distance")

            cv2.setMouseCallback(
                "Pixel Distance",
                mouse_callback
            )

            cv2.imshow("Pixel Distance", captured_image)

            print("Image captured. Click two points.")

        elif key == ord('r'):
            points = []
            if captured_image is not None:
                captured_image = frame.copy()
                cv2.imshow("Pixel Distance", captured_image)
            print("Points reset")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()