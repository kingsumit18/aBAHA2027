
import cv2
import numpy as np
import math
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

VIDEO_PATH = Path(
    r"C:\Users\Sumit\Downloads\ABAHA PART1\lka_project\input_video.mp4\workshop_lane_video.mp4"
)

# Later, for Logitech C922:
USE_CAMERA = False
CAMERA_INDEX = 2

# =====================================================
# GLOBAL VARIABLES
# =====================================================

cap = None
current_frame = None
paused = True

selected_plane = "ground"
action = "measure"

calibration_points = []
measurement_points = []

calibration_pending = False

# Separate calibration for ground and object surface
homographies = {
    "ground": None,
    "object": None
}

pixel_distance = None
real_distance = None

# =====================================================
# OPEN VIDEO OR CAMERA
# =====================================================

if USE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
else:
    cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        "Could not open video or camera. Check VIDEO_PATH or CAMERA_INDEX."
    )

ret, current_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read the first video frame.")

paused = True

# =====================================================
# DISTANCE CALCULATION
# =====================================================

def calculate_pixel_distance(p1, p2):
    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


def calculate_real_distance(points, homography):
    points_array = np.array(
        [points],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        points_array,
        homography
    )[0]

    p1 = transformed[0]
    p2 = transformed[1]

    distance = np.linalg.norm(p2 - p1)

    return float(distance)


# =====================================================
# CALIBRATION
# =====================================================

def finish_calibration():
    global calibration_pending
    global homographies
    global calibration_points

    print("\n========== CALIBRATION ==========")

    print("Selected plane:", selected_plane)

    try:
        width_cm = float(
            input("Enter actual rectangle WIDTH in cm: ")
        )

        length_cm = float(
            input("Enter actual rectangle LENGTH in cm: ")
        )

        if width_cm <= 0 or length_cm <= 0:
            raise ValueError("Dimensions must be positive.")

    except ValueError:
        print("Invalid dimensions. Calibration cancelled.")
        calibration_points.clear()
        calibration_pending = False
        return

    source_points = np.array(
        calibration_points,
        dtype=np.float32
    )

    destination_points = np.array(
        [
            [0, 0],
            [width_cm, 0],
            [width_cm, length_cm],
            [0, length_cm]
        ],
        dtype=np.float32
    )

    H, status = cv2.findHomography(
        source_points,
        destination_points
    )

    if H is None:
        print("Calibration failed. Try selecting better points.")
    else:
        homographies[selected_plane] = H

        print(
            selected_plane.upper(),
            "calibration completed successfully."
        )

    calibration_points.clear()
    calibration_pending = False


# =====================================================
# MOUSE CLICK FUNCTION
# =====================================================

def mouse_callback(event, x, y, flags, param):

    global calibration_points
    global measurement_points
    global pixel_distance
    global real_distance
    global calibration_pending

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # ---------------------------------------------
    # CALIBRATION MODE
    # ---------------------------------------------

    if action == "calibrate":

        if len(calibration_points) < 4:

            calibration_points.append((x, y))

            print(
                "Calibration point",
                len(calibration_points),
                ":",
                (x, y)
            )

            if len(calibration_points) == 4:
                calibration_pending = True

    # ---------------------------------------------
    # MEASUREMENT MODE
    # ---------------------------------------------

    elif action == "measure":

        if len(measurement_points) < 2:

            measurement_points.append((x, y))

            print("Measurement point:", (x, y))

        if len(measurement_points) == 2:

            p1 = measurement_points[0]
            p2 = measurement_points[1]

            pixel_distance = calculate_pixel_distance(
                p1, p2
            )

            real_distance = None

            H = homographies[selected_plane]

            if H is not None:
                real_distance = calculate_real_distance(
                    measurement_points,
                    H
                )

            print("\n========== RESULT ==========")

            print(
                f"Pixel distance: {pixel_distance:.2f} px"
            )

            if real_distance is not None:
                print(
                    f"Real distance: {real_distance:.2f} cm"
                )
            else:
                print(
                    "Real distance unavailable. "
                    "Calibrate this plane first."
                )


# =====================================================
# DRAW RESULTS
# =====================================================

def draw_interface(frame):

    output = frame.copy()

    # Calibration points
    for i, point in enumerate(calibration_points):

        cv2.circle(
            output,
            point,
            6,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            output,
            str(i + 1),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # Connect calibration points
    if len(calibration_points) >= 2:

        for i in range(len(calibration_points) - 1):

            cv2.line(
                output,
                calibration_points[i],
                calibration_points[i + 1],
                (0, 255, 255),
                2
            )

    # Measurement points
    for point in measurement_points:

        cv2.circle(
            output,
            point,
            7,
            (0, 255, 0),
            -1
        )

    # Measurement line and result
    if len(measurement_points) == 2:

        p1 = measurement_points[0]
        p2 = measurement_points[1]

        cv2.line(
            output,
            p1,
            p2,
            (255, 0, 0),
            2
        )

        if pixel_distance is not None:

            cv2.putText(
                output,
                f"Pixels: {pixel_distance:.2f} px",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if real_distance is not None:

            cv2.putText(
                output,
                f"Real: {real_distance:.2f} cm",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Information panel
    cv2.putText(
        output,
        f"Plane: {selected_plane.upper()}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    mode_text = (
        "CALIBRATION MODE"
        if action == "calibrate"
        else "MEASUREMENT MODE"
    )

    cv2.putText(
        output,
        mode_text,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    return output


# =====================================================
# MAIN VIDEO LOOP
# =====================================================

window_name = "Two Point Distance Measurement"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("\n========== CONTROLS ==========")
print("SPACE : Play / Pause")
print("G     : Ground calibration")
print("O     : Object surface calibration")
print("M     : Measurement mode")
print("R     : Reset selected points")
print("A     : Previous frame (while paused)")
print("D     : Next frame (while paused)")
print("Q     : Quit")
print("==============================")

while True:

    if not paused:

        ret, new_frame = cap.read()

        if not ret:
            paused = True
        else:
            current_frame = new_frame

    # Ask for dimensions after selecting 4 calibration corners
    if calibration_pending:
        finish_calibration()

    display_frame = draw_interface(current_frame)

    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(30) & 0xFF

    # Play / Pause
    if key == ord(" "):
        paused = not paused

    # Ground calibration
    elif key == ord("g"):

        selected_plane = "ground"
        action = "calibrate"

        calibration_points.clear()
        measurement_points.clear()

        pixel_distance = None
        real_distance = None

        print("\nGROUND CALIBRATION")
        print("Click four ground rectangle corners:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")

    # Object surface calibration
    elif key == ord("o"):

        selected_plane = "object"
        action = "calibrate"

        calibration_points.clear()
        measurement_points.clear()

        pixel_distance = None
        real_distance = None

        print("\nOBJECT SURFACE CALIBRATION")
        print("Click four corners of a known rectangular")
        print("surface on the object, in this order:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")

    # Measurement mode
    elif key == ord("m"):

        action = "measure"

        calibration_points.clear()
        measurement_points.clear()

        pixel_distance = None
        real_distance = None

        print("\nMEASUREMENT MODE")
        print("Selected plane:", selected_plane)
        print("Click your first and second points.")

    # Reset
    elif key == ord("r"):

        calibration_points.clear()
        measurement_points.clear()

        pixel_distance = None
        real_distance = None

        print("Points reset.")

    # Previous frame
    elif key == ord("a") and paused and not USE_CAMERA:

        current_position = cap.get(cv2.CAP_PROP_POS_FRAMES)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            max(0, current_position - 2)
        )

        ret, new_frame = cap.read()

        if ret:
            current_frame = new_frame

    # Next frame
    elif key == ord("d") and paused and not USE_CAMERA:

        ret, new_frame = cap.read()

        if ret:
            current_frame = new_frame

    # Quit
    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print("Program closed.")
