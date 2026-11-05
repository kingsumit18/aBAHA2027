
import cv2
import numpy as np
from pathlib import Path

# ---------- CONFIGURATION ----------
INPUT_VIDEO =  "harder_challenge_video.mp4"
OUTPUT_VIDEO = "lka_output.mp4"

# ---------- LANE DETECTION ----------
def detect_lane(frame):
    height, width = frame.shape[:2]

    # 1. Convert to grayscale and reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Detect strong edges
    edges = cv2.Canny(blur, 50, 150)

    # 3. Keep the lower region of the image
    #    This assumes the road is in the lower half.
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.40), int(height * 0.55)),
        (int(width * 0.60), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    road_edges = cv2.bitwise_and(edges, mask)

    # 4. Detect candidate line segments
    segments = cv2.HoughLinesP(
        road_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=25,
        maxLineGap=60
    )

    left_lines = []
    right_lines = []

    if segments is not None:
        for segment in segments:
            x1, y1, x2, y2 = segment[0]

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / dx

            # Reject nearly horizontal segments
            if abs(slope) < 0.35:
                continue

            # Approximate left/right using line direction
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    def fit_lane_line(lines, y_bottom, y_top):
        if not lines:
            return None

        points = []
        for x1, y1, x2, y2 in lines:
            points.extend([(x1, y1), (x2, y2)])

        pts = np.array(points, dtype=np.float32)
        if len(pts) < 2:
            return None

        # Fit x as a function of y
        try:
            a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
        except (ValueError, np.linalg.LinAlgError):
            return None

        x_bottom = int(a * y_bottom + b)
        x_top = int(a * y_top + b)

        return (x_bottom, y_bottom, x_top, y_top)

    y_bottom = height - 1
    y_top = int(height * 0.60)

    left = fit_lane_line(left_lines, y_bottom, y_top)
    right = fit_lane_line(right_lines, y_bottom, y_top)

    result = frame.copy()

    # Draw detected candidate lane boundaries
    if left is not None:
        cv2.line(
            result, (left[0], left[1]), (left[2], left[3]),
            (0, 255, 0), 4
        )

    if right is not None:
        cv2.line(
            result, (right[0], right[1]), (right[2], right[3]),
            (0, 255, 0), 4
        )

    # 5. Estimate apparent lane-center offset
    offset_px = None

    if left is not None and right is not None:
        lane_center = (left[0] + right[0]) / 2
        image_center = width / 2
        offset_px = lane_center - image_center

        cv2.line(
            result,
            (int(lane_center), y_bottom),
            (int(lane_center), y_top),
            (255, 0, 0), 3
        )

        cv2.line(
            result,
            (int(image_center), y_bottom),
            (int(image_center), y_top),
            (0, 255, 255), 2
        )

        status = (
            "Lane center RIGHT of image center"
            if offset_px > 0
            else "Lane center LEFT of image center"
        )

        cv2.putText(
            result, f"Offset: {offset_px:.1f} px",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        cv2.putText(
            result, status,
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
    else:
        cv2.putText(
            result, "LANE UNCERTAIN - NO CONTROL",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )

    return result, offset_px


# ---------- VIDEO PROCESSING ----------
def main():
    if not Path(INPUT_VIDEO).is_file():
        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}. "
            "Place the video beside this Python file."
        )

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output_frame, offset = detect_lane(frame)
            writer.write(output_frame)

            cv2.imshow("LKA - Video Prototype", output_frame)

            # Press q to stop processing early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Finished. Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
