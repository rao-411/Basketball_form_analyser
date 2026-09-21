"""
shooting_form_analyzer.py

Analyzes basketball shooting form AND shot outcome from a video,
using YOLOv8-Pose (body keypoints) and YOLOv8 (ball detection).

On first run, this window will ask you to draw a box around the
hoop/rim in the first frame — click and drag, then press ENTER
(or SPACE) to confirm.

Usage:
    python shooting_form_analyzer.py input_video.mp4 [left|right]
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8-Pose keypoint indices (COCO format, 17 keypoints)
KEYPOINTS = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

ARM_BONES = [(5, 7), (7, 9), (6, 8), (8, 10)]
TORSO_BONES = [(5, 6), (5, 11), (6, 12), (11, 12)]
LEG_BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]

COLOR_ARM = (60, 200, 255)
COLOR_TORSO = (255, 255, 255)
COLOR_LEG = (255, 180, 60)
COLOR_JOINT = (40, 220, 40)
COLOR_GOOD = (80, 220, 80)
COLOR_WARN = (60, 180, 255)
COLOR_BALL = (0, 140, 255)
COLOR_HOOP = (255, 0, 255)

ELBOW_GOOD_MIN = 160
KNEE_BEND_GOOD_MIN = 15

COCO_SPORTS_BALL_CLASS = 32  # YOLOv8's default model class index for "sports ball"


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def draw_translucent_panel(frame, x, y, w, h, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_text_with_shadow(frame, text, pos, color, scale=0.5, thickness=1):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(frame, kpts):
    def valid(idx):
        return kpts[idx][0] > 0 and kpts[idx][1] > 0

    def line(a, b, color):
        if valid(a) and valid(b):
            pa = tuple(kpts[a].astype(int))
            pb = tuple(kpts[b].astype(int))
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

    for a, b in TORSO_BONES:
        line(a, b, COLOR_TORSO)
    for a, b in LEG_BONES:
        line(a, b, COLOR_LEG)
    for a, b in ARM_BONES:
        line(a, b, COLOR_ARM)

    for x, y in kpts:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, COLOR_JOINT, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), 1, cv2.LINE_AA)


def get_hoop_roi(first_frame):
    print("\nDraw a box around the hoop/rim, then press ENTER or SPACE to confirm.")
    roi = cv2.selectROI("Select hoop location", first_frame, showCrosshair=True)
    cv2.destroyWindow("Select hoop location")
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No hoop region selected — make/miss detection will be skipped.")
        return None
    return (x, y, x + w, y + h)  # (x1, y1, x2, y2)


def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def fit_launch_angle(trajectory):
    """Estimate launch angle (degrees above horizontal) using the earliest
    contiguous stretch of ball detections (the release phase)."""
    if len(trajectory) < 4:
        return None

    # Take points from the first detection onward, but only while frame gaps stay small
    # (a big gap means the ball was lost and re-detected somewhere unrelated)
    early_points = [trajectory[0]]
    for i in range(1, len(trajectory)):
        prev_frame = early_points[-1][0]
        curr_frame, curr_pt = trajectory[i]
        if curr_frame - prev_frame > 5:
            break
        early_points.append((curr_frame, curr_pt))
        if len(early_points) >= 6:
            break

    if len(early_points) < 4:
        return None

    xs = np.array([p[1][0] for p in early_points])
    ys = np.array([p[1][1] for p in early_points])
    coeffs = np.polyfit(xs, ys, 1)
    slope = coeffs[0]
    angle = np.degrees(np.arctan(-slope))
    return abs(angle)


def analyze_video(input_path, output_path="output_annotated.mp4", shooting_side="right"):
    pose_model = YOLO("yolov8n-pose.pt")
    ball_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        print("Error: could not read the first frame.")
        return
    hoop_box = get_hoop_roi(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    shoulder_idx = KEYPOINTS[f"{shooting_side}_shoulder"]
    elbow_idx = KEYPOINTS[f"{shooting_side}_elbow"]
    wrist_idx = KEYPOINTS[f"{shooting_side}_wrist"]
    hip_idx = KEYPOINTS[f"{shooting_side}_hip"]
    knee_idx = KEYPOINTS[f"{shooting_side}_knee"]
    ankle_idx = KEYPOINTS[f"{shooting_side}_ankle"]

    elbow_angles = []
    knee_angles = []
    ball_trajectory = []  # list of (x, y) centers, in frame order
    ball_frames_detected = 0
    entered_hoop = False
    outcome = None  # "MAKE", "MISS", or None if undetermined
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # --- Pose detection ---
        pose_results = pose_model(frame, verbose=False)
        elbow_angle = None
        knee_angle = None

        if pose_results and pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
            kpts = pose_results[0].keypoints.xy[0].cpu().numpy()
            shoulder, elbow, wrist = kpts[shoulder_idx], kpts[elbow_idx], kpts[wrist_idx]
            hip, knee, ankle = kpts[hip_idx], kpts[knee_idx], kpts[ankle_idx]

            draw_skeleton(frame, kpts)

            if not (np.all(shoulder == 0) or np.all(elbow == 0) or np.all(wrist == 0)):
                angle_candidate = calculate_angle(shoulder, elbow, wrist)
                if 5 < angle_candidate < 179:  # filter out likely tracking glitches
                    elbow_angle = angle_candidate
                    elbow_angles.append(elbow_angle)

            if not (np.all(hip == 0) or np.all(knee == 0) or np.all(ankle == 0)):
                angle_candidate = calculate_angle(hip, knee, ankle)
                if 5 < angle_candidate < 179:
                    knee_angle = angle_candidate
                    knee_angles.append(knee_angle)

        # --- Ball detection (larger inference size helps catch small/blurry balls) ---
        ball_results = ball_model(frame, verbose=False, classes=[COCO_SPORTS_BALL_CLASS], conf=0.12, imgsz=960)
        ball_center = None
        ball_detections_total = 0
        if ball_results and len(ball_results[0].boxes) > 0:
            box = ball_results[0].boxes[0]  # highest-confidence detection
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            candidate = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Reject implausible jumps (likely a false detection on a different round object)
            accept = True
            if ball_trajectory:
                last_frame_num, (lx, ly) = ball_trajectory[-1]
                frame_gap = frame_num - last_frame_num
                dist = np.hypot(candidate[0] - lx, candidate[1] - ly)
                max_plausible_dist = 0.15 * width * max(frame_gap, 1)
                if dist > max_plausible_dist:
                    accept = False

            if accept:
                ball_center = candidate
                cv2.circle(frame, ball_center, 10, COLOR_BALL, 3, cv2.LINE_AA)
                ball_trajectory.append((frame_num, ball_center))
                ball_frames_detected += 1

        # Draw ball trajectory as a visible line
        traj_points = [pt for _, pt in ball_trajectory]
        for i in range(1, len(traj_points)):
            cv2.line(frame, traj_points[i - 1], traj_points[i], COLOR_BALL, 2, cv2.LINE_AA)

        # --- Make/miss detection ---
        if hoop_box is not None and ball_center is not None:
            cv2.rectangle(frame, (hoop_box[0], hoop_box[1]), (hoop_box[2], hoop_box[3]), COLOR_HOOP, 2)

            if outcome is None:
                in_hoop_now = point_in_box(ball_center[0], ball_center[1], hoop_box)
                if in_hoop_now:
                    entered_hoop = True
                elif entered_hoop and ball_center[1] > hoop_box[3]:
                    # Ball was in the hoop box and is now below it -> went through
                    outcome = "MAKE"
                elif entered_hoop and ball_center[1] < hoop_box[1] - 20:
                    # Ball bounced back up above the rim -> rejected
                    outcome = "MISS"

        # Stats panel (bottom-left, translucent, smaller)
        panel_y = height - 55
        draw_translucent_panel(frame, 12, panel_y, 110, 45)
        if elbow_angle is not None:
            color = COLOR_GOOD if elbow_angle >= ELBOW_GOOD_MIN else COLOR_WARN
            draw_text_with_shadow(frame, f"Elbow {elbow_angle:5.1f}", (20, panel_y + 18), color, scale=0.4)
        else:
            draw_text_with_shadow(frame, "Elbow --", (20, panel_y + 18), (150, 150, 150), scale=0.4)

        if knee_angle is not None:
            bend = 180 - knee_angle
            color = COLOR_GOOD if bend >= KNEE_BEND_GOOD_MIN else COLOR_WARN
            draw_text_with_shadow(frame, f"Knee  {bend:5.1f}", (20, panel_y + 35), color, scale=0.4)
        else:
            draw_text_with_shadow(frame, "Knee  --", (20, panel_y + 35), (150, 150, 150), scale=0.4)

        if outcome is not None:
            color = (80, 220, 80) if outcome == "MAKE" else (60, 60, 220)
            draw_text_with_shadow(frame, outcome, (width // 2 - 40, 40), color, scale=0.9, thickness=2)

        out.write(frame)

    cap.release()
    out.release()

    print(f"\nProcessed {frame_num} frames. Annotated video saved to '{output_path}'.")
    print(f"Ball detected in {ball_frames_detected}/{frame_num} frames.")
    if ball_frames_detected == 0:
        print("The ball was never detected. Try: better lighting/contrast, a ball that stands out from the background, or filming closer/more zoomed-in.")

    if elbow_angles:
        print(f"\n--- Shot Summary ---")
        print(f"Max elbow extension: {max(elbow_angles):.1f} degrees")
        print(f"Min elbow angle (set point): {min(elbow_angles):.1f} degrees")
        if knee_angles:
            print(f"Max knee bend: {180 - min(knee_angles):.1f} degrees of flex")

        if max(elbow_angles) < ELBOW_GOOD_MIN:
            print("Note: elbow may not be fully extending on release (check follow-through).")
        if knee_angles and (180 - min(knee_angles)) < KNEE_BEND_GOOD_MIN:
            print("Note: minimal knee bend detected (less power generation from legs).")
    else:
        print("No pose detected in this video. Try a clearer, well-lit clip with the full body visible.")

    if len(ball_trajectory) >= 8:
        launch_angle = fit_launch_angle(ball_trajectory)
        if launch_angle is not None:
            print(f"Estimated shot arc (launch angle): {launch_angle:.1f} degrees above horizontal")
    else:
        print(f"Only {len(ball_trajectory)} ball detections — too few for a reliable arc estimate. Try a clip with better ball visibility.")

    if outcome is not None:
        print(f"Shot outcome: {outcome}")
    elif hoop_box is not None:
        print("Shot outcome: undetermined (ball didn't clearly pass through or bounce off the marked hoop region)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python shooting_form_analyzer.py <video_path> [left|right]")
        sys.exit(1)

    video_path = sys.argv[1]
    side = sys.argv[2] if len(sys.argv) > 2 else "right"
    analyze_video(video_path, shooting_side=side)
