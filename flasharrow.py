

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_NAME = "yolov8n.pt"
TARGET_CLASS = "person"
CONF_THRESH = 0.25
CAM_INDEX = 0
DEADZONE = 30


model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(CAM_INDEX)


def draw_arrow(frame, direction):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)

    if direction == "LEFT":
        end = (center[0] - 150, center[1])
    elif direction == "RIGHT":
        end = (center[0] + 150, center[1])
    elif direction == "UP":
        end = (center[0], center[1] - 150)
    elif direction == "DOWN":
        end = (center[0], center[1] + 150)
    else:
        return

    cv2.arrowedLine(frame, center, end, (0, 0, 255), 6, tipLength=0.4)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Run YOLO inference every frame
    results = model(frame, verbose=False)[0]

    best_box = None
    best_conf = 0

    # Find best detection of target class
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        if cls_name == TARGET_CLASS and conf > CONF_THRESH:
            if conf > best_conf:
                best_conf = conf
                best_box = box

    target_center = None

    # If object found
    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        target_center = (cx, cy)

        # Draw detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        cv2.putText(frame, f"{TARGET_CLASS.upper()} {best_conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    # Draw center crosshair
    cv2.circle(frame, (W // 2, H // 2), 8, (255, 255, 255), -1)

    # Arrow Guidance
    if target_center is not None:
        cx, cy = target_center

        error_x = cx - (W // 2)
        error_y = cy - (H // 2)

        direction = None

        # Choose dominant direction
        if abs(error_x) > abs(error_y):
            if error_x > DEADZONE:
                direction = "RIGHT"
            elif error_x < -DEADZONE:
                direction = "LEFT"
        else:
            if error_y > DEADZONE:
                direction = "DOWN"
            elif error_y < -DEADZONE:
                direction = "UP"

        if direction:
            draw_arrow(frame, direction)
            cv2.putText(frame, f"MOVE {direction}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "LOCKED",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "SEARCHING...",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 3)

    cv2.imshow("YOLO Flashlight Arrow Tracker (No Tracker)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
