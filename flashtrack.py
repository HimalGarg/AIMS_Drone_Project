
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_NAME = "yolov8n.pt"
TARGET_CLASS = "person"
CONF_THRESH = 0.25
SMOOTHING = 0.15                  
CAM_INDEX = 0                     


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# Load YOLO model
model = YOLO(MODEL_NAME)

# Open webcam
cap = cv2.VideoCapture(CAM_INDEX)

# Flashlight dot (simulated beam)
beam_x, beam_y = None, None

print("Press Q to quit.")
print(f"Tracking object: {TARGET_CLASS}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    if beam_x is None:
        beam_x, beam_y = W // 2, H // 2

    # Run YOLO inference
    results = model(frame, verbose=False)[0]

    best_box = None
    best_conf = 0

    # Iterate detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        if cls_name == TARGET_CLASS and conf > CONF_THRESH:
            if conf > best_conf:
                best_conf = conf
                best_box = box

    # If object found
    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

        # Object center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Smooth movement of flashlight dot
        beam_x = int(beam_x + SMOOTHING * (cx - beam_x))
        beam_y = int(beam_y + SMOOTHING * (cy - beam_y))

        # Clamp beam inside frame
        beam_x = clamp(beam_x, 0, W - 1)
        beam_y = clamp(beam_y, 0, H - 1)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        cv2.putText(frame, f"{TARGET_CLASS} {best_conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    # Draw center crosshair
    cv2.line(frame, (W // 2 - 20, H // 2), (W // 2 + 20, H // 2), (255, 255, 255), 2)
    cv2.line(frame, (W // 2, H // 2 - 20), (W // 2, H // 2 + 20), (255, 255, 255), 2)

    # Draw simulated flashlight beam
    overlay = frame.copy()
    cv2.circle(overlay, (beam_x, beam_y), 50, (0, 255, 255), -1)
    alpha = 0.25
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.circle(frame, (beam_x, beam_y), 8, (0, 255, 255), -1)
    cv2.putText(frame, "FLASHLIGHT", (beam_x + 10, beam_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Flashlight Object Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
