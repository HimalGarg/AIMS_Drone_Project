import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Screen center threshold sensitivity
THRESH = 80  

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        command = "NO HAND"

        # Draw screen center point
        cx_screen, cy_screen = w // 2, h // 2
        cv2.circle(frame, (cx_screen, cy_screen), 5, (255, 0, 0), -1)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Wrist landmark (0)
                wrist = hand_landmarks.landmark[0]

                # Index finger tip landmark (8)
                index_tip = hand_landmarks.landmark[8]

                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)

                # Draw wrist and index tip
                cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 255, 255), -1)
                cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)

                dx = ix - wrist_x
                dy = iy - wrist_y

                # Decide direction based on displacement
                if abs(dx) > abs(dy):  
                    if dx > THRESH:
                        command = "RIGHT"
                    elif dx < -THRESH:
                        command = "LEFT"
                    else:
                        command = "STOP"
                else:
                    if dy > THRESH:
                        command = "DOWN"
                    elif dy < -THRESH:
                        command = "UP"
                    else:
                        command = "STOP"

        # Display command
        cv2.putText(frame, f"COMMAND: {command}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Gesture Controlled Drone - Command Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
