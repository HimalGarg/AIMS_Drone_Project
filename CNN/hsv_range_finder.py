import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

# Create trackbars for HSV lower and upper
cv2.createTrackbar("H Lower", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("S Lower", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Lower", "Trackbars", 0, 255, nothing)

cv2.createTrackbar("H Upper", "Trackbars", 20, 179, nothing)
cv2.createTrackbar("S Upper", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Upper", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read trackbar positions
    hL = cv2.getTrackbarPos("H Lower", "Trackbars")
    sL = cv2.getTrackbarPos("S Lower", "Trackbars")
    vL = cv2.getTrackbarPos("V Lower", "Trackbars")

    hU = cv2.getTrackbarPos("H Upper", "Trackbars")
    sU = cv2.getTrackbarPos("S Upper", "Trackbars")
    vU = cv2.getTrackbarPos("V Upper", "Trackbars")

    lower = np.array([hL, sL, vL])
    upper = np.array([hU, sU, vU])

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Show results
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    # Print values when you press 's'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("HSV_LOWER =", lower)
        print("HSV_UPPER =", upper)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
