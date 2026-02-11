# 🔦 YOLOv8 Object Tracking Prototype (Flashlight + Arrow Guidance)

This project is a **software prototype** that detects objects in a live webcam feed using **YOLOv8** and helps aim a flashlight/laser system toward the detected object.

It includes two modes:

✅ **Flashlight Mode** – simulates a flashlight beam that follows the detected object  
✅ **Arrow Mode** – shows directional arrows telling how to move the flashlight to lock onto the object  

This prototype can later be integrated with a real drone + gimbal + flashlight/laser.

---

## 📌 Features

### 🔦 Flashlight Tracker
- Detects a specific object using YOLOv8
- Draws bounding box around the detected object
- Simulates a flashlight beam (yellow circle overlay)
- Smooth movement of beam towards the object

### 🏹 Arrow Guidance Mode
- Detects a specific object using YOLOv8
- Displays arrows (`UP`, `DOWN`, `LEFT`, `RIGHT`) based on target position
- Shows **LOCKED** when the target is centered

---

## 📂 Project Files
├── prototype_flashlight_tracker.py
├── prototype_flashlight_arrows_yolo_only.py
├── CNN                                     (This is where I have trained my own CNN)
├── main.py                                 (Gesture detection using Mediapipe)
├── requirements.txt
└── README.md

---

## 🎯 Selecting Target Object

Both scripts detect a single object class specified in this line:

```python
TARGET_CLASS = "cell phone"
TARGET_CLASS = "person"
TARGET_CLASS = "bottle"
TARGET_CLASS = "laptop"
TARGET_CLASS = "chair"
TARGET_CLASS = "car"
```
## Configuration Options

Inside the scripts:

Confidence Threshold:
```
CONF_THRESH = 0.25
```

Increase → fewer false detections but more missed frames

Decrease → more detections but may include wrong objects

For small objects like cell phone, try:
```
CONF_THRESH = 0.15
```
Deadzone (Arrow Mode)
```
DEADZONE = 30
```

If the object is within this many pixels of the center, the system shows LOCKED.

## 🧠 How It Works (Basic Logic)

YOLO outputs bounding box coordinates:

(x1, y1, x2, y2)


Target center is calculated as:
```
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2

```
Then error is computed relative to screen center:
```
error_x = cx - (W/2)
error_y = cy - (H/2)
```

Arrow direction is chosen based on which error is larger.