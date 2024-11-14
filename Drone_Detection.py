import cv2
import torch
import warnings
import serial
import time
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

def load_model(path):
    try:
        return torch.hub.load('ultralytics/yolov5', 'custom', path=path, source='github')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_and_draw(frame, model, confidence_threshold=0.5):
    img = Image.fromarray(frame[..., ::-1])
    results = model(img, size=640)
    detections = []

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > confidence_threshold and int(cls) == 0:
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Coords: ({center_x}, {center_y})", (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detections.append((center_x, center_y))
    return frame, detections

def send_coordinates(arduino, coords):
    try:
        for center_x, center_y in coords:
            arduino.write(f"{center_x},{center_y}\n".encode('utf-8'))
    except serial.SerialException as e:
        print(f"Serial communication error: {e}")

model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\nilab\Downloads\Drone-Detection\best.pt', source='github', force_reload=True)
cap = cv2.VideoCapture(0)
arduino = serial.Serial('COM6', 9600)
time.sleep(2)

if not cap.isOpened() or not model:
    print("Error: Unable to initialize camera or model.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame, detections = detect_and_draw(frame, model)
        send_coordinates(arduino, detections)
        cv2.imshow('Drone Tracker', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
arduino.close()
cv2.destroyAllWindows()
