import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (most powerful version)
model = YOLO("yolo11x.pt")  # 'yolov8x.pt' is the largest model (YOLOv8x)

# Open a connection to the webcam (usually device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLOv8 on the frame
    results = model(frame)

    # Convert the results to a format suitable for OpenCV (annotated with bounding boxes)
    annotated_frame = results[0].plot()  # Annotate frame with detected objects

    # Display the annotated frame
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()