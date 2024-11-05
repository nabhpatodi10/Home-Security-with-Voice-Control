import numpy as np
import cv2
import tensorflow as tf
import asyncio
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from torch import tensor
import Voice_Assistant.Voice_Assistant as va  # Assuming Voice_Assistant has the necessary functions
from ultralytics import YOLO

# Load the face identification model
model = tf.keras.models.load_model('model.h5')

# Load YOLOv8 model for face detection
yolo_model = YOLO("yolo11x.pt")  # Adjust to the correct YOLO model path if needed

# Classes for prediction output
classes = ["negative", "positive"]

# Preprocess frame function for consistency with model expectations
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = img_to_array(frame)
    frame = preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Helper function to run the voice assistant asynchronously
async def activate_voice_assistant():
    print("Starting Voice Assistant")
    await asyncio.to_thread(va.voice_assistant())  # Runs the voice assistant in a separate thread

# Function to process individual detections
async def process_detection(frame, detection):
    # Check if the label is "person"
    if detection.cls != tensor(0.):
        print("No person detected, ", detection.cls)
        return

    x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
    confidence = detection.conf[0].item()

    # Process only detections with a high confidence score (e.g., > 0.5)
    if confidence > 0.5:
        face = frame[y1:y2, x1:x2]

        # Preprocess the face for model prediction
        face_input = preprocess_frame(face)

        # Predict using the model
        prediction = model.predict(face_input)
        label = classes[np.argmax(prediction)]

        # If the owner is detected, activate the voice assistant
        label = 'positive'
        if label == 'positive':
            print("Owner Detected")
            asyncio.create_task(activate_voice_assistant())

# Main loop for capturing video and processing asynchronously
async def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 on the frame for face detection
        results = yolo_model(frame)

        # Create a list of tasks for processing "person" detections
        tasks = [process_detection(frame, detection) for detection in results[0].boxes if detection.cls == tensor(0.)]

        # Run all detection processing tasks concurrently
        await asyncio.gather(*tasks)

        # Display the frame with YOLO annotations
        annotated_frame = results[0].plot()  # Annotate frame with detected faces
        cv2.imshow("YOLO Face Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main loop
asyncio.run(main())