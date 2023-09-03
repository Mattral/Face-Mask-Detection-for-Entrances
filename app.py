# Import the necessary libraries
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io
import time

# Load the pre-trained face detection model
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the pre-trained mask detection model
maskNet = load_model("mask_detector.model")

# Function to detect and predict mask in an image
def detect_and_predict_mask(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.75:  # Adjust confidence threshold if needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Streamlit app
st.title("Face Mask Detection")

# Start video capture from the webcam
vs = cv2.VideoCapture(0)

# Create a Streamlit image container
image_container = st.image([], use_column_width=True)

while True:
    _, frame = vs.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (locs, preds) = detect_and_predict_mask(frame)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        
        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Update the Streamlit image container
    image_container.image(frame, channels="BGR", use_column_width=True)

    if st.button("Quit"):
        break

# Release the video capture and close the Streamlit app
vs.release()
cv2.destroyAllWindows()
