# Import necessary libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import serial  # For Arduino communication

# Initialize Arduino connection
arduino = serial.Serial('COM3', 9600)

# Define a confidence threshold for mask detection
lowConfidence = 0.75

# Function to detect and predict mask
def detectAndPredictMask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > lowConfidence:
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

# Load face detection model and mask detection model
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

# Initialize video stream
vs = VideoStream(src=0).start()

# Initialize variables for tracking mask status changes
previous_mask_status = None

# Main loop
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    (locs, preds) = detectAndPredictMask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Determine the current mask status
        if label == "Mask":
            current_mask_status = "Mask"
        else:
            current_mask_status = "No Mask"

        # Check if the mask status has changed
        if current_mask_status != previous_mask_status:
            # Mask status has changed, send data to Arduino
            if current_mask_status == "Mask":
                print("ACCESS GRANTED")
                print("Opening Door")
                # For Arduino connection
                arduino.write(b'H')
            else:
                print("ACCESS DENIED")
                print("Alarm")
                # For Arduino connection
                arduino.write(b'L')

            # Update the previous mask status
            previous_mask_status = current_mask_status

        # Display label and bounding box on the frame
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the frame
    cv2.imshow("Press q to quit", frame)

    # Check for the 'q' key to exit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
