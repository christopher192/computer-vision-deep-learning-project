import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("saved_model/lenet-smile-detection.keras")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame = imutils.resize(frame, width = 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clone_frame = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,
        minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    for (fX, fY, fW, fH) in rects:
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)

        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        cv2.putText(clone_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(clone_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
    
    cv2.imshow('Real-Time Video', clone_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()