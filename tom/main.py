#https://www.youtube.com/watch?v=O3b8lVF93jU

import numpy as np
import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

vid=1 # (widthxheight)
if vid==1:
    cap = cv2.VideoCapture("highway.mp4")  # Orig (1280x720)
elif vid==2:
    cap = cv2.VideoCapture("Nov2018_OL_63391.mp4") # SpOT (776x582)
elif vid==3:
    cap = cv2.VideoCapture("VID_20201212_120331645.mp4") # Car (1920x1080)
elif vid==4:
    cap = cv2.VideoCapture("balls1.mp4") # balls (1920x1080)


# Object detection from Stable camera
if vid==1:
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # Orig
elif vid==2:
    #object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=200) # SpOT
    object_detector = cv2.createBackgroundSubtractorMOG2() # SpOT
elif vid==3:
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # Car
    #object_detector = cv2.createBackgroundSubtractorMOG2() # SpOT
elif vid==4:
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # balls

cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    if vid==1:
        roi = frame[340: 720,500: 800]  # Orig
    elif vid==2:
        roi = frame[0:582, 0:776]     # SpOT
    elif vid==3:
        roi = frame[500:700,300:1920]     # Car
    elif vid==4:
        roi = frame[0:1080,0:1920]     # balls

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    if vid==4:
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         #_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        cv2.imshow("maskMe", mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if (vid==1 and area > 100) or (vid==2 and area > 3) or (vid==3 and area > 500) or (vid==4):
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        #break
        wait = input("Press key to continue")

cap.release()
cv2.destroyAllWindows()
