import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

cap = cv2.VideoCapture(1)
while cap.isOpened():

    # Frame read, pretty fast
    ret, frame = cap.read()
    # Output on screen
    cv2.imshow('OpenCV Feed', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()