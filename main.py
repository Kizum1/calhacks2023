import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB
    image_copy = np.copy(image)  # create a copy of the image
    image_copy.flags.writeable = False  # set the copy as non-writable
    results = model.process(image_copy)
    image_copy.flags.writeable = True  # set the copy as writable again
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)  # convert RGB -> BGR
    return image_copy, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
mp_holistic.POSE_CONNECTIONS

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Frame read, pretty fast
        ret, frame = cap.read()

        # make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # draw landmarkss
        draw_landmarks(image,results)

        # Output on screen
        cv2.imshow('OpenCV Feed', image)

        # break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# print(len(results.left_hand_landmarks.landmark))

# draw_landmarks(frame, results)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# Save the frame to a file
# cv2.imwrite('output_image.jpg', frame)

# Open the image using VS Code's image viewer
# os.system('code output_image.jpg')

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))