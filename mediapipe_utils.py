import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    mage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB
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

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, # Draw face connections
                              mp_drawing.DrawingSpec(color=(80,110, 10), thickness=1, circle_radius=1), # Color land mark, dot 
                              mp_drawing.DrawingSpec(color=(80,256, 110), thickness=1, circle_radius=1)) # Color connections, line
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, # Draw pose connections
                              mp_drawing.DrawingSpec(color=(0,256,256), thickness=2, circle_radius=4), # Color land mark, dot 
                              mp_drawing.DrawingSpec(color=(0,0,256), thickness=2, circle_radius=2)) # Color connections, line)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, # Draw left hand connections
                              mp_drawing.DrawingSpec(color=(256,0, 0), thickness=2, circle_radius=4), # Color land mark, dot 
                              mp_drawing.DrawingSpec(color=(0,0, 256), thickness=2, circle_radius=2)) # Color connections, line)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, # Draw right hand connections
                              mp_drawing.DrawingSpec(color=(0,256, 0), thickness=2, circle_radius=4), # Color land mark, dot 
                              mp_drawing.DrawingSpec(color=(0,0, 256), thickness=2, circle_radius=2)) # Color connections, line)
