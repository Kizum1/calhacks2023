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

#original, no format
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


cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Frame read, pretty fast
        ret, frame = cap.read()

        # make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # draw landmarks
        draw_styled_landmarks(image,results)

        # Output on screen
        cv2.imshow('OpenCV Feed', image)

        # break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

# extracting keypoint values from each part


# pose = []
# for res in results.pose_landmarks.landmark:
#     test = np.array([res.x, res.y, res.z, res.visibility])
#     pose.append(test)
# the following funciton is what this does again ^

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

print(extract_keypoints(results)[:10])

# Path for exported data, numpy arrays, where to store
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# 30 videos of data
num_sequences = 30

# 30 frames in the vid
sequence_length = 30

for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# print(len(results.left_hand_landmarks.landmark))

# draw_landmarks(frame, results)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# Save the frame to a file
# cv2.imwrite('output_image.jpg', frame)

# Open the image using VS Code's image viewer
# os.system('code output_image.jpg')

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))