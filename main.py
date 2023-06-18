import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from mediapipe_utils import mediapipe_detection, draw_styled_landmarks
from data_utils import extract_keypoints, create_data_directories, collect_frames_and_export_keypoints, load_data
from model_utils import train_model, save_model, load_model, evaluate_model, calculate_accuracy, calculate_confusion_matrix
from camera import createModel

# Set up paths and parameters
DATA_PATH = os.path.join('MP_Data')
LOG_DIR = os.path.join('Logs')
MODEL_FILE = 'action.h5'

actions = np.array(['hello'])
num_sequences = 50
sequence_length = 30

# Step 1: Create data directories
# create_data_directories(DATA_PATH, actions, num_sequences)

# Step 2: Collect frames and export keypoints
cap = cv2.VideoCapture(0)
createModel(cap)


#collect_frames_and_export_keypoints(cap, DATA_PATH, actions, num_sequences, sequence_length)

# # Step 3: Load data
# X, y = load_data(DATA_PATH, actions, sequence_length)

# # Step 4: Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# # Step 5: Train the model
# model = train_model(X_train, y_train, LOG_DIR)

# # Step 6: Save the trained model
# save_model(model, MODEL_FILE)

# # Step 7: Load the trained model
# loaded_model = load_model(MODEL_FILE)

# # Step 8: Evaluate the model
# yhat, ytrue = evaluate_model(loaded_model, X_test, y_test)

# # Step 9: Calculate accuracy
# accuracy = calculate_accuracy(yhat, ytrue)

# # Step 10: Calculate confusion matrix
# confusion_matrix = calculate_confusion_matrix(yhat, ytrue)

# # Print the results
# print(f"Accuracy: {accuracy}")
# print(f"Confusion Matrix:\n{confusion_matrix}")
