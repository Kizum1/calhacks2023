import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from functions.data_utils import actions

def train_model(X_train, y_train, log_dir):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

    return model

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    model = load_model(filename)
    return model

def evaluate_model(model, X_test, y_test):
    res = model.predict(X_test)
    yhat = np.argmax(res, axis=1).tolist()
    ytrue = np.argmax(y_test, axis=1).tolist()

    return yhat, ytrue

def calculate_accuracy(yhat, ytrue):
    accuracy = accuracy_score(ytrue, yhat)
    return accuracy

def calculate_confusion_matrix(yhat, ytrue):
    confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
    return confusion_matrix