import os

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config

def load_command_data(commands, data_dir):
    """
    We preprocess data into MFCCs of dimensions 32x32x1, which is the equivalent of a grayscale image
    
    Parameters:
    
    Returns:
    """
    X = []
    y = []
    for idx, command in enumerate(commands):
        command_dir = os.path.join(data_dir, command)
        for file in os.listdir(command_dir):
            filepath = os.path.join(command_dir, file)
            audio, sr = librosa.load(filepath, sr=16000)
            mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
            mfcc = np.resize(mfcc, (32, 32))
            X.append(mfcc)
            y.append(idx)
    X = np.array(X)
    X = X[..., np.newaxis]
    y = np.array(y)
    return X, y

def create_command_model(input_shape, num_commands):
    """
    Create the model, this specific NN architecture was used to make for a lightweight model 
    that can easily run on a raspberry pi
    
    """
    model = models.sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_commands, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("loading command data")
    X_train, y_train = load_command_data(config.VOICE_COMMANDS, DATA_DIR)
    print("data loaded, training model!")
    input_shape = (32, 32, 1)
    num_commands = len(config.VOICE_COMMANDS)
    model = create_command_model(input_shape, num_commands)
    model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.2)
    model.save(config.MODEL_SAVE_PATH)
    print("model completed, saved!")

if __name__ == "__main__":
    main()
