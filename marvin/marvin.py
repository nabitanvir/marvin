import config
import utils

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import models

# Constants
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE  # 1 second of audio
MODEL_PATH = "/usr/src/app/models/wake_word_model.h5"  # Update with the actual path to your model
THRESHOLD = 0.5

def preprocess_audio(audio):
    """Preprocess audio to match model input."""
    audio = librosa.util.fix_length(audio, size=NUM_SAMPLES)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    max_length = 32
    if mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]
    elif mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    mfcc = mfcc[..., np.newaxis]  # Shape: (13, 32, 1)
    return mfcc

def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    model = models.load_model(MODEL_PATH)
    return model

def load_audio_files(directory):
    """Load audio files from a directory labeled as positive or negative."""
    files = []
    labels = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.startswith("positive_"):
            label = 1
        elif filename.startswith("negative_"):
            label = 0
        else:
            continue

        filepath = os.path.join(directory, filename)
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        files.append(audio)
        labels.append(label)
        filenames.append(filename)

    return files, labels, filenames

def predict(model, audio):
    """Predict the probability of the wake word."""
    mfcc = preprocess_audio(audio)
    mfcc = np.expand_dims(mfcc, axis=0)  # Shape: (1, 13, 32, 1)
    prediction = model.predict(mfcc)[0][0]
    return prediction

def evaluate_files_individually(model, audio_files, labels, filenames):
    """Evaluate each file and provide live feedback."""
    correct = 0
    total = len(audio_files)

    for audio, label, filename in zip(audio_files, labels, filenames):
        print(f"Processing file: {filename} | Expected label: {'Positive' if label == 1 else 'Negative'}")
        probability = predict(model, audio)
        prediction = 1 if probability > THRESHOLD else 0

        if prediction == label:
            print("marvin: hello!")
            correct += 1
        else:
            print("marvin: seems no one's here :/")

    accuracy = (correct / total) * 100
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}%")

def main():
    print("Loading the wake word detection model...")
    model = load_model()
    print("Model loaded successfully.\n")

    directory = '/usr/src/app/test_model'
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    print("Loading audio files...")
    audio_files, labels, filenames = load_audio_files(directory)
    if not audio_files:
        print("No valid audio files found in the directory.")
        return

    print("Evaluating files individually...")
    evaluate_files_individually(model, audio_files, labels, filenames)

if __name__ == "__main__":
    main()

