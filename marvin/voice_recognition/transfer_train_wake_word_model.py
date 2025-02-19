import config
import utils

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split


# Preprocess our audio samples to match the input of YAMnet, which is directly trained on raw audio waveforms
def load_audio_files(file_path):
    waveforms = []
    for path in file_path:
        audio_binary = tf.io.read_file(path)
        waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        waveform = tf.squeeze(waveform, axis=-1)
        if tf.shape(waveform)[0] < (config.SAMPLE_RATE * config.DURATION):
            zero_padding = tf.zeros([config.SAMPLE_RATE * config.DURATION] - tf.shape(waveform), dtype=tf.float32)
            waveform = tf.concat([waveform, zero_padding], 0)
        else:
            waveform = waveform[:config.NUM_SAMPLES]
        waveforms.append(waveform)
    return tf.stack(waveforms)

def load_wake_word_data(positive_dir, negative_dir):
    positive_files = []
    negative_files = []

    for f in os.listdir(positive_dir):
        if f.endswith('.wav'):
            positive_files.append(os.path.join(positive_dir, f))

    for f in os.listdir(negative_dir):
        if f.endswith('.wav'):
            negative_files.append(os.path.join(negative_dir, f))

    positive_waveforms = load_audio_files(positive_files)
    negative_waveforms = load_audio_files(negative_files)

    X = tf.concat([positive_waveforms, negative_waveforms], axis=0)
    y = tf.concat([tf.ones(len(positive_waveforms)), tf.zeros(len(negative_waveforms))], axis=0)

    return X.numpy(), y.numpy()

# We import YAMnet and add our own layers
def create_wake_word_model():
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    input_waveform = tf.keras.Input(shape=(NUM_SAMPLES,), dtype=tf.float32)

    # YAMnet import dimsdd: [batch size, num samples], so we expand dims
    embeddings = yamnet_model(tf.expand_dims(input_waveform, axis=0))
    embeddings = embeddings['embedding']
    embeddings = tf.reduce_mean(embeddings, axis=0)

    x = tf.keras.layers.Dense(128, activation='relu')(embeddings)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_waveform, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#def extract_embeddings(waveform):
#    _, embeddings, _ = yamnet_model(waveform)
#    return embeddings

def main():
    print("Loading wake word data")
    X, y = load_wake_word_data(config.POSITIVE_DIRECTORY, config.NEGATIVE_DIRECTORY)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(16).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16).prefetch(tf.data.AUTOTUNE)

    model = create_wake_word_model()
    model.summary()

    print("Training wake_word_model.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    ]
    model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=callbacks, verbose=1)

    model.save(config.MODEL_PATH, save_format='keras_v3')
    print("wake word model training complete!")

if __name__ == "__main__":
    main()
