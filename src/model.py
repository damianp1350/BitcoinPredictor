import tensorflow as tf
import numpy as np
from data_preparation import load_and_clean_data, feature_engineering, normalize_data, split_data


def create_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        target.append(data[i + seq_length])
    return np.array(sequences), np.array(target)
