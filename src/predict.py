import tensorflow as tf
import numpy as np
from data_preparation import load_and_clean_data, feature_engineering, normalize_data


def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        target.append(data[i + seq_length])
    return np.array(sequences), np.array(target)


def predict_next_day_close(model_path, recent_data, scaler, seq_length):
    model = tf.keras.models.load_model(model_path)

    data = feature_engineering(recent_data)
    data_normalized, _ = normalize_data(data[['Close']])

    input_seq, _ = create_sequences(data_normalized.values, seq_length)
    last_sequence = input_seq[-1].reshape(1, seq_length, 1)

    next_day_close_normalized = model.predict(last_sequence)[0][0]
    next_day_close = scaler.inverse_transform([[next_day_close_normalized]])[0][0]

    return next_day_close


if __name__ == "__main__":
    file_path = '../data/raw/data.csv'
    recent_data = load_and_clean_data(file_path)

    model_path = '../models/lstm_model_seq1.h5'
    SEQ_LENGTH = 1

    data = load_and_clean_data(file_path)
    data = feature_engineering(data)
    data_normalized, scaler = normalize_data(data[['Close']])

    next_day_close = predict_next_day_close(model_path, recent_data, scaler, SEQ_LENGTH)

    print(f"Predicted Close Price for the Next Day: ${next_day_close:.2f}")
