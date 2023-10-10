from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from data_preparation import load_and_clean_data, feature_engineering, normalize_data, split_data
from model import create_sequences


def evaluate_model(model_path, X_test, y_test, scaler):
    model = tf.keras.models.load_model(model_path)

    loss, _ = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss on Test Data: {loss:.4f}")

    predictions = model.predict(X_test)

    y_test_original = scaler.inverse_transform(y_test)
    predictions_original = scaler.inverse_transform(predictions)

    mae = mean_absolute_error(y_test_original, predictions_original)
    print(f"Mean Absolute Error on Test Data: {mae:.4f}")

    plot_predictions(y_test_original, predictions_original)


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label="Actual", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted Close Prices")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    SEQ_LENGTH = 1

    file_path = '../data/raw/data.csv'
    data = load_and_clean_data(file_path)
    data = feature_engineering(data)
    data_normalized, scaler = normalize_data(data[['Close']])
    _, _, test_data = split_data(data_normalized)

    X_test, y_test = create_sequences(test_data.values, SEQ_LENGTH)
    model_path = '../models/lstm_model_seq1_v2.h5'
    evaluate_model(model_path, X_test, y_test, scaler)
