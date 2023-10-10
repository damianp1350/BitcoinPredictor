from data_preparation import load_and_clean_data, feature_engineering, normalize_data, split_data
from model import create_lstm_model, create_sequences


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return model, history


def plot_training_history(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()


if __name__ == "__main__":
    SEQ_LENGTH = 1
    EPOCHS = 1000
    BATCH_SIZE = 32

    file_path = '../data/raw/data.csv'
    data = load_and_clean_data(file_path)
    data = feature_engineering(data)
    data_normalized, _ = normalize_data(data[['Close']])
    train_data, val_data, _ = split_data(data_normalized)

    X_train, y_train = create_sequences(train_data.values, SEQ_LENGTH)
    X_val, y_val = create_sequences(val_data.values, SEQ_LENGTH)

    model = create_lstm_model((SEQ_LENGTH, 1))
    model, history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

    model.save('../models/lstm_model_seq1_v2.h5')
