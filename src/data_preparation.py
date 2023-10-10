import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, date_parser=True)
    df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df['CloseTime'] = pd.to_datetime(df['CloseTime'], unit='ms')
    df = df.drop(columns=['Unused', 'Id'])
    return df


def feature_engineering(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()

    df['20_day_std'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['MA10'] + (df['20_day_std'] * 2)
    df['lower_band'] = df['MA10'] - (df['20_day_std'] * 2)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Fib_38.2'] = df['Close'].rolling(window=30).apply(lambda x: x.min() + 0.382 * (x.max() - x.min()))
    df['Fib_61.8'] = df['Close'].rolling(window=30).apply(lambda x: x.min() + 0.618 * (x.max() - x.min()))
    df['Fib_23.6'] = df['Close'].rolling(window=30).apply(lambda x: x.min() + 0.236 * (x.max() - x.min()))

    return df.dropna()


def normalize_data(df):
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df_scaled = df.copy()
    df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
    return df_scaled, scaler


def split_data(df, test_size=0.2, validation_size=0.25):
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=validation_size, shuffle=False)
    return train_data, val_data, test_data


if __name__ == "__main__":
    file_path = '../data/raw/data.csv'
    data = load_and_clean_data(file_path)
    data = feature_engineering(data)
    data_normalized, scaler = normalize_data(data)
    train_data, val_data, test_data = split_data(data_normalized)

    print(f'Training Data: {train_data.shape}')
    print(f'Validation Data: {val_data.shape}')
    print(f'Test Data: {test_data.shape}')
