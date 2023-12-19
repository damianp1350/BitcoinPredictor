from flask import Flask, request, jsonify
from flasgger import Swagger
from predict import predict_next_day_close, load_and_clean_data, feature_engineering, normalize_data
import numpy as np

app = Flask(__name__)
Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
        Predict the Next Day Close Price
        ---
        consumes:
          - multipart/form-data
        parameters:
          - name: file
            in: formData
            type: file
            required: true
        responses:
          200:
            description: The output values
        """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        temp_file_path = "../data/raw/temp_data.csv"
        file.save(temp_file_path)

        recent_data = load_and_clean_data(temp_file_path)

        model_path = '../models/lstm_model_seq1.h5'
        SEQ_LENGTH = 1

        data = feature_engineering(recent_data)
        data_normalized, scaler = normalize_data(data[['Close']])

        next_day_close = predict_next_day_close(model_path, recent_data, scaler, SEQ_LENGTH)

        return jsonify({'predicted_close_price': next_day_close})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
