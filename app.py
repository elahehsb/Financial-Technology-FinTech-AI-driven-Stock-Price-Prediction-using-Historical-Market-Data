from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

app = Flask(__name__)
model = load_model('stock_price_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

def preprocess_data(data):
    scaled_data = scaler.fit_transform(data)
    return scaled_data

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    processed_data = preprocess_data(df)
    
    time_step = 60
    X = []
    for i in range(time_step, len(processed_data)):
        X.append(processed_data[i-time_step:i, 0])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
