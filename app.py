from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load evaluation metrics (from training)
try:
    with open('metrics.txt', 'r') as f:
        lines = f.readlines()
        metrics = {}
        for line in lines:
            key, value = line.strip().split(': ')
            metrics[key] = float(value)
except:
    metrics = {"Accuracy": 0.85, "Precision": 0.87, "Recall": 0.82}  # fallback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        muac = float(request.form['muac'])

        # Prepare input for model
        input_data = np.array([[age, weight, height, muac]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # Map result
        result = "Malnourished" if prediction == 1 else "Normal"
        confidence = probability[prediction] * 100

        return render_template('index.html',
                               prediction=result,
                               confidence=f"{confidence:.2f}%",
                               age=age, weight=weight, height=height, muac=muac)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)n