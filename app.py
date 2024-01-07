import pandas as pd
from joblib import load
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model_filename = 'spamdetection_model.joblib'
loaded_model = load(model_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the HTML form
    user_subject = request.form.get('subject')
    user_message = request.form.get('message')

    # Combine subject and message
    user_data = [f'{user_subject}: {user_message}']

    # Predict using the loaded model
    predictions = loaded_model.predict(user_data)

    # Convert prediction to result string
    result = 'Spam' if predictions[0] == 1 else 'Not Spam'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
