from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

# Render the home page with the form
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        user_input = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': int(request.form['dependents']),
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'ApplicantIncome': int(request.form['applicant_income']),
            'CoapplicantIncome': int(request.form['coapplicant_income']),
            'LoanAmount': int(request.form['loan_amount']),
            'Loan_Amount_Term': int(request.form['loan_amount_term']),
            'Credit_History': int(request.form['credit_history']),
            'Property_Area': request.form['property_area']
        }

        # Convert user input to a DataFrame for model prediction
        input_df = pd.DataFrame([user_input])

        # One-hot encode categorical columns
        input_df = pd.get_dummies(input_df)

        # Scale numerical columns using the saved scaler
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Perform prediction using the loaded model
        prediction = model.predict(input_df)

        # Map the binary prediction to a human-readable result
        result = 'Approved' if prediction[0] == 1 else 'Not Approved'

        # Render the result on the same page
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
