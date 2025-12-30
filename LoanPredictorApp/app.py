from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

FEATURE_COLUMNS = [
    ['Dependents',
 'ApplicantIncome',
 'CoapplicantIncome',
 'LoanAmount',
 'Loan_Amount_Term',
 'Credit_History',
 'Gender_Female',
 'Gender_Male',
 'Married_No',
 'Married_Yes',
 'Education_Graduate',
 'Education_Not Graduate',
 'Self_Employed_encoded',
 'Property_Area_Rural',
 'Property_Area_Semiurban',
 'Property_Area_Urban']
]

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    data = {
        "ApplicantIncome": float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount": float(request.form["LoanAmount"]),
        "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
        "Credit_History": float(request.form["Credit_History"])
    }

    gender = request.form["Gender"]
    married = request.form["Married"]
    education = request.form["Education"]
    self_employed = request.form["Self_Employed"]
    property_area = request.form["Property_Area"]
    dependents = request.form["Dependents"]


    data["Gender_Male"] = 1 if gender == "Male" else 0
    data["Married_Yes"] = 1 if married == "Yes" else 0
    data["Education_Not Graduate"] = 1 if education == "Not Graduate" else 0
    data["Self_Employed_Yes"] = 1 if self_employed == "Yes" else 0

    data["Property_Area_Semiurban"] = 1 if property_area == "Semiurban" else 0
    data["Property_Area_Urban"] = 1 if property_area == "Urban" else 0

    data["Dependents_1"] = 1 if dependents == "1" else 0
    data["Dependents_2"] = 1 if dependents == "2" else 0
    data["Dependents_3+"] = 1 if dependents == "3+" else 0

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    result = "Loan Approved" if prediction[0]==1 else "Loan Rejected"
    return render_template("result.html", prediction=result) 

if __name__ == "__main__":
    app.run()