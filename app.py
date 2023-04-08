from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

app = Flask(__name__)

# Load label encoders and scaler
label_encoders = pickle.load('label_encoders.pkl', 'rb')
scaler = pickle.load(open('scaler.pkl', 'rb'))

def preprocess_input(input_df):
    # Encode categorical values using LabelEncoder
    for col, encoder in label_encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # Scale the features using StandardScaler
    input_scaled = scaler.transform(input_df)

    return input_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load your model
    model =pickle.load(open('model.pkl','rb'))

    # Get the input values from the HTML form
    input_values = {
        'Technology_Primary': request.form['Technology_Primary'],
        'City': request.form['City'],
        'B2B_Sales_Medium': request.form['B2B_Sales_Medium'],
        'Sales_Velocity': int(request.form['Sales_Velocity']),
        'Sales_Stage_Iterations': int(request.form['Sales_Stage_Iterations']),
        'Opportunity_Size_USD': float(request.form['Opportunity_Size_USD']),
        'Client_Revenue_Sizing': request.form['Client_Revenue_Sizing'],
        'Client_Employee_Sizing': request.form['Client_Employee_Sizing'],
        'Business_from_Client_Last_Year': request.form['Business_from_Client_Last_Year'],
        'Compete_Intel': request.form['Compete_Intel']
    }

    # Preprocess the input values
    input_df = pd.DataFrame([input_values])
    preprocessed_input = preprocess_input(input_df)

    # Make predictions
    prediction = model.predict(preprocessed_input)


    # Render the predicted result on the HTML page
    if prediction == 0:
        return render_template('index.html', prediction_text='The opportunity is not likely to be won.')
    else:
        return render_template('index.html', prediction_text='The opportunity is likely to be won!')

    # # Return the result
    # return render_template('index.html', prediction_text=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
