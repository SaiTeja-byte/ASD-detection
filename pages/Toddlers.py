import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
if st.button("Go to Home"):
    st.switch_page("ASD.py")
# Load the trained Logistic Regression model
with open('todlers_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load the MinMaxScaler
with open('minmax_scaler(todler).pkl', 'rb') as f:
    minmax_scaler = pickle.load(f)

# Load dataset to get unique values for categorical features
data = pd.read_csv(r'Toddler Autism1.csv')
data.dropna(inplace=True)

# Extract unique values for categorical inputs
unique_genders = data['gender'].unique()
unique_ethnicities = data['ethnicity'].unique()
unique_jundice = data['jundice'].unique()
unique_autism = data['autism'].unique()

# Define feature columns
feature_columns = ['age', 'gender', 'ethnicity', 'jundice', 'autism', 'result',
                   'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                   'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']

# Process training data for reference columns
features_raw = data[feature_columns].copy()
features_transformed = features_raw.copy()
features_transformed[['age', 'result']] = minmax_scaler.transform(features_raw[['age', 'result']])
features_final = pd.get_dummies(features_transformed)
reference_columns = features_final.columns

def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    input_df[['age', 'result']] = minmax_scaler.transform(input_df[['age', 'result']])
    input_df = pd.get_dummies(input_df)
    
    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    return input_df[reference_columns]

# Streamlit UI
st.title("Toddler Autism Spectrum Disorder (ASD) Prediction")
st.write("Fill in the details below to get a prediction.")

age = st.number_input("Age (in Months)", min_value=1, max_value=36, value=5)
gender = st.selectbox("Gender", unique_genders)
ethnicity = st.selectbox("Ethnicity", unique_ethnicities)
jundice = st.selectbox("Had jaundice?", unique_jundice)
autism = st.selectbox("Family member with autism?", unique_autism)
result = st.number_input("Test Result (e.g., 10.0)", min_value=0.0, value=10.0)

# Updated Autism test questions with Yes/No format
a1 = 1 if st.radio("Does your toddler look at you when you call his/her name?", ["Yes", "No"]) == "Yes" else 0
a2 = 1 if st.radio("Does your toddler make eye contact often?", ["Yes", "No"]) == "Yes" else 0
a3 = 1 if st.radio("Does your toddler point to indicate he/she wants something?", ["Yes", "No"]) == "Yes" else 0
a4 = 1 if st.radio("Does your toddler point to show interest in something?", ["Yes", "No"]) == "Yes" else 0
a5 = 1 if st.radio("Does your toddler engage in pretend play?", ["Yes", "No"]) == "Yes" else 0
a6 = 1 if st.radio("Does your toddler follow where you are looking?", ["Yes", "No"]) == "Yes" else 0
a7 = 1 if st.radio("Does your toddler show signs of comfort when someone is visibly upset?", ["Yes", "No"]) == "Yes" else 0
a8 = 0 if st.radio("Are your toddler's first words typical?", ["Yes", "No"]) == "Yes" else 1  # Encoded as per previous logic
a9 = 1 if st.radio("Does your toddler use gestures (e.g., waving goodbye)?", ["Yes", "No"]) == "Yes" else 0
a10 = 1 if st.radio("Does your toddler stare at nothing for no apparent reason?", ["Yes", "No"]) == "Yes" else 0

if st.button("Detect ASD"):
    user_input = {
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'jundice': jundice,
        'autism': autism,
        'result': result,
        'A1_Score': a1,
        'A2_Score': a2,
        'A3_Score': a3,
        'A4_Score': a4,
        'A5_Score': a5,
        'A6_Score': a6,
        'A7_Score': a7,
        'A8_Score': a8,
        'A9_Score': a9,
        'A10_Score': a10
    }
    
    input_df = preprocess_input(user_input)
    prediction = lr_model.predict(input_df)
    if prediction[0] == 1:
        st.error("The model predicts that the individual **has ASD**.")
    else:
        st.success("The model predicts that the individual **does not have ASD**.")
