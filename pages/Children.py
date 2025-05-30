import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
if st.button("Go to Home"):
    st.switch_page("ASD.py")
# Load the trained Logistic Regression model
with open(r'Child_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load the MinMaxScaler
with open(r'minmax_scaler(Childs).pkl', 'rb') as f:
    minmax_scaler = pickle.load(f)

# Load the StandardScaler
with open(r'standard_scaler(Childs).pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

# Load dataset to get unique values for categorical features
data = pd.read_csv(r'Preprocessed_Autism_Data_child.csv')
data.dropna(inplace=True)

# Extract unique values for categorical inputs
unique_genders = data['gender'].unique()
unique_ethnicities = data['ethnicity'].unique()
unique_jundice = data['jundice'].unique()
unique_austim = data['austim'].unique()
unique_countries = data['contry_of_res'].unique()
unique_relations = data['relation'].unique()

# Define feature columns
feature_columns = ['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'result',
                   'relation', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                   'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']

# Process training data for reference columns
features_raw = data[feature_columns].copy()
features_transformed = features_raw.copy()
features_transformed[['age', 'result']] = minmax_scaler.transform(features_raw[['age', 'result']])
features_transformed[['age', 'result']] = standard_scaler.transform(features_transformed[['age', 'result']])
features_final = pd.get_dummies(features_transformed)
reference_columns = features_final.columns

def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    input_df[['age', 'result']] = minmax_scaler.transform(input_df[['age', 'result']])
    input_df[['age', 'result']] = standard_scaler.transform(input_df[['age', 'result']])
    input_df = pd.get_dummies(input_df)
    
    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    return input_df[reference_columns]

# Streamlit UI
st.title("Autism Spectrum Disorder Detection for Children (4-11 years)")
st.write("Fill in the details below to get a prediction.")

age = st.number_input("Age", min_value=4, max_value=11, value=5)
gender = st.selectbox("Gender", unique_genders)
ethnicity = st.selectbox("Ethnicity", unique_ethnicities)
jundice = st.selectbox("Had jaundice?", unique_jundice)
austim = st.selectbox("Family member with autism?", unique_austim)
country = st.selectbox("Country of residence", unique_countries)
result = st.number_input("Test Result (e.g., 10.0)", min_value=0.0, max_value=10.00, value=0.0)
relation = st.selectbox("Relation", unique_relations)

# Autism test scores (Yes/No to 1/0 conversion)
a1 = 1 if st.radio("Does your child speak very little and give unrelated answers to questions?", ["Yes","No"]) == "Yes" else 0
a2 = 1 if st.radio("Does your child not respond to their name or avoid eye contact?", ["Yes","No"]) == "Yes" else 0
a3 = 1 if st.radio("Does your child not engage in games of pretend with other children?", ["Yes","No"]) == "Yes" else 0
a4 = 1 if st.radio("Does your child struggle to understand other people’s feelings?", ["Yes","No"]) == "Yes" else 0
a5 = 1 if st.radio("Is your child easily upset by small changes?", ["Yes","No"]) == "Yes" else 0
a6 = 1 if st.radio("Does your child have obsessive interests?", ["Yes","No"]) == "Yes" else 0
a7 = 1 if st.radio("Is your child over or under-sensitive to smells, tastes, or touch?", ["Yes","No"]) == "Yes" else 0
a8 = 1 if st.radio("Does your child struggle to socialize with other children?", ["Yes","No"]) == "Yes" else 0
a9 = 1 if st.radio("Does your child avoid physical contact?", ["Yes","No"]) == "Yes" else 0
a10 = 1 if st.radio("Does your child show little awareness of dangerous situations?", ["Yes","No"]) == "Yes" else 0

if st.button("Detect ASD"):
    user_input = {
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'jundice': jundice,
        'austim': austim,
        'contry_of_res': country,
        'result': result,
        'relation': relation,
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
