import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
if st.button("Go to Home"):
    st.switch_page("ASD.py")
# Load the trained Logistic Regression model from the pickle file
with open(r'logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load the scaler used for preprocessing
with open(r'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the dataset to get unique values for categorical features
data = pd.read_csv(r"cleaned_autism_data.csv")

# Get unique values for categorical features
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

# Preprocess training data to get reference columns
data.dropna(inplace=True)
features_raw = data[feature_columns]
features_minmax_transform = features_raw.copy()
features_minmax_transform[['age', 'result']] = scaler.transform(features_raw[['age', 'result']])
features_final = pd.get_dummies(features_minmax_transform)
reference_columns = features_final.columns

# Function to preprocess user input
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    input_df[['age', 'result']] = scaler.transform(input_df[['age', 'result']])
    input_df = pd.get_dummies(input_df)
    
    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[reference_columns]
    return input_df

# Streamlit UI
st.title("Autism Spectrum Disorder Detection for Adults (18 and above)")

age = st.number_input("Enter age (18 and above)", min_value=18, max_value=100, step=1)
gender = st.selectbox("Select gender", unique_genders)
ethnicity = st.selectbox("Select ethnicity", unique_ethnicities)
jundice = st.selectbox("Had jaundice?", unique_jundice)
austim = st.selectbox("Family member with autism?", unique_austim)
contry_of_res = st.selectbox("Country of residence", unique_countries)
result = st.number_input("Enter test result", min_value=0.0, max_value=20.0, step=0.1)
relation = st.selectbox("Relation", unique_relations)

# Questions for A1-A10 Scores
questions = [
    "Do you often notice small sounds that others don't?",
    "Do you find it hard to understand when someone is joking or being sarcastic?",
    "Do you prefer to focus on details rather than the big picture?",
    "Do you struggle to make eye contact during conversations?",
    "Do you have specific routines that you need to follow closely?",
    "Do you find it difficult to read facial expressions?",
    "Do you often feel overwhelmed in crowded places?",
    "Do you get very upset when your routine is changed?",
    "Do you find it difficult to make small talk?",
    "Do you prefer doing things the same way every time?"
]

# Store responses
responses = {}

for i, question in enumerate(questions):
    response = st.radio(question, ["Yes", "No"], key=f"A{i+1}_Score")
    responses[f"A{i+1}_Score"] = 1 if response == "Yes" else 0

if st.button("Detect ASD"):
    user_input = {
        'age': age, 'gender': gender, 'ethnicity': ethnicity, 'jundice': jundice, 'austim': austim,
        'contry_of_res': contry_of_res, 'result': result, 'relation': relation, **responses
    }
    
    input_df = preprocess_input(user_input)
    prediction = lr_model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("The model predicts that the individual has ASD.")
    else:
        st.success("The model predicts that the individual does not have ASD.")
