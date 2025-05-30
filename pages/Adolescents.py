import streamlit as st
import pickle
import pandas as pd
if st.button("Go to Home"):
    st.switch_page("ASD.py")
# Load the trained model
with open(r'best_adolescents_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Load the scalers used for preprocessing
with open(r'minmax_scaler(adolescents).pkl', 'rb') as f:
    minmax_scaler = pickle.load(f)

with open(r'standard_scaler(adolescents).pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

# Load the dataset to get unique values for categorical features
data = pd.read_csv(r'Autism_Adolescent_Preprocessed.csv')

# Get unique values for categorical features
unique_genders = data['gender'].unique()
unique_ethnicities = data['ethnicity'].unique()
unique_jundice = data['jundice'].unique()
unique_autism = data['autism'].unique()
unique_countries = data['contry_of_res'].unique()
unique_relations = data['relation'].unique()

# Define the feature columns
feature_columns = ['age', 'gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res', 'result',
                   'relation', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                   'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']
data.dropna(inplace=True)
features_raw = data[['age', 'gender', 'ethnicity', 'jundice', 'autism', 'relation',
                     'contry_of_res', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                     'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'result']]
features_transform = pd.DataFrame(data=features_raw)

# Apply both MinMaxScaler and StandardScaler
features_transform[['age', 'result']] = minmax_scaler.transform(features_raw[['age', 'result']])
features_transform[['age', 'result']] = standard_scaler.transform(features_transform[['age', 'result']])

features_final = pd.get_dummies(features_transform)
reference_columns = features_final.columns
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    input_df[['age', 'result']] = minmax_scaler.transform(input_df[['age', 'result']])
    input_df[['age', 'result']] = standard_scaler.transform(input_df[['age', 'result']])
    input_df = pd.get_dummies(input_df)
    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[reference_columns]
    return input_df

# Streamlit App
st.title("Autism Spectrum Disorder Detection for Adolescents (Ages 12-17)")

# User Inputs
age = st.number_input("Enter age (12-17):", min_value=12, max_value=17, step=1)
gender = st.selectbox("Select gender:", unique_genders)
ethnicity = st.selectbox("Select ethnicity:", unique_ethnicities)
jundice = st.selectbox("Had jaundice:", unique_jundice)
autism = st.selectbox("Family member with autism:", unique_autism)
contry_of_res = st.selectbox("Select country of residence:", unique_countries)
result = st.number_input("Enter test result:", min_value=0.0, max_value=10.0, step=0.1)
relation = st.selectbox("Select relation:", unique_relations)

# A1-A10 Scores with Questions
questions = [
    "Do you often notice patterns in things all the time?",
    "Do you find it hard to understand what someone is thinking or feeling by looking at their face?",
    "Do you prefer to focus on the big picture rather than small details?",
    "Is it difficult to join in conversations with friends?",
    "Do you have specific routines that you need to follow closely?",
    "Do you get overwhelmed easily in crowded places?",
    "Do you find it hard to make eye contact when talking to someone?",
    "Do you often repeat the same phrases or words?",
    "Do you struggle to understand jokes or sarcasm?",
    "Do you prefer playing alone with objects rather than interacting with other people in imaginative play?"
]

scores = []
for i, question in enumerate(questions):
    score = st.radio(question, ["Yes", "No"])
    scores.append(1 if score == "Yes" else 0)

# Predict Button
if st.button("Detect ASD"):
    user_input = {
        'age': age, 
        'gender': gender, 
        'ethnicity': ethnicity, 
        'jundice': jundice,
        'autism': autism, 
        'contry_of_res': contry_of_res, 
        'result': result, 
        'relation': relation,
        'A1_Score': scores[0], 
        'A2_Score': scores[1], 
        'A3_Score': scores[2], 
        'A4_Score': scores[3],
        'A5_Score': scores[4], 
        'A6_Score': scores[5], 
        'A7_Score': scores[6], 
        'A8_Score': scores[7],
        'A9_Score': scores[8], 
        'A10_Score': scores[9]
    }

    input_df = preprocess_input(user_input)
    prediction = best_model.predict(input_df)

    if prediction[0] == 1:
        st.error("The model predicts that the individual **has ASD**.")
    else:
        st.success("The model predicts that the individual **does not have ASD**.")
