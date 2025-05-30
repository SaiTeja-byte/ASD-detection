import streamlit as st
import pickle
import pandas as pd

from sklearn.preprocessing import MinMaxScaler,StandardScaler
def home_page():
    st.title("Autism Spectrum Disorder (ASD) Detection")
    st.write("Select the category for ASD detection:")
    
    option = st.radio("Choose a category:", ["Adults", "Children", "Toddlers", "Adolescents"])
    
    if st.button("Proceed"):
        st.session_state["page"] = option

        # Navigation Logic
        if "page" not in st.session_state:
            home_page()
        elif st.session_state["page"] == "Adults":
            adults_page()
        elif st.session_state["page"] == "Children":
            children_page()
        elif st.session_state["page"] == "Toddlers":
            toddlers_page()
        elif st.session_state["page"] == "Adolescents":
            adolescents_page()

def adults_page():
    st.title("ASD Detection for Adults")
    st.write("Fill out the questionnaire for ASD detection in adults.")
   

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

    # st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'home'))

def children_page():
    st.title("ASD Detection for Children")
    st.write("Fill out the questionnaire for ASD detection in children.")

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

    st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'home'))

def toddlers_page():
    st.title("ASD Detection for Toddlers")
    st.write("Fill out the questionnaire for ASD detection in toddlers.")
   

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

    st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'home'))

def adolescents_page():
    st.title("ASD Detection for Adolescents")
    st.write("Fill out the questionnaire for ASD detection in adolescents.")


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

    def preprocess_input(user_input):
        input_df = pd.DataFrame([user_input], columns=feature_columns)
        input_df[['age', 'result']] = minmax_scaler.transform(input_df[['age', 'result']])
        input_df[['age', 'result']] = standard_scaler.transform(input_df[['age', 'result']])
        input_df = pd.get_dummies(input_df)
        for col in reference_columns: # type: ignore
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[reference_columns] # type: ignore
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

    st.button("Back", on_click=lambda: setattr(st.session_state, 'page', 'home'))

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'adults':
        adults_page()
    elif st.session_state.page == 'children':
        children_page()
    elif st.session_state.page == 'toddlers':
        toddlers_page()
    elif st.session_state.page == 'adolescents':
        adolescents_page()

if __name__ == "__main__":
    main()
