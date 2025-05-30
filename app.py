from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import pandas as pd
import numpy as np
import os
import warnings
import dice_ml
from dice_ml.utils import helpers
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load models and scalers
def load_models():
    models = {}
    
    # Load adult model and scaler
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        models['adult_model'] = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        models['adult_scaler'] = pickle.load(f)
    
    # Load children model and scalers
    with open('models/Child_model.pkl', 'rb') as f:
        models['child_model'] = pickle.load(f)
    with open('models/minmax_scaler(Childs).pkl', 'rb') as f:
        models['child_minmax'] = pickle.load(f)
    with open('models/standard_scaler(Childs).pkl', 'rb') as f:
        models['child_standard'] = pickle.load(f)
    
    # Load toddler model and scaler
    with open('models/todlers_model.pkl', 'rb') as f:
        models['toddler_model'] = pickle.load(f)
    with open('models/minmax_scaler(todler).pkl', 'rb') as f:
        models['toddler_scaler'] = pickle.load(f)
    
    # Load adolescent model and scalers
    with open('models/best_adolescents_model.pkl', 'rb') as f:
        models['adolescent_model'] = pickle.load(f)
    with open('models/minmax_scaler(adolescents).pkl', 'rb') as f:
        models['adolescent_minmax'] = pickle.load(f)
    with open('models/standard_scaler(adolescents).pkl', 'rb') as f:
        models['adolescent_standard'] = pickle.load(f)
    
    return models

# Load reference data
def load_reference_data():
    ref_data = {}
    
    # Load adult data
    adult_data = pd.read_csv("data/cleaned_autism_data.csv")
    adult_data.dropna(inplace=True)
    features_raw = adult_data[['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'result',
                    'relation', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']]
    features_minmax_transform = features_raw.copy()
    features_minmax_transform[['age', 'result']] = models['adult_scaler'].transform(features_raw[['age', 'result']])
    features_final = pd.get_dummies(features_minmax_transform)
    ref_data['adult_columns'] = features_final.columns
    ref_data['adult_unique'] = {
        'genders': adult_data['gender'].unique(),
        'ethnicities': adult_data['ethnicity'].unique(),
        'jundice': adult_data['jundice'].unique(),
        'austim': adult_data['austim'].unique(),
        'countries': adult_data['contry_of_res'].unique(),
        'relations': adult_data['relation'].unique()
    }
    
    # Load child data
    child_data = pd.read_csv("data/Preprocessed_Autism_Data_child.csv")
    child_data.dropna(inplace=True)
    child_features = child_data[['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'result',
                    'relation', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']]
    child_transformed = child_features.copy()
    child_transformed[['age', 'result']] = models['child_minmax'].transform(child_features[['age', 'result']])
    child_transformed[['age', 'result']] = models['child_standard'].transform(child_transformed[['age', 'result']])
    child_final = pd.get_dummies(child_transformed)
    ref_data['child_columns'] = child_final.columns
    ref_data['child_unique'] = {
        'genders': child_data['gender'].unique(),
        'ethnicities': child_data['ethnicity'].unique(),
        'jundice': child_data['jundice'].unique(),
        'austim': child_data['austim'].unique(),
        'countries': child_data['contry_of_res'].unique(),
        'relations': child_data['relation'].unique()
    }
    
    # Load toddler data
    toddler_data = pd.read_csv("data/Toddler Autism1.csv")
    toddler_data.dropna(inplace=True)
    toddler_features = toddler_data[['age', 'gender', 'ethnicity', 'jundice', 'autism', 'result',
                    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']]
    toddler_transformed = toddler_features.copy()
    toddler_transformed[['age', 'result']] = models['toddler_scaler'].transform(toddler_features[['age', 'result']])
    toddler_final = pd.get_dummies(toddler_transformed)
    ref_data['toddler_columns'] = toddler_final.columns
    ref_data['toddler_unique'] = {
        'genders': toddler_data['gender'].unique(),
        'ethnicities': toddler_data['ethnicity'].unique(),
        'jundice': toddler_data['jundice'].unique(),
        'autism': toddler_data['autism'].unique(),
        'relations': toddler_data['Who completed the test'].unique()
    }
    
    # Load adolescent data
    adolescent_data = pd.read_csv("data/Autism_Adolescent_Preprocessed.csv")
    adolescent_data.dropna(inplace=True)
    adolescent_features = adolescent_data[['age', 'gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res', 'result',
                    'relation', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']]
    adolescent_transformed = adolescent_features.copy()
    adolescent_transformed[['age', 'result']] = models['adolescent_minmax'].transform(adolescent_features[['age', 'result']])
    adolescent_transformed[['age', 'result']] = models['adolescent_standard'].transform(adolescent_transformed[['age', 'result']])
    adolescent_final = pd.get_dummies(adolescent_transformed)
    ref_data['adolescent_columns'] = adolescent_final.columns
    ref_data['adolescent_unique'] = {
        'genders': adolescent_data['gender'].unique(),
        'ethnicities': adolescent_data['ethnicity'].unique(),
        'jundice': adolescent_data['jundice'].unique(),
        'autism': adolescent_data['autism'].unique(),
        'countries': adolescent_data['contry_of_res'].unique(),
        'relations': adolescent_data['relation'].unique()
    }
    
    return ref_data

# Initialize models and reference data
models = load_models()
ref_data = load_reference_data()
def decode_one_hot(cf_df, categorical_features, prefix_map):
    """Convert one-hot encoded columns back to categorical values."""
    decoded_df = cf_df.copy()
    for feature, prefix in prefix_map.items():
        if feature in categorical_features:
            one_hot_cols = [col for col in decoded_df.columns if col.startswith(prefix)]
            if one_hot_cols:
                # Get the column with the highest value (1 for one-hot)
                decoded_df[feature] = decoded_df[one_hot_cols].idxmax(axis=1).str.replace(prefix, '')
                decoded_df = decoded_df.drop(columns=one_hot_cols)
    return decoded_df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/adults', methods=['GET', 'POST'])
def adults():
    if request.method == 'POST':
        # Get form data
        user_input = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity'],
            'jundice': request.form['jundice'],
            'austim': request.form['austim'],
            'contry_of_res': request.form['contry_of_res'],
            'result': float(request.form['result']),
            'relation': request.form['relation']
        }
        
        # Get AQ-10 scores
        for i in range(1, 11):
            score_key = f'A{i}_Score'
            user_input[score_key] = 1 if request.form.get(score_key) == 'Yes' else 0
        
        # Preprocess input
        input_df = pd.DataFrame([user_input])
        input_df[['age', 'result']] = models['adult_scaler'].transform(input_df[['age', 'result']])
        input_df = pd.get_dummies(input_df)
        
        # Ensure all columns from training are present
        for col in ref_data['adult_columns']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[ref_data['adult_columns']]
        # Ensure all data is numeric
        input_df = input_df.astype(float)
        # Debug: Print data types
        print("input_df dtypes:\n", input_df.dtypes)
        
        # Make prediction
        prediction = models['adult_model'].predict(input_df)
        result = "has ASD" if prediction[0] == 1 else "does not have ASD"
        probability=round(models['adult_model'].predict_proba(input_df)[0][0]*100,2)
        
        # Generate counterfactual explanations using DiCE-ML
        d = dice_ml.Data(dataframe=pd.concat([input_df, pd.DataFrame({'Class': [prediction[0]]})], axis=1),
                        continuous_features=['age', 'result'],
                        outcome_name='Class')
        m = dice_ml.Model(model=models['adult_model'], backend='sklearn')
        exp = dice_ml.Dice(d, m, method='random')  # Revert to random method
        
        query_instance = input_df
        prefix_map = {'gender_': 'gender_', 'ethnicity_': 'ethnicity_', 'jundice_': 'jundice_', 
              'austim_': 'austim_', 'contry_of_res_': 'contry_of_res_', 'relation_': 'relation_'}
        try:
            cfs = exp.generate_counterfactuals(query_instance,
                                            total_CFs=3,
                                            desired_class="opposite",
                                            features_to_vary=['age', 'result',
                                                            'A1_Score', 'A2_Score', 'A3_Score',
                                                            'A4_Score', 'A5_Score', 'A6_Score',
                                                            'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score'],
                                            permitted_range={'age': [18, 100], 'result': [0, 100]},
                                            proximity_weight=1.0,
                                            diversity_weight=1.0)
            cf_df = cfs.cf_examples_list[0].final_cfs_df
            if cf_df is not None and not cf_df.empty:
                cf_df[['age', 'result']] = models['adult_scaler'].inverse_transform(cf_df[['age', 'result']])
                cf_df = decode_one_hot(cf_df, ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'relation'], prefix_map)
                cf_df = cf_df.round(2)
                counterfactuals = cf_df.to_dict(orient='records')
                counterfactual_message = None
            else:
                counterfactuals = []
                counterfactual_message = "No counterfactuals found. Try adjusting the input or model configuration."
        except Exception as e:
            counterfactuals = []
            counterfactual_message = f"Error generating counterfactuals: {str(e)}"

        return render_template('adults.html', 
                              unique_data=ref_data['adult_unique'], 
                              prediction=result,
                              counterfactuals=counterfactuals,
                              counterfactual_message=counterfactual_message,
                              probability=probability,
                              show_result=True)
    
    return render_template('adults.html', unique_data=ref_data['adult_unique'], show_result=False)

@app.route('/children', methods=['GET', 'POST'])
def children():
    if request.method == 'POST':
        # Get form data
        user_input = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity'],
            'jundice': request.form['jundice'],
            'austim': request.form['austim'],
            'contry_of_res': request.form['contry_of_res'],
            'result': float(request.form['result']),
            'relation': request.form['relation']
        }
        
        # Get AQ-10 scores
        for i in range(1, 11):
            score_key = f'A{i}_Score'
            user_input[score_key] = 1 if request.form.get(score_key) == 'Yes' else 0
        
        # Preprocess input
        input_df = pd.DataFrame([user_input])
        input_df[['age', 'result']] = models['child_minmax'].transform(input_df[['age', 'result']])
        input_df[['age', 'result']] = models['child_standard'].transform(input_df[['age', 'result']])
        input_df = pd.get_dummies(input_df)
        
        # Ensure all columns from training are present
        for col in ref_data['child_columns']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[ref_data['child_columns']]
        # Ensure all data is numeric
        input_df = input_df.astype(float)
        # Debug: Print data types
        print("input_df dtypes:\n", input_df.dtypes)
        
        # Make prediction
        prediction = models['child_model'].predict(input_df)
        result = "has ASD" if prediction[0] == 1 else "does not have ASD"
        probability=round(models['child_model'].predict_proba(input_df)[0][0]*100,2)
        # Generate counterfactual explanations using DiCE-ML
        d = dice_ml.Data(dataframe=pd.concat([input_df, pd.DataFrame({'Class': [prediction[0]]})], axis=1),
                        continuous_features=['age', 'result'],
                        outcome_name='Class')
        m = dice_ml.Model(model=models['child_model'], backend='sklearn')
        exp = dice_ml.Dice(d, m, method='random')  # Revert to random method
        
        query_instance = input_df
        prefix_map = {'gender': 'gender_', 'ethnicity': 'ethnicity', 'jundice': 'jundice_', 
              'austim': 'austim', 'contry_of_res': 'contry_of_res', 'relation': 'relation_'}
        try:
            cfs = exp.generate_counterfactuals(query_instance,
                                            total_CFs=3,
                                            desired_class="opposite",
                                            features_to_vary=['age', 'result', 
                                                            'A1_Score', 'A2_Score', 'A3_Score',
                                                            'A4_Score', 'A5_Score', 'A6_Score',
                                                            'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score'],
                                            permitted_range={'age': [0, 18], 'result': [0, 10]},
                                            proximity_weight=1.0,
                                            diversity_weight=1.0)
            # cf_df=cfs.visualize_as_dataframe(show_only_changes=True)
            print(cfs.visualize_as_dataframe(show_only_changes=True))
            # print(cf_df.columns)
            cf_df = cfs.cf_examples_list[0].final_cfs_df
            if cf_df is not None and not cf_df.empty:
                # print(cf_df.visualize_as_dataframe(show_only_changes=True))
                # Inverse transform standard scaler
                cf_df[['age', 'result']] = models['child_standard'].inverse_transform(cf_df[['age', 'result']])
                # Inverse transform minmax scaler
                cf_df[['age', 'result']] = models['child_minmax'].inverse_transform(cf_df[['age', 'result']])
                cf_df = decode_one_hot(cf_df, ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'relation'], prefix_map)
                cf_df = cf_df.round(2)
                counterfactuals = cf_df.to_dict(orient='records')
                counterfactual_message = None
            else:
                counterfactuals = []
                counterfactual_message = "No counterfactuals found. Try adjusting the input or model configuration."
        except Exception as e:
            counterfactuals = []
            counterfactual_message = f"Error generating counterfactuals: {str(e)}"
        
        return render_template('children.html', 
                              unique_data=ref_data['child_unique'], 
                              prediction=result,
                              counterfactuals=counterfactuals,
                              counterfactual_message=counterfactual_message,
                              probability=probability,
                              show_result=True)
    
    return render_template('children.html', unique_data=ref_data['child_unique'], show_result=False)

@app.route('/toddlers', methods=['GET', 'POST'])
def toddlers():
    if request.method == 'POST':
        # Get form data
        user_input = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity'],
            'jundice': request.form['jundice'],
            'autism': request.form['autism'],
            'result': float(request.form['result'])
        }
        
        # Get AQ-10 scores
        for i in range(1, 11):
            score_key = f'A{i}_Score'
            user_input[score_key] = 1 if request.form.get(score_key) == 'Yes' else 0
        
        # Preprocess input
        input_df = pd.DataFrame([user_input])
        input_df[['age', 'result']] = models['toddler_scaler'].transform(input_df[['age', 'result']])
        input_df = pd.get_dummies(input_df)
        
        # Ensure all columns from training are present
        for col in ref_data['toddler_columns']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[ref_data['toddler_columns']]
        # Ensure all data is numeric
        input_df = input_df.astype(float)
        # Debug: Print data types
        print("input_df dtypes:\n", input_df.dtypes)
        
        # Make prediction
        prediction = models['toddler_model'].predict(input_df)
        result = "has ASD" if prediction[0] == 1 else "does not have ASD"
        probability=round(models['toddler_model'].predict_proba(input_df)[0][0]*100,2)
        # Generate counterfactual explanations using DiCE-ML
        d = dice_ml.Data(dataframe=pd.concat([input_df, pd.DataFrame({'Class': [prediction[0]]})], axis=1),
                        continuous_features=['age', 'result'],
                        outcome_name='Class')
        m = dice_ml.Model(model=models['toddler_model'], backend='sklearn')
        exp = dice_ml.Dice(d, m, method='random')  # Revert to random method
        
        query_instance = input_df
        prefix_map = {'gender': 'gender_', 'ethnicity': 'ethnicity', 'jundice': 'jundice_', 'autism': 'autism_'}
        # try:
        cfs = exp.generate_counterfactuals(query_instance,
                                        total_CFs=3,
                                        desired_class="opposite",
                                        features_to_vary=['age', 'result', 
                                                        'A1_Score', 'A2_Score', 'A3_Score',
                                                        'A4_Score', 'A5_Score', 'A6_Score',
                                                        'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score'],
                                        permitted_range={'age': [0, 4], 'result': [0, 100]},
                                        proximity_weight=1.0,
                                        diversity_weight=1.0)
        cf_df = cfs.cf_examples_list[0].final_cfs_df
        if cf_df is not None and not cf_df.empty:
            cf_df[['age', 'result']] = models['toddler_scaler'].inverse_transform(cf_df[['age', 'result']])
            cf_df = decode_one_hot(cf_df, ['gender', 'ethnicity', 'jundice', 'autism'], prefix_map)
            cf_df = cf_df.round(2)
            counterfactuals = cf_df.to_dict(orient='records')
            counterfactual_message = None
        else:
            counterfactuals = []
            counterfactual_message = "No counterfactuals found. Try adjusting the input or model configuration."
        # except Exception as e:
        #     counterfactuals = []
        #     counterfactual_message = f"Error generating counterfactuals: {str(e)}"
        
        return render_template('toddlers.html', 
                              unique_data=ref_data['toddler_unique'], 
                              prediction=result,
                              counterfactuals=counterfactuals,
                              counterfactual_message=counterfactual_message,
                              probability=probability,
                              show_result=True)
    
    return render_template('toddlers.html', unique_data=ref_data['toddler_unique'], show_result=False)

@app.route('/adolescents', methods=['GET', 'POST'])
def adolescents():
    if request.method == 'POST':
        # Get form data
        user_input = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity'],
            'jundice': request.form['jundice'],
            'autism': request.form['autism'],
            'contry_of_res': request.form['contry_of_res'],
            'result': float(request.form['result']),
            'relation': request.form['relation']
        }
        
        # Get AQ-10 scores
        for i in range(1, 11):
            score_key = f'A{i}_Score'
            user_input[score_key] = 1 if request.form.get(score_key) == 'Yes' else 0
        
        # Preprocess input
        input_df = pd.DataFrame([user_input])
        input_df[['age', 'result']] = models['adolescent_minmax'].transform(input_df[['age', 'result']])
        input_df[['age', 'result']] = models['adolescent_standard'].transform(input_df[['age', 'result']])
        input_df = pd.get_dummies(input_df)
        
        # Ensure all columns from training are present
        for col in ref_data['adolescent_columns']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[ref_data['adolescent_columns']]
        # Ensure all data is numeric
        input_df = input_df.astype(float)
        # Debug: Print data types
        print("input_df dtypes:\n", input_df.dtypes)
        input_df = input_df[models['adolescent_model'].feature_names_in_]
        # Make prediction
        prediction = models['adolescent_model'].predict(input_df)
        result = "has ASD" if prediction[0] == 1 else "does not have ASD"
        probability=round(models['adolescent_model'].predict_proba(input_df)[0][0]*100,2)
        
        # Generate counterfactual explanations using DiCE-ML
        d = dice_ml.Data(dataframe=pd.concat([input_df, pd.DataFrame({'Class': [prediction[0]]})], axis=1),
                        continuous_features=['age', 'result'],
                        outcome_name='Class')
        m = dice_ml.Model(model=models['adolescent_model'], backend='sklearn')
        exp = dice_ml.Dice(d, m, method='random')  # Revert to random method
        
        query_instance = input_df
        prefix_map = {'gender': 'gender_', 'ethnicity': 'ethnicity_', 'jundice': 'jundice_', 
              'autism': 'autism_', 'contry_of_res': 'contry_of_res', 'relation': 'relation_'}
        try:
            cfs = exp.generate_counterfactuals(query_instance,
                                            total_CFs=3,
                                            desired_class="opposite")
            cf_df = cfs.cf_examples_list[0].final_cfs_df
            if cf_df is not None and not cf_df.empty:
                # Inverse transform standard scaler
                cf_df[['age', 'result']] = models['adolescent_standard'].inverse_transform(cf_df[['age', 'result']])
                # Inverse transform minmax scaler
                cf_df[['age', 'result']] = models['adolescent_minmax'].inverse_transform(cf_df[['age', 'result']])
                cf_df = decode_one_hot(cf_df, ['gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res', 'relation'], prefix_map)
                cf_df = cf_df.round(2)
                counterfactuals = cf_df.to_dict(orient='records')
                counterfactual_message = None
            else:
                counterfactuals = []
                counterfactual_message = "No counterfactuals found. Try adjusting the input or model configuration."
        except Exception as e:
            counterfactuals = []
            counterfactual_message = f"Error generating counterfactuals: {str(e)}"
        
        return render_template('adolescents.html', 
                              unique_data=ref_data['adolescent_unique'], 
                              prediction=result,
                              counterfactuals=counterfactuals,
                              counterfactual_message=counterfactual_message,
                              probability=probability,
                              show_result=True)
    
    return render_template('adolescents.html', unique_data=ref_data['adolescent_unique'], show_result=False)

if __name__ == '__main__':
    app.run(debug=True)