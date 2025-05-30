
# 🧠 Early-Stage Detection of Autism Spectrum Disorders (ASD) Using Machine Learning

This project was developed as part of our curriculum requirement at *SASTRA University. It focuses on using machine learning models to detect early signs of Autism Spectrum Disorder (ASD) across various age groups. A user-friendly **Streamlit interface* was developed to enable interactive predictions.

## 🎯 Objective

* Predict early-stage ASD symptoms in individuals across *4 age groups*.
* Compare performance of multiple models and select the best one based on:

  * *Accuracy*
  * *Training Time*
  * *Model Size*
* Provide real-time prediction via an interactive *Streamlit-based web app*.


## 📊 Datasets Used

| Dataset    | Filename         | Age Group      |
| ---------- | ---------------- | -------------- |
| Toddler    | Toddlers.csv   | 1.5 to 3 years |
| Child      | Child.csv      | 4 to 11 years  |
| Adolescent | Adolescent.csv | 12 to 17 years |
| Adult      | Adults.csv     | 18+ years      |



## ⚙ Model Overview

* *Algorithms Evaluated*: Logistic Regression, SVM, Random Forest, XGBoost, KNN
* *Evaluation Criteria*:

  * Classification Accuracy
  * F1 Score
  * Time & Space Complexity
* *Tools Used*: scikit-learn, xgboost, pandas, Streamlit



## 🌐 Web App Interface

A *Streamlit-based web application* was developed to:

* Accept user inputs based on age group
* Display the best-performing model's prediction
* Provide an intuitive, real-time prediction experience for users and clinicians

To run the app:

bash
streamlit run app.py
