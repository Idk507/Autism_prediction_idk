import streamlit as st
import pandas as pd
import joblib

# Load the saved pipelines
pipelines = joblib.load('autism_detection_pipelines.pkl')
imputer = joblib.load('autism_imputer.pkl')
# Create a Streamlit web application
st.title("Autism Detection App")
st.write("Enter your information to get a prediction on whether you have Autism or not:")

training_columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'result',
       'gender_m', 'ethnicity_South Asian', 'ethnicity_Asian',
       'ethnicity_Black', 'ethnicity_Hispanic', 'ethnicity_Latino',
       'ethnicity_Others', 'ethnicity_Pasifika', 'ethnicity_Turkish',
       'ethnicity_White-European', 'jundice_yes', 'austim_yes',
       'contry_of_res_Isle of Man', 'contry_of_res_New Zealand',
       'contry_of_res_Saudi Arabia', 'contry_of_res_South Africa',
       'contry_of_res_South Korea', 'contry_of_res_U.S. Outlying Islands',
       'contry_of_res_United Arab Emirates',
       'contry_of_res_United Kingdom', 'contry_of_res_United States',
       'contry_of_res_Afghanistan', 'contry_of_res_Argentina',
       'contry_of_res_Armenia', 'contry_of_res_Australia',
       'contry_of_res_Austria', 'contry_of_res_Bahrain',
       'contry_of_res_Bangladesh', 'contry_of_res_Bhutan',
       'contry_of_res_Brazil', 'contry_of_res_Bulgaria',
       'contry_of_res_Canada', 'contry_of_res_China', 'contry_of_res_Egypt',
       'contry_of_res_Europe', 'contry_of_res_Georgia',
       'contry_of_res_Germany', 'contry_of_res_Ghana', 'contry_of_res_India',
       'contry_of_res_Iraq', 'contry_of_res_Ireland', 'contry_of_res_Italy',
       'contry_of_res_Japan', 'contry_of_res_Jordan', 'contry_of_res_Kuwait',
       'contry_of_res_Latvia', 'contry_of_res_Lebanon', 'contry_of_res_Libya',
       'contry_of_res_Malaysia', 'contry_of_res_Malta', 'contry_of_res_Mexico',
       'contry_of_res_Nepal', 'contry_of_res_Netherlands',
       'contry_of_res_Nigeria', 'contry_of_res_Oman', 'contry_of_res_Pakistan',
       'contry_of_res_Philippines', 'contry_of_res_Qatar',
       'contry_of_res_Romania', 'contry_of_res_Russia', 'contry_of_res_Sweden',
       'contry_of_res_Syria', 'contry_of_res_Turkey', 'used_app_before_yes',
       'relation_Parent', 'relation_Relative', 'relation_Self',
       'relation_self']
# Input fields
a1_score = st.number_input("Have difficulties in understanding other people's feelings ever been a concern for you?", value=0)
a2_score = st.number_input(" Do you often prefer to stick to your routines and get upset if they are disrupted?", value=0)
a3_score = st.number_input("Are you comfortable with making small talk and initiating conversations with others?", value=0)
a4_score = st.number_input(" Do you find it challenging to interpret facial expressions and body language?", value=0)
a5_score = st.number_input(" Are you sensitive to certain textures or sounds that others might not notice?", value=0)
a6_score = st.number_input(" Have you ever been told that you talk about the same topics excessively?", value=0)
a7_score = st.number_input("Do you struggle to understand sarcasm and jokes that are not literal?", value=0)
a8_score = st.number_input("Do you tend to avoid eye contact when talking to others?", value=0)
a9_score = st.number_input("Are you comfortable in crowded or noisy environments?", value=0)
a10_score = st.number_input("Have you ever been diagnosed with Autism Spectrum Disorder or a related condition?", value=0)
age = st.text_input("Age", "")
gender = st.selectbox("Gender", ["male", "female"])
ethnicity = st.text_input("Ethnicity", "")
jundice = st.text_input("Jundice", "")
austim = st.text_input("Austim", "")
contry_of_res = st.text_input("Country of Residence", "")
used_app_before = st.text_input("Used App Before", "")
result = st.number_input("Result", value=0)
age_desc = st.text_input("Age Description", "")
relation = st.text_input("Relation", "")


classifier_name = st.selectbox("Select Classifier", [name for name, _ in pipelines])

# Predict button
if st.button("Predict"):
    selected_pipeline = dict(pipelines)[classifier_name]
    input_data = pd.DataFrame({
        'A1_Score': [a1_score],
        'A2_Score': [a2_score],
        'A3_Score': [a3_score],
        'A4_Score': [a4_score],
        'A5_Score': [a5_score],
        'A6_Score': [a6_score],
        'A7_Score': [a7_score],
        'A8_Score': [a8_score],
        'A9_Score': [a9_score],
        'A10_Score': [a10_score],
        'age': [age],
        'gender': [gender],
        'ethnicity': [ethnicity],
        'jundice': [jundice],
        'austim': [austim],
        'contry_of_res': [contry_of_res],
        'used_app_before': [used_app_before],
        'result': [result],
        'age_desc': [age_desc],
        'relation': [relation]
    })
    categorical_features = ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation']
    input_data[categorical_features] = input_data[categorical_features].astype(str)
    
     # Use the same one-hot encoding as in the training data
    encoded_input = pd.get_dummies(input_data)
    
    # Reindex columns to match the training data columns
    missing_columns = set(training_columns) - set(encoded_input.columns)
    for column in missing_columns:
        encoded_input[column] = 0
    encoded_input = encoded_input[training_columns]
    
    # Impute missing values using the imputer from the pipeline
    input_imputed = imputer.transform(encoded_input)
    
    # Predict using the classifier from the pipeline
    prediction = selected_pipeline.named_steps['classifier'].predict(input_imputed)
    st.write("Prediction:", prediction[0])