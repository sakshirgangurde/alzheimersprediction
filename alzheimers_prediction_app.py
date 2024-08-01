import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the model
model_path = 'random_forest_model.pkl'
model = joblib.load(model_path)

# Function to make predictions
def predict_diagnosis(input_data):
    # Define the correct order of features as expected by the model
    feature_order = [
        'Age', 'Gender', 'BMI', 'Smoking', 'AlcoholConsumption',
        'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
        'MemoryComplaints', 'BehavioralProblems', 'Confusion', 
        'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 
        'Forgetfulness'
    ]
    
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data, index=[0])
    
    # Ensure the DataFrame has the correct order of features
    df = df[feature_order]
    
    # Make prediction
    prediction = model.predict(df)
    return prediction[0]

# Streamlit app
st.title('Alzheimer\'s Disease Prediction App')

st.header('Please provide the following information:')

# Collect user inputs
age = st.number_input('Age', min_value=60, max_value=90, value=75)
gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
bmi = st.number_input('BMI', min_value=15.0, max_value=40.0, value=25.0)
smoking = st.selectbox('Smoking', [0, 1])
alcohol_consumption = st.number_input('Alcohol Consumption', min_value=0.0, max_value=20.0, value=10.0, format='%f')
sleep_quality = st.number_input('Sleep Quality', min_value=0.0, max_value=10.0, value=7.0)
family_history_alzheimers = st.selectbox('Family History of Alzheimer\'s', [0, 1])
cardiovascular_disease = st.selectbox('Cardiovascular Disease', [0, 1])
diabetes = st.selectbox('Diabetes', [0, 1])
depression = st.selectbox('Depression', [0, 1])
head_injury = st.selectbox('Head Injury', [0, 1])
hypertension = st.selectbox('Hypertension', [0, 1])
memory_complaints = st.selectbox('Memory Complaints', [0, 1])
behavioral_problems = st.selectbox('Behavioral Problems', [0, 1])
confusion = st.selectbox('Confusion', [0, 1])
disorientation = st.selectbox('Disorientation', [0, 1])
personality_changes = st.selectbox('Personality Changes', [0, 1])
difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', [0, 1])
forgetfulness = st.selectbox('Forgetfulness', [0, 1])

# Input data dictionary
input_data = {
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'Smoking': smoking,
    'AlcoholConsumption': alcohol_consumption,
    'SleepQuality': sleep_quality,
    'FamilyHistoryAlzheimers': family_history_alzheimers,
    'CardiovascularDisease': cardiovascular_disease,
    'Diabetes': diabetes,
    'Depression': depression,
    'HeadInjury': head_injury,
    'Hypertension': hypertension,
    'MemoryComplaints': memory_complaints,
    'BehavioralProblems': behavioral_problems,
    'Confusion': confusion,
    'Disorientation': disorientation,
    'PersonalityChanges': personality_changes,
    'DifficultyCompletingTasks': difficulty_completing_tasks,
    'Forgetfulness': forgetfulness
}

# Make prediction and display result
if st.button('Predict Diagnosis'):
    try:
        diagnosis = predict_diagnosis(input_data)
        if diagnosis == 1:
            st.error('The model predicts that the person will be diagnosed with Alzheimer\'s.')
        else:
            st.success('The model predicts that the person will not be diagnosed with Alzheimer\'s.')
    except ValueError as e:
        st.error(f"An error occurred: {e}")
