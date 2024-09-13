import pandas as pd
import numpy as np

# Parameters
num_samples = 100000
np.random.seed(42)

# Possible values for categorical variables
sex_options = ['Male', 'Female']
binary_options = ['Yes', 'No']

# Generate random data for the parameters
ages = np.random.randint(18, 90, num_samples)
sex = np.random.choice(sex_options, num_samples)
diabetes_history = np.random.choice(binary_options, num_samples)
weights = np.round(np.random.uniform(50, 120, num_samples), 1)
bmi = np.round(np.random.uniform(18, 35, num_samples), 1)
hypertension_history = np.random.choice(binary_options, num_samples)
thyroid_history = np.random.choice(binary_options, num_samples)

# Initialize dataframe
data = pd.DataFrame({
    'Age': ages,
    'Sex': sex,
    'Diabetes History': diabetes_history,
    'Weight': weights,
    'BMI': bmi,
    'Hypertension History': hypertension_history,
    'Thyroid History': thyroid_history
})

# Symptoms list
symptoms = [
    'Fever', 'Cough', 'Shortness of breath', 'Chest pain', 'Fatigue', 'Headache', 'Dizziness',
    'Nausea', 'Vomiting', 'Diarrhea', 'Abdominal pain', 'Back pain', 'Muscle pain', 'Joint pain',
    'Sore throat', 'Rash', 'Palpitations', 'Edema (swelling)', 'Weight loss', 'Weight gain',
    'Loss of appetite', 'Insomnia', 'Anxiety', 'Depression', 'Confusion', 'Blurred vision',
    'Urinary frequency', 'Urinary urgency', 'Hematuria', 'Syncope'
]

# Define severity scores for each symptom
severity_scores = {
    'Fever': 3, 'Cough': 2, 'Shortness of breath': 5, 'Chest pain': 5, 'Fatigue': 3, 'Headache': 2,
    'Dizziness': 3, 'Nausea': 2, 'Vomiting': 3, 'Diarrhea': 2, 'Abdominal pain': 4, 'Back pain': 3,
    'Muscle pain': 3, 'Joint pain': 2, 'Sore throat': 1, 'Rash': 1, 'Palpitations': 4, 'Edema (swelling)': 3,
    'Weight loss': 4, 'Weight gain': 3, 'Loss of appetite': 3, 'Insomnia': 2, 'Anxiety': 2, 'Depression': 4,
    'Confusion': 4, 'Blurred vision': 3, 'Urinary frequency': 2, 'Urinary urgency': 2, 'Hematuria': 4, 'Syncope': 5
}

# Generate base symptom ratings
base_ratings = np.random.randint(1, 4, (num_samples, len(symptoms)))

# Adjust ratings based on parameters
for i in range(num_samples):
    age = data.loc[i, 'Age']
    diabetes = data.loc[i, 'Diabetes History']
    hypertension = data.loc[i, 'Hypertension History']
    thyroid = data.loc[i, 'Thyroid History']

    # Age adjustments
    if age > 60:
        base_ratings[i, symptoms.index('Fatigue')] += 1
        base_ratings[i, symptoms.index('Joint pain')] += 1
        base_ratings[i, symptoms.index('Dizziness')] += 1

    # Diabetes adjustments
    if diabetes == 'Yes':
        base_ratings[i, symptoms.index('Fatigue')] += 1
        base_ratings[i, symptoms.index('Blurred vision')] += 1
        base_ratings[i, symptoms.index('Urinary frequency')] += 1

    # Hypertension adjustments
    if hypertension == 'Yes':
        base_ratings[i, symptoms.index('Headache')] += 1
        base_ratings[i, symptoms.index('Chest pain')] += 1
        base_ratings[i, symptoms.index('Shortness of breath')] += 1

    # Thyroid adjustments
    if thyroid == 'Yes':
        base_ratings[i, symptoms.index('Weight gain')] += 1
        base_ratings[i, symptoms.index('Fatigue')] += 1
        base_ratings[i, symptoms.index('Depression')] += 1

# Clip ratings to ensure they are within 1-5 range
base_ratings = np.clip(base_ratings, 1, 5)

# Add symptom ratings to dataframe
for idx, symptom in enumerate(symptoms):
    data[symptom] = base_ratings[:, idx]

# Calculate criticality score for each patient
data['Criticality'] = 0
for symptom in symptoms:
    data['Criticality'] += data[symptom] * severity_scores[symptom]

# Normalize the criticality score by dividing by the number of symptoms
num_symptoms = len(symptoms)
data['Normalized Criticality'] = data['Criticality'] / num_symptoms

# Save to CSV
data.to_csv('realistic_symptoms_dataset_with_criticality_normalized.csv', index=False)

print("Dataset generated and saved to 'realistic_symptoms_dataset_with_criticality_normalized.csv'")
