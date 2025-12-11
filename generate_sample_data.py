"""
Sample Depression Dataset Generator
This script generates a synthetic dataset for testing the depression prediction app
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate synthetic data
data = {
    'Age': np.random.randint(18, 70, n_samples),
    'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples, p=[0.48, 0.48, 0.04]),
    'Sleep_Hours': np.random.uniform(4, 10, n_samples).round(1),
    'Work_Hours': np.random.randint(20, 80, n_samples),
    'Physical_Activity': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
    'Social_Support': np.random.randint(1, 11, n_samples),  # Scale 1-10
    'Stress_Level': np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_samples, p=[0.2, 0.3, 0.35, 0.15]),
    'Anxiety_Score': np.random.randint(0, 21, n_samples),  # GAD-7 scale (0-21)
    'Work_Satisfaction': np.random.randint(1, 11, n_samples),  # Scale 1-10
    'Relationship_Status': np.random.choice(['Single', 'Relationship', 'Married', 'Divorced'], n_samples, p=[0.3, 0.25, 0.35, 0.1]),
    'Financial_Stress': np.random.randint(1, 11, n_samples),  # Scale 1-10
    'Chronic_Illness': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
    'Family_History': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
    'Therapy_History': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'Medication': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
    'Screen_Time': np.random.uniform(2, 14, n_samples).round(1),  # Hours per day
    'Alcohol_Consumption': np.random.choice(['None', 'Occasional', 'Moderate', 'Heavy'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    'Diet_Quality': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples, p=[0.15, 0.35, 0.35, 0.15]),
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate Depression levels based on certain factors (with some randomness)
def calculate_depression(row):
    """
    Calculate depression level based on various factors
    This is a simplified model for synthetic data generation
    """
    score = 0
    
    # Sleep hours (poor sleep increases depression)
    if row['Sleep_Hours'] < 6:
        score += 2
    elif row['Sleep_Hours'] < 7:
        score += 1
    
    # Stress level
    stress_scores = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
    score += stress_scores.get(row['Stress_Level'], 0)
    
    # Anxiety score (normalized)
    score += row['Anxiety_Score'] / 7
    
    # Social support (lack of support increases depression)
    if row['Social_Support'] < 4:
        score += 2
    elif row['Social_Support'] < 6:
        score += 1
    
    # Work satisfaction (low satisfaction increases depression)
    if row['Work_Satisfaction'] < 4:
        score += 2
    elif row['Work_Satisfaction'] < 6:
        score += 1
    
    # Financial stress
    if row['Financial_Stress'] > 7:
        score += 2
    elif row['Financial_Stress'] > 5:
        score += 1
    
    # Physical activity (lack of activity increases depression)
    activity_scores = {'None': 2, 'Light': 1, 'Moderate': 0, 'Heavy': -1}
    score += activity_scores.get(row['Physical_Activity'], 0)
    
    # Family history
    if row['Family_History'] == 'Yes':
        score += 1
    
    # Chronic illness
    if row['Chronic_Illness'] == 'Yes':
        score += 1
    
    # Screen time (excessive screen time)
    if row['Screen_Time'] > 10:
        score += 1
    
    # Add some randomness
    score += np.random.uniform(-1, 1)
    
    # Categorize depression level based on score
    if score < 3:
        return 'Minimal'
    elif score < 6:
        return 'Mild'
    elif score < 9:
        return 'Moderate'
    elif score < 12:
        return 'Moderately Severe'
    else:
        return 'Severe'

# Apply depression calculation
df['Depression'] = df.apply(calculate_depression, axis=1)

# Add some missing values randomly (5% of data)
missing_cols = ['Sleep_Hours', 'Social_Support', 'Work_Satisfaction', 'Anxiety_Score']
for col in missing_cols:
    mask = np.random.random(n_samples) < 0.05
    df.loc[mask, col] = np.nan

# Display dataset info
print("=" * 60)
print("Sample Depression Dataset Generated Successfully!")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples: {len(df)}")
print(f"\nDepression Level Distribution:")
print(df['Depression'].value_counts().sort_index())
print(f"\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\n" + "=" * 60)
print("Dataset Preview:")
print("=" * 60)
print(df.head(10))

# Save to CSV
filename = 'depression_dataset.csv'
df.to_csv(filename, index=False)
print(f"\n✅ Dataset saved as '{filename}'")

# Display column information
print("\n" + "=" * 60)
print("Column Information:")
print("=" * 60)
for col in df.columns:
    print(f"- {col}: {df[col].dtype}")

print("\n" + "=" * 60)
print("Statistical Summary:")
print("=" * 60)
print(df.describe())

print("\n🎉 You can now use this dataset in the Depression Predictor App!")
print(f"📁 File location: ./{filename}")
print("\n📝 To use in the app:")
print("   1. Run: streamlit run app.py")
print(f"   2. Upload the file: {filename}")
print("   3. Or use the absolute path as URL")