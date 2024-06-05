import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from datavisualization import visualize_data

def engineer_features():
    data, categorical_features, numerical_features = visualize_data()

    # Initialize a LabelEncoder
    label_encoders = {}
    
    # Convert categorical feature values to numerical values using LabelEncoder
    for categorical_feature in categorical_features:
        le = LabelEncoder()
        data[categorical_feature] = le.fit_transform(data[categorical_feature])
        label_encoders[categorical_feature] = le  # Store the label encoder if needed later
        unique_values = le.classes_
        print(f"Unique values in {categorical_feature}: {unique_values}")

    print(data.head())

    data.to_csv('credit_card_eligibility_cleansed_data.csv', index=False)
    
    return data

engineer_features()
