import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    
    # If use case is used for clustering, drop Target column
    # data.drop(['Target'], axis=1, inplace=True)

    # If use case is used for classification, balance Target classes
    '''
    # Upsample the minority class to address class imbalance
    data_majority = data[data['Target'] == 0]
    data_minority = data[data['Target'] == 1]
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=len(data_majority), random_state=1)
    data = pd.concat([data_majority, data_minority_upsampled], axis=0)

    # Display the class distribution after upsampling
    print("Class distribution after upsampling:")
    print(data['Target'].value_counts())
    '''
    
    print(data.head())

    data.to_csv('credit_card_eligibility_cleansed_data.csv', index=False)
    
    return data

engineer_features()
