import pandas as pd
from data_analysis import analyze_data

def preprocess_data():
  data, categorical_features, numerical_features = analyze_data()
  
  data.drop(['ID', 'Num_children', 'Income_type', 'Work_phone', 'Email'], axis=1, inplace=True)
  data['Age'] = data['Age'].astype(int)
  data['Total_income'] = data['Total_income'].astype(int)
  data['Years_employed'] = data['Years_employed'].astype(int)
  
  print(data.head()) 
  categorical_features = data.select_dtypes("object").columns
  numerical_features = data.select_dtypes("number").columns
  
  return data, categorical_features, numerical_features

# preprocess_data()
