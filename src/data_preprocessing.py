# src/data_preprocessing.py

import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Example preprocessing steps
    data['age'] = data['age'].fillna(data['age'].mean())
    data['income'] = data['income'].fillna(data['income'].mean())
    # Additional preprocessing steps can be added here
    return data

if __name__ == "__main__":
    data = preprocess_data('../data/loan_data.csv')
    print(data.head())

