# app.py

from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.explain_decision import explain_decision
import torch

def main():
    data_path = 'data/loan_data.csv'
    data = preprocess_data(data_path)
    
    # Train model
    model, scaler = train_model(data)
    
    # Save model and scaler
    torch.save(model.state_dict(), 'model/loan_approval_model.pth')
    torch.save(scaler, 'model/scaler.pth')

    # Feature names for SHAP explanations
    feature_names = data.drop('loan_status', axis=1).columns.tolist()
    
    # Explain a decision for the first instance
    instance_idx = 0
    explanation = explain_decision(model, data.drop('loan_status', axis=1), instance_idx, scaler, feature_names)
    print(explanation)

if __name__ == "__main__":
    main()

