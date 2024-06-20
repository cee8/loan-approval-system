# src/audit_system.py

import pandas as pd
from sklearn.metrics import classification_report
import torch
from src.model_training import LoanApprovalModel
from src.data_preprocessing import preprocess_data

def audit_model(model, data, scaler):
    X = data.drop('loan_status', axis=1).values
    y = data['loan_status'].values

    X = torch.tensor(scaler.transform(X), dtype=torch.float32)
    y_pred = model(X).detach().numpy().round()

    report = classification_report(y, y_pred, target_names=['Rejected', 'Approved'])
    print(report)

if __name__ == "__main__":
    data = preprocess_data('../data/loan_data.csv')
    model = LoanApprovalModel(data.shape[1]-1)
    model.load_state_dict(torch.load('../model/loan_approval_model.pth'))
    model.eval()
    scaler = torch.load('../model/scaler.pth')
    audit_model(model, data, scaler)

