# src/explain_decision.py

import shap
import pandas as pd
import torch
from model_training import LoanApprovalModel, preprocess_data

def explain_decision(model, data, instance_idx, scaler):
    explainer = shap.DeepExplainer(model, torch.tensor(scaler.transform(data.values), dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(scaler.transform(data.values), dtype=torch.float32))
    shap.initjs()
    return shap.force_plot(explainer.expected_value[0].item(), shap_values[0][instance_idx], data.iloc[instance_idx])

if __name__ == "__main__":
    data = preprocess_data('../data/loan_data.csv')
    instance_idx = 0
    model = LoanApprovalModel(data.shape[1]-1)
    model.load_state_dict(torch.load('../model/loan_approval_model.pth'))
    model.eval()
    scaler = torch.load('../model/scaler.pth')
    explanation = explain_decision(model, data.drop('loan_status', axis=1), instance_idx, scaler)
    explanation

