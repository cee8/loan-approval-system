# src/explain_decision.py

import shap
import pandas as pd
import numpy as np
import torch
from src.model_training import LoanApprovalModel
from src.data_preprocessing import preprocess_data

def model_predict(data, model):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(data_tensor).numpy()
    return outputs

def explain_decision(model, data, instance_idx, scaler, feature_names):
    data_scaled = scaler.transform(data.values)
    
    explainer = shap.KernelExplainer(lambda x: model_predict(x, model), data_scaled)
    
    shap_values = explainer.shap_values(data_scaled[instance_idx:instance_idx+1])
    
    print(f"Data shape: {data.shape}")
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    print(f"Data instance shape: {data.iloc[instance_idx].shape}")
    
    shap.initjs()
    
    base_value = explainer.expected_value[0]
    instance_shap_values = np.array(shap_values[0]).flatten()
    instance_data = data.iloc[instance_idx].values.flatten()
    
    print(f"Expected Value: {base_value}")
    print(f"SHAP Values for instance: {instance_shap_values}")
    print(f"Instance data: {instance_data}")
    
    force_plot = shap.force_plot(base_value, instance_shap_values, instance_data, feature_names=feature_names)

    shap.save_html("shap_force_plot.html", force_plot)
    
    return force_plot

if __name__ == "__main__":
    data = preprocess_data('../data/loan_data.csv')
    feature_names = data.drop('loan_status', axis=1).columns.tolist()
    instance_idx = 0
    model = LoanApprovalModel(data.shape[1]-1)
    model.load_state_dict(torch.load('../model/loan_approval_model.pth'))
    model.eval()
    scaler = torch.load('../model/scaler.pth')
    explanation = explain_decision(model, data.drop('loan_status', axis=1), instance_idx, scaler, feature_names)
    print(explanation)

