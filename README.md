## AI-Driven Cloud-IoT Healthcare Ecosystem 🏥 ☁️ ###
## MSc Dissertation Project - University of the West of Scotland (UWS) ##

## Project Title:Architecting AI-Driven Cloud-IOT Ecosystems for Global Healthcare: A Multi Disease Predictive Analytics Approach to Clinical Decision Support and Strategic Management. ##
📌 Project Overview

This project addresses the "Regulatory Paradox" in healthcare AI by deploying a high-fidelity inference engine that predicts patient mortality risk with 98.8% AUC-ROC accuracy. The system integrates Gradient Boosting Machines (XGBoost & LightGBM) into a cloud-native dashboard, providing real-time clinical decision support (CDS) based on a 61-feature clinical vector.
🚀 Key Features

    Predictive Analytics: Dual-model ensemble (XGBoost/LightGBM) optimized via 5-Fold Stratified Cross-Validation.

    Interpretability Layer: Full integration of SHAP (Shapley Additive Explanations) to provide transparent, feature-level insights for clinicians.

    Real-time Dashboard: Streamlit-powered interface for high-velocity telemetry visualization and patient risk stratification.

    Ethical Auditing: Automated fairness checks using Disparate Impact (DI) Ratios to ensure equitable healthcare delivery across demographics.

    Outlier Detection: Isolation Forest implementation to identify anomalous clinical records that may indicate sensor failure or unique medical emergencies.

## 🛠️ Tech Stack ##

    Language: Python 3.9+

    AI/ML: Scikit-learn, XGBoost, LightGBM, SHAP, Imbalanced-learn.

    Deployment: Streamlit Cloud / Azure IoT Integration.

    Data Processing: Pandas, NumPy, Joblib (for model persistence).

    Research Framework: Scoping Study Methodology.

## 📂 Repository Structure ##

    B01796218_Healthcare_Dashboard.py: The primary Streamlit application file.

    B01796218_DarpanFinal_Thesis_UWS.ipynb: Full research pipeline including data synthesis, model training, and SHAP analysis.

    models/: Directory containing serialized (.pkl or .joblib) model files.

## 🔧 Installation & Setup ##

    Clone the repository:https://github.com/DarpanN/B01796218_Healthcare_multi_disease_analytics_uws.git
    
## Install dependencies:  ##

# --- CORE DATA HANDLING (LATEST STABLE) ---
numpy>=2.1.0        # Support for new vectorization standards
pandas>=2.2.0       # Enhanced Arrow-backed performance

# --- VISUALIZATION ---
matplotlib>=3.10.0
seaborn>=0.13.0

# --- MACHINE LEARNING FRAMEWORKS ---
scikit-learn>=1.5.0 # Required for advanced IterativeImputer stability
xgboost>=2.1.0      # GPU-accelerated and memory-optimized
lightgbm>=4.4.0     # Improved leaf-wise growth algorithms

# --- EXPLAINABLE AI (XAI) ---
shap>=0.46.0        # Optimized TreeSHAP for GBDT v2.0+
lime>=0.2.0.1       # Stable local interpretability

# --- DASHBOARD & CLOUD DEPLOYMENT ---
streamlit>=1.40.0   # Enhanced state management for clinical dashboards

# --- UTILITIES ---
joblib>=1.4.0
scipy>=1.13.0

## Run the Dashboard: streamlit run B01796218_Healthcare_Dashboard.py

## ⚖️ License

## Author: DarpanN (B01796218) ##
## Institution: University of the West of Scotland (UWS) ##

    
