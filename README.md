# Heart Failure Mortality Prediction & Streamlit Web App

## Project Overview
This is an end-to-end Machine Learning Capstone Project focused on predicting mortality risk in heart failure patients based on clinical records. The goal of this project is not just to build an accurate model, but to create an **interpretable, deployable clinical tool** that medical practitioners can use for real-time risk assessment.

The project encompasses the entire data science lifecycle: Exploratory Data Analysis (EDA), Data Pre-processing, Handling Imbalanced Data, Model Optimization, and Web Deployment.

## Tech Stack & Libraries
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Decision Tree, GridSearchCV)
* **Data Resampling:** `imbalanced-learn` (SMOTE)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Deployment:** Streamlit

## The Machine Learning Pipeline
1. **Exploratory Data Analysis:** Identified class imbalance (67.9% survived vs. 32.1% deceased) and analyzed feature distributions. Maintained clinical outliers as they represent critical patient conditions.
2. **Handling Imbalanced Data:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the training set to prevent data leakage and bias toward the majority class.
3. **Model Training & Tuning:** 
   * The baseline Decision Tree model suffered from severe overfitting (near 100% training accuracy, but poor test generalization).
   * Applied **GridSearchCV** (with 3-fold Cross-Validation) to tune hyperparameters (`max_depth`, `min_samples_split`, `criterion`), effectively curing the overfitting issue.
4. **Model Interpretability:** Extracted feature importance, revealing that `time` (follow-up period), `serum_creatinine`, and `ejection_fraction` were the most critical predictors of mortality, aligning perfectly with medical domain knowledge.

## Key Results
* **Overall Accuracy:** Increased to **86.67%**.
* **Minority Class Recall:** Increased from 60% to **70%**. By reducing False Negatives, the optimized model is significantly safer and more reliable for clinical mortality prediction.

## Web Application Deployment (Streamlit)
To bridge the gap between data science and clinical application, the optimized model was exported (`.joblib`) and deployed using **Streamlit**. 

**App Features:**
* **Interactive Input Form:** Allows users to input 12 clinical features with predefined medical boundaries.
* **Real-time Prediction:** Instant mortality risk classification.
* **Interactive Dashboards (Plotly):** Displays a Gauge Chart for prediction confidence levels and a Bar Chart for probability distributions.

## Repository Structure
* `notebook.ipynb`: The Google Colab notebook containing data exploration, SMOTE, and model training.
* `app.py`: The Streamlit web application script.
* `model.joblib`: The exported optimized Decision Tree model.
* `README.md`: Project documentation.
