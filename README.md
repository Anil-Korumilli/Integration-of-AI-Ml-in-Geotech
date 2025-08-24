# Integration-of-AI-Ml-in-Geotech

## 🔹 Project Title

Machine Learning Model Evaluation and Interactive GUI for Prediction

## 🔹 Overview

This project demonstrates the development of machine learning models for predicting engineering parameters using a structured dataset. It includes two major components:

Model Development (ML Project.ipynb) – Training, evaluation, and comparison of multiple machine learning algorithms.

Interactive GUI (INTERACTIVE_GUI.ipynb) – A user-friendly interface built with Tkinter that allows end-users to input values, evaluate trained models, and generate predictions without deep coding knowledge.

The dataset is provided in ML_Project Excel Data.xlsx and includes multiple input parameters with one output parameter for supervised learning.

## 🔹 Tech Stack

Data Source: Kaggle dataset (Excel format).

# Tools Used:

Excel → Data storage and raw dataset.

Jupyter Notebook → Model development & experimentation.

# Libraries & Frameworks:

Pandas, NumPy → Data cleaning & preprocessing.

Matplotlib, Seaborn → Data visualization & exploratory analysis.

Scikit-learn → Machine learning model development & evaluation.

XGBoost → Advanced gradient boosting model.

SHAP → Feature interpretability (optional).

Tkinter → GUI for interactive model predictions.

## 🔹 Features

✔️ Preprocessing and handling of dataset from Excel.

✔️ Training and comparison of various ML models such as:

  1.Random Forest

  2.XGBoost

  3.Support Vector Regression (SVR)

  4.K-Nearest Neighbors (KNN)

  5.Linear Regression

✔️ Hyperparameter tuning for performance optimization.

✔️ Model performance evaluation using metrics such as R² Score, MAE, RMSE.

✔️ SHAP-based interpretability for feature importance (optional).

✔️ A Tkinter-based interactive GUI to:

Input new data

Select model for prediction

Display results instantly

# 🔹 Usage
1. Running the ML Models

Load the dataset.

Train and evaluate models.

Compare model performance.

2. Running the Interactive GUI

Open INTERACTIVE_GUI.ipynb and run all cells.

Enter input parameters in the Tkinter window.

Select a trained ML model.

Click Predict to get the output instantly.

# 🔹 Results

Achieved a maximum R² score of ~0.98 with XGBoost (depending on dataset).

GUI enables real-time predictions with user-provided input.

Provides interpretability using SHAP plots for feature influence.

# 🔹 Future Enhancements

Add more advanced models (Neural Networks, Ensemble Learning).

Deploy GUI as a standalone desktop app or web-based app.

Integrate automatic feature selection.

Enhance visualization for better interpretability.

# 🔹SNAPSHOTS 

## 📊  Upload Dataset Interface

![Model Performance](https://github.com/Anil-Korumilli/Integration-of-AI-Ml-in-Geotech/blob/main/Snapshot%20of%20uploading%20data.png)

## Model Comparison Results (Metrics Table)   

![Model Performance]

## Model Comparison (Bar Chart)  

![Model Performance]

## Model Performance Heatmap  

![Model Performance]

## Manual Prediction GUI  

![Model Performance]
