# Colorectal Cancer Prediction MLOps Project

This project implements an end-to-end Machine Learning pipeline for predicting colorectal cancer survival outcomes. It encompasses data preprocessing, automated feature selection, model training with experiment tracking, and deployment using Docker and Kubeflow Pipelines.

## Project Overview
The primary goal of this repository is to demonstrate a robust MLOps workflow that handles raw medical data, identifies critical health indicators, trains a predictive model, and provides a web-based interface for real-time predictions.

### Key Features
* **Automated Data Pipeline**: Handles cleaning, multi-type encoding (Ordinal, Nominal, Numerical), and feature scaling.
* **Intelligent Feature Selection**: Utilizes a `RandomForestClassifier` to identify the top 5 most impactful features, optimizing the model for deployment.
* **Experiment Tracking**: Integrated with **MLflow** to log metrics like Accuracy, Precision, F1-Score, and ROC-AUC.
* **Orchestration**: Uses **Kubeflow Pipelines (KFP)** to manage the workflow as a series of containerized components.
* **Web Deployment**: A Flask-based application provides a user interface for entering patient data and receiving predictions.

---

## Project Structure
```text
.
├── application.py              # Flask web application
├── Dockerfile                  # Containerization configuration
├── requirements.txt            # Project dependencies
├── setup.py                    # Package installation script
├── src/
│   ├── data_processing.py      # Data cleaning and feature engineering
│   ├── model_training.py       # Model training and MLflow logging
│   ├── logger.py               # Custom logging utility
│   └── custom_exception.py     # Error handling
├── kubeflow.pipeline/
│   └── mlops_pipeline.py       # Kubeflow pipeline definition
└── artifacts/                  # Generated data and model files
```

---

## Technical Details

### 1. Data Processing & Feature Engineering
The `DataProcessing` class in `src/data_processing.py` performs the following:
* **Encoding**: 
    * **Ordinal**: For features with inherent order like `Cancer_Stage` and `Obesity_BMI`.
    * **Nominal**: One-Hot Encoding for features like `Country` and `Gender`.
    * **Numerical**: Standard scaling for continuous variables like `Age` and `Healthcare_Costs`.
* **Feature Selection**: To reduce model complexity, the pipeline calculates feature importance and selects the top 5 features (typically `Healthcare_Costs`, `Tumor_Size_mm`, `Incidence_Rate_per_100K`, `Age`, and `Country`).

### 2. Model Training
The project uses a **Gradient Boosting Classifier** for the final prediction task.
* **Hyperparameters**: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.
* **Logging**: MLflow tracks the entire training process, saving metrics and the final `model.pkl`.

### 3. Pipeline Orchestration
The Kubeflow pipeline (`kubeflow.pipeline/mlops_pipeline.py`) defines two main stages:
1.  **Data Processing Operation**: Runs the preprocessing script and outputs a `Dataset` artifact.
2.  **Model Training Operation**: Takes the processed data and trains the Gradient Boosting model.

Both operations run inside a Docker container using the image `himanshu863/my-mlops-app:latest`.

---

## Setup and Usage

### Installation
1.  Clone the repository.
2.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```
    *(This will also install dependencies from `requirements.txt` such as pandas, scikit-learn, Flask, and MLflow).*

### Running the Web Application
Launch the Flask server to access the prediction UI:
```bash
python application.py
```
The app will be available at `http://0.0.0.0:5000`.

### Containerization
To build and run the project using Docker:
```bash
docker build -t colorectal-cancer-pred .
docker run -p 5000:5000 colorectal-cancer-pred
```

---

## API & Prediction
The Flask app exposes a `/predict` endpoint that expects the following inputs via a POST form:
* `Country`
* `Healthcare_Costs`
* `Tumor_Size_mm`
* `Incidence_Rate_per_100K`
* `Age`

It returns a survival prediction based on the trained model.