from src.custom_exception import CustomException
from src.logger import get_logger
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow 
import mlflow.sklearn


logger=get_logger(__name__)

class ModelTraining:
    def __init__(self,processed_data_path="artifacts/processed"):
        self.processed_data_path=processed_data_path
        self.model_dir="artifacts/models"
        os.makedirs(self.model_dir,exist_ok= True)

        logger.info("Model Training Intialization")

    def load_data(self):
        try:
            self.X_train_transformed=joblib.load(os.path.join(self.processed_data_path,"X_train_transformed.pkl"))
            self.X_test_transformed=joblib.load(os.path.join(self.processed_data_path,"X_test_transformed.pkl"))
            self.y_train=joblib.load(os.path.join(self.processed_data_path,"y_train.pkl"))
            self.y_test=joblib.load(os.path.join(self.processed_data_path,"y_test.pkl"))

            # self.preprocessor=joblib.load("artifacts/processed/preprocessor.pkl")

            logger.info("Data Loaded for Model")

        except Exception as e:
            logger.error("Error While loading data ")
            raise CustomException("Failed to load data")
        
    def train_model(self):
        try:
            self.model=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1 , max_depth=3, random_state=42)
            self.model.fit(self.X_train_transformed,self.y_train)

            joblib.dump(self.model, os.path.join(self.model_dir,"model.pkl"))

            logger.info("Model Trained and saved")

        except Exception as e:
            logger.error("Error While Training model ")
            raise CustomException("Failed to train model")
        
    def evaluate_model(self):
        try:
            y_pred=self.model.predict(self.X_test_transformed)

            accuracy=accuracy_score(self.y_test,y_pred)
            precision=precision_score(self.y_test,y_pred)
            f1=f1_score(self.y_test,y_pred)
            recall=recall_score(self.y_test,y_pred)

            logger.info(f"accuracy: {accuracy}, Precision: {precision}, f1: {f1}, Recall: {recall}")

            mlflow.log_metric("Accuracy",accuracy)
            mlflow.log_metric("Precision",precision)
            mlflow.log_metric("f1",f1)
            mlflow.log_metric("Recall",recall)

            y_prob = self.model.predict_proba(self.X_test_transformed)[:,1]
            roc_auc = roc_auc_score(self.y_test, y_prob)
            logger.info(f"ROC_AUC: {roc_auc}")

            mlflow.log_metric("ROC-AUC",roc_auc)
            
            logger.info("Model Evaulation Done")



        except Exception as e:
            logger.error("Error While Evaluating model ")
            raise CustomException("Failed to Evaluate model")

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__=="__main__":
    with mlflow.start_run():
        model_train=ModelTraining()
        model_train.run()