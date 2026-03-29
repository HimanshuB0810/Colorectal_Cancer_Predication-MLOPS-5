from src.custom_exception import CustomException
from src.logger import get_logger
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

logger=get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path,output_path):
        self.input_path=input_path
        self.output_path=output_path

        self.label_encoder = LabelEncoder()
        self.ohe = OneHotEncoder()
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None



        self.ordinal_cols = None
        self.nominal_cols = None
        self.num_cols = None

        self.X_train_transformed = None
        self.X_test_transformed = None

        
        os.makedirs(output_path,exist_ok=True)
        logger.info("Data Processing Initialized")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data Loaded Successfully")

        except Exception as e:
            logger.error("Error While Loading Data")
            raise CustomException("Failed to Load Data")
        
    def preprocess_data(self):
        try:
            self.df=self.df.drop(columns=['Patient_ID'])

            self.X=self.df.drop(columns=['Survival_Prediction'])
            self.y=self.df['Survival_Prediction']
            logger.info("X and Y made")

        except Exception as e:
            logger.error("Error While preprocessing Data")
            raise CustomException("Failed to preprocess Data")
        
    def train_test_split(self):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logger.info("Train Test Split Done")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            logger.error("Error While Train Test Split")
            raise CustomException("Failed to Train Test Split")
        
    def defining_columns(self):
        try:
            # HAVE ORDER
            self.ordinal_cols=['Cancer_Stage','Obesity_BMI','Diet_Risk','Physical_Activity','Healthcare_Access']

            # DOES NOT HAVE ORDER
            self.nominal_cols=['Country', 'Gender', 'Family_History', 'Smoking_History', 'Alcohol_Consumption', 'Diabetes', 'Inflammatory_Bowel_Disease', 
                        'Genetic_Mutation', 'Screening_History', 'Early_Detection', 'Treatment_Type', 'Survival_5_years', 'Mortality', 'Urban_or_Rural','Economic_Classification','Insurance_Status' ]

            # NUMERICAL COLUMNS
            self.num_cols = ['Age','Tumor_Size_mm','Healthcare_Costs','Incidence_Rate_per_100K','Mortality_Rate_per_100K']

            logger.info({'ordinal_cols':self.ordinal_cols,'nominal_cols':self.nominal_cols,'num_cols':self.num_cols})

        except Exception as e:
            logger.error("Error While Defining Columns")
            raise CustomException("Failed to Define Columns")

    def making_pipelines(self):
        try:
            # Numerical Pipeline
            num_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])

            # Categorical Pipeline
            cat_pipeline = Pipeline([
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Ordinal Pipeline
            ordinal_pipeline = Pipeline([
                ('encoder',OrdinalEncoder(categories=[
                    ['Localized', 'Regional', 'Metastatic'], # Cancer_Stage
                    ['Normal','Overweight','Obese'], # Obesity_BMI
                    ['Low','Moderate','High'], # Diet_Risk
                    ['Low','Moderate','High'], # Physical_Activity
                    ['Low','Moderate','High'], # Healthcare_Access

                ]))
            ])
            logger.info("num_pipeline, cat_pipeline, ordinal_pipeline Made")
            return num_pipeline, cat_pipeline, ordinal_pipeline

        except Exception as e:
            logger.error("Error While Making Pipeline")
            raise CustomException("Failed to make Pipeline")
        
    def preprocessor(self,num_pipeline,cat_pipeline,ordinal_pipeline):
        try:
            preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.nominal_cols),
            ('ord', ordinal_pipeline, self.ordinal_cols)
            ])

            logger.info("Preprocessor Made")
            return preprocessor

        except Exception as e:
            logger.error("Error While Making Preprocessor")
            raise CustomException("Failed to make Preprocessor")
        
    def transform(self,preprocessor,X_train,X_test,y_train,y_test):
        try:
            self.X_train_transformed = preprocessor.fit_transform(X_train)
            self.X_test_transformed = preprocessor.transform(X_test)

            logger.info("X_train_transformed and X_test_transformed made")

            self.y_train = self.label_encoder.fit_transform(y_train)
            y_test= self.label_encoder.transform(y_test)

            logger.info("Y_train and y_test label encoder done")

            return self.X_train_transformed,self.X_test_transformed,y_train,y_test
        
        except Exception as e:
            logger.error("Error While Making Preprocessor")
            raise CustomException("Failed to make Preprocessor")
        
    def feature_selection(self,preprocessor):
        try:
            model = RandomForestClassifier()
            model.fit(self.X_train_transformed,self.y_train)
            logger.info("RandomForestClassifier in the feature selection fitted")

            feature_names = preprocessor.get_feature_names_out()
            importance = model.feature_importances_
            feat_imp = pd.DataFrame({'feature':feature_names,'importance':importance})
            feat_imp=feat_imp.sort_values(by='importance',ascending=False)

            logger.info("Important Features are selected")
            return feat_imp

        except Exception as e:
            logger.error("Error While Selecting Important Features")
            raise CustomException("Failed to Select Important Features")

    
    def map_to_original(self, transformed_name):
        """Maps a single transformed feature name back to its original column name."""
        try:
            all_cols = self.num_cols + self.nominal_cols + self.ordinal_cols
            no_prefix = transformed_name.split('__')[1]  # remove num__, cat__, ord__
            for col in all_cols:
                if no_prefix.startswith(col):
                    return col
            return no_prefix  # fallback if no match

        except Exception as e:
            logger.error("Error While Mapping to Original")
            raise CustomException("Failed to Map to Original")

    def group_feature_importances(self, feat_imp):
        """Groups and sums importances by original feature name."""
        try:
            feat_imp['original_feature'] = feat_imp['feature'].apply(
                lambda x: self.map_to_original(x)
            )

            grouped = (
                feat_imp.groupby('original_feature')['importance']
                        .sum()
                        .sort_values(ascending=False)
                        .reset_index()
            )

            logger.info("Grouped feature importances by original column")
            return grouped

        except Exception as e:
            logger.error("Error While Grouping Feature Importances")
            raise CustomException("Failed to Group Feature Importances")
        
    def top_feature(self,grouped):
        try:
            top_features=grouped.head(5)['original_feature'].tolist()
            self.selected_features = top_features
            logger.info(f"Top Features: {self.selected_features}")

            self.X = self.X[self.selected_features] 
            logger.info("Feature Selection Done")

        except Exception as e:
            logger.error("Error While Feature Selection")
            raise CustomException("Failed to Feature Selection")
        
    def splitting_data_and_making_pipeline(self,num_pipeline,cat_pipeline):
        try:
            X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info("Train Test Split on top 5 Features done")

            new_num_cols      = ['Healthcare_Costs', 'Tumor_Size_mm', 'Incidence_Rate_per_100K', 'Age']
            new_nominal_cols  = ['Country']

            preprocessor = ColumnTransformer([
            ('num', num_pipeline, new_num_cols),
            ('cat', cat_pipeline, new_nominal_cols),
        ])
            
            logger.info("Preprocessor of new_num_cols, new_nominal_cols made")

            return X_train,X_test,y_train,y_test,preprocessor
        
        except Exception as e:
            logger.error("Error While splitting_data_and_making_pipeline")
            raise CustomException("Failed to splitting_data_and_making_pipeline")
        
    def transform_data(self,X_train,X_test,y_train,y_test,preprocessor):
        try:

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)

            return X_train_transformed,X_test_transformed,y_train,y_test
        
        except Exception as e:
            logger.error("Error While transforming data")
            raise CustomException("Failed to transform data")

        
    def save_data_and_preprocessor(self,X_train_transformed,X_test_transformed,y_train,y_test,preprocessor):
        try:
            joblib.dump(X_train_transformed,os.path.join(self.output_path, "X_train_transformed.pkl"))
            joblib.dump(X_test_transformed,os.path.join(self.output_path, "X_test_transformed.pkl"))
            joblib.dump(y_train,os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test,os.path.join(self.output_path, "y_test.pkl"))

            joblib.dump(preprocessor,os.path.join(self.output_path, "preprocessor.pkl"))
            logger.info("Saved X_train,X_test,y_train,y_test,preprocessor in .pkl")

        except Exception as e:
            logger.error("Error While save_data_and_preprocessor")
            raise CustomException("Failed to save_data_and_preprocessor")
        
    def run(self):
        self.load_data()
        self.preprocess_data()
        X_train, X_test, y_train, y_test=self.train_test_split()
        self.defining_columns()
        num_pipeline, cat_pipeline, ordinal_pipeline = self.making_pipelines()
        preprocessor = self.preprocessor(num_pipeline, cat_pipeline, ordinal_pipeline)
        X_train_transformed,X_test_transformed,y_train,y_test = self.transform(preprocessor,X_train, X_test, y_train, y_test)
        feat_imp = self.feature_selection(preprocessor)
        grouped = self.group_feature_importances(feat_imp)   
        self.top_feature(grouped)
        X_train,X_test,y_train,y_test,preprocessor = self.splitting_data_and_making_pipeline(num_pipeline,cat_pipeline)
        X_train_transformed,X_test_transformed,y_train,y_test = self.transform_data(X_train,X_test,y_train,y_test,preprocessor)
        self.save_data_and_preprocessor(X_train_transformed,X_test_transformed,y_train,y_test,preprocessor)

        logger.info("Data Processing Pipeline Completed")

if __name__=="__main__":
    input_path = "artifacts/raw/data.csv"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    output_path = args.output_path

    processor = DataProcessing(input_path,output_path)
    processor.run()