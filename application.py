from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd


app=Flask(__name__)

model_path = "artifacts/models/model.pkl"
preprocessor_path = "artifacts/processed/preprocessor.pkl"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

@app.route('/')
def home():
    return render_template("index.html",predication=None)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        country = str(request.form["country"])
        healthcare_cost = int(request.form['Healthcare_Costs'])
        tumor_size = int(request.form['Tumor_Size_mm'])
        incidence_rate = int(request.form['Incidence_Rate_per_100K'])
        age = int(request.form['age'])


        input_df = pd.DataFrame([{
            "Country": country,
            "Healthcare_Costs": healthcare_cost,
            "Tumor_Size_mm": tumor_size,
            "Incidence_Rate_per_100K": incidence_rate,
            "Age": age
        }])

        preprocessored_input = preprocessor.transform(input_df)

        predication = model.predict(preprocessored_input)[0]

        return render_template('index.html',predication=predication)
    
    except Exception as e:
        return str(e)
    
if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)