from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline,CustomData
from src.exception import CustomException
import sys


application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )

            data_framed = data.change_into_data_frame()
            print(data_framed)
            preds = PredictPipeline()
            prediction = preds.prediction(data=data_framed)
            print(prediction)
            return render_template('home.html',results=prediction[0])
        
    except Exception as e:
        raise CustomException(e,sys)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
