from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np

# from sklearn.preprocessing import 

from src.pipeline.predict_pipeline import CustomData,PredictPipeline



application=Flask(__name__)
app=application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_data",methods=["GET","POST"])
def prediction_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        get1=request.form.get

        gender=get1("gender")
        race_ethnicity=get1("race_ethnicity")
        parental_level_of_education=get1("parental_level_of_education")
        lunch=get1("lunch")
        test_preparation_course=get1("test_preparation_course")
        reading_score=get1("reading_score")
        writing_score=get1("writing_score")


        data=CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,            
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        # print(f"gender : {data.gender}")
        # print(f"race_ethnicity : {data.race_ethnicity}")
        # print(f"parental_level_of_education : {data.parental_level_of_education}")
        # print(f"lunch : {data.lunch}")
        # print(f"test_preparation_course : {data.test_preparation_course}")
        # print(f"reading_score : {data.reading_score}")
        # print(f"writing_score : {data.writing_score}")

        prep_df=data.get_data_as_frame()
        
        # print(f"df columms : {prep_df.columns}")
        # print(f"df shape : {prep_df.shape}")

        results=PredictPipeline().predict(prep_df)

        return render_template("home.html",results=results[0])
        # return render_template("home.html")
    



if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=8000)