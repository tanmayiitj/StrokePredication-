from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("stroke_rf.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        #Gender
        gender=int(request.form['gender'])

        #ever_married
        ever_married=int(request.form['ever_married'])

        #work_type
        work_type=int(request.form['work_type'])

        #smoking_status
        smoking_status=int(request.form['smoking_status'])

        #Residence_type
        Residence_type=int(request.form['Residence_type'])

        # Age
        age = int(request.form["age"])

        # Hypertension
        hypertension = int(request.form["hypertension"])

        # Heart disease
        heart_disease = int(request.form["heart_disease"])

        #avg_glucose_level
        avg_glucose_level = float(request.form["avg_glucose_level"])

        #BMI
        bmi = float(request.form["bmi"])

        print(request.form)
        
        prediction=model.predict(pd.DataFrame([[
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            Residence_type,
            avg_glucose_level,
            bmi,
            smoking_status
        ]]))

        output=prediction

        if output == 1:
            return render_template('home.html',predicted_text="The patient is likely to have Stroke Disease!!")
        else:
            return render_template('home.html',predicted_text="The patient is not likely to have Stroke Disease!!")
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)