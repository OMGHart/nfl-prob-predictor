import streamlit as st
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('ui_model.pkl')

@app.route("/", methods = ['GET', 'POST'])

def index():
    prediction = None
    if request.method == "POST":
        home_score_differential = float(request.form['home_score_differential'])
        yardline_100_home = float(request.form['yardline_100_home'])
        time_weight = float(request.form['time_weight'])

        X_input = np.array([[home_dif, yardline_100_home, time_weight]])
        prediction = model.predict(X_input)[0]
    return render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)