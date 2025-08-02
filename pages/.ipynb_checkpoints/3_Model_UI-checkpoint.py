import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

st.title("Hart's NFL Win Probability Predictor")

model = joblib.load('pipe_model.pkl')


st.markdown("""
This tool predicts home win probability.
""")
home_dif = st.slider("Score Differential (Home-Away)", -30, 30, 0)

home_pos = st.radio(
    "Home Possession?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
qtr = st.radio(
    "Quarter",
    options=[1, 2, 3, 4])

clock = st.slider("Clock Minutes", 0, 15, 15)
yardline = st.slider("Distance from Goal", 0, 100, 50)


down = st.radio(
    "Down",
    options=[1, 2, 3, 4]
)
ydstogo = st.slider("First Down Yards To go", 0, 30, 10)

home_spread = st.slider("Home Team Pregame Spread", -14, 14, 0)


# st.write("Quarter Selected:", qtr)

game_seconds_remaining = ((4-qtr)*15+clock)*60
time_weight = 1-(game_seconds_remaining/3600)

feature_columns = ['yardline_100_home', 
                   'down', 
                   'ydstogo', 
                   'home_pos', 
                   'home_dif', 
                   'home_spread_line', 
                   'time_weight'
                  ]



user_inputs = {'yardline_100_home': yardline,
 'down': down,
 'ydstogo': ydstogo,
 'home_pos': home_pos,
 'home_dif': home_dif,
 'home_spread_line': home_spread, 
 'time_weight':time_weight}

print("✅ Model loaded:", type(model))


# if st.button("Predict"):

X_input = pd.DataFrame([[user_inputs.get(col, 0) for col in user_inputs.keys()]], 
                       columns=user_inputs.keys())

# if X_input['home_pos'] == 0:
#     X_input['yardline_100_home'] = (100 - X_input['yardline_100_home'])


X_input['yardline_100_home'] = X_input.apply(lambda X:(100-X['yardline_100_home']) if X['home_pos'] == 0 else X['yardline_100_home'], axis = 1)
# st.write(X_input)


# X_input['time_weight'] = .5

prediction = model.predict(X_input)[0]
st.success(f"Home Win Probability: {prediction}")
    
    
    