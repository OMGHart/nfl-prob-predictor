import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from scipy.special import logit, expit

st.title("Hart's NFL Win Probability Predictor")

model = joblib.load('ui_model.pkl')


st.markdown("""
This tool predicts home win probability.
""")


feature_columns = ['yardline_100_home', 
                   'down', 
                   'ydstogo', 
                   'home_pos', 
                   'home_score_differential', 
                   'home_spread_line', 
                   'time_weight'
                  ]





DEFAULTS = {

    "qtr": 1,
    "clock": 15,
    "yardline": 50,
    "home_pos": 1,
    "score_differential": 0,
    "home_spread": 0,
    "ydstogo": 10,
    "down": 1
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.button("🔄 Reset to Default"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()



# qtr = st.slider("Quarter", 1, 4, key="qtr")
# clock = st.slider("Clock Minutes", 0, 15, key="clock")
# yardline = st.slider("Distance from Goal", 0, 100, key="yardline")
# down = st.radio("Down", options=[1, 2, 3, 4], key="down")
# ydstogo = st.slider("First Down Yards To go", 0, 30, key="ydstogo")
# home_spread = st.slider("Home Team Pregame Spread", -14, 14, key="home_spread")

score_differential = st.slider("Score Differential (Home-Away)", -30, 30, 0, key="score_differential")
home_pos = st.radio(
    "Possession",
    options=[1, 0],
    format_func=lambda x: "Home" if x == 1 else "Away",
    key="home_pos"
)
qtr = st.radio(
    "Quarter",
    options=[1, 2, 3, 4], 
    key="qtr")
clock = st.slider("Clock (Minutes Remaining)", 0, 15, 15, key="clock")
yardline = st.slider("Distance from Score", 0, 100, 50, key="yardline")
down = st.radio(
    "Down",
    options=[1, 2, 3, 4],
    key="down"
)
ydstogo = st.slider("First Down Yards To Go", 0, 30, 10, key="ydstogo")
home_spread = st.slider("Home Team Pregame Spread", -14, 14, 0, key="home_spread")
game_seconds_remaining = ((4-qtr)*15+clock)*60
time_weight = 1-(game_seconds_remaining/3600)

user_inputs = {
    'yardline_100_home': yardline,
    'down': down,
    'ydstogo': ydstogo,
    'home_pos': home_pos,
    'home_score_differential': score_differential,
    'home_spread_line': home_spread, 
    'time_weight':time_weight
 }

# st.write("Quarter Selected:", qtr)




print("✅ Model loaded:", type(model))


# if st.button("Predict"):

X_input = pd.DataFrame([[user_inputs.get(col, 0) for col in user_inputs.keys()]], 
                       columns=user_inputs.keys())

# if X_input['home_pos'] == 0:
#     X_input['yardline_100_home'] = (100 - X_input['yardline_100_home'])


X_input['yardline_100_home'] = X_input.apply(lambda X:(100-X['yardline_100_home']) if X['home_pos'] == 0 else X['yardline_100_home'], axis = 1)
# st.write(X_input)


# X_input['time_weight'] = .5

#Convert probability to American odds.
def prob_to_odds(prob):
    if prob > 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)


def prob_to_market_prob(prob):
    p1 = prob
    p2 = 1-p1
    hold = 0.0476
    overround = 1 + hold
    fair_total = p1 + p2
    vig_p1 = min((p1 / fair_total) * overround, 0.99999)
    vig_p2 = overround - vig_p1
    return vig_p1, vig_p2


# Combine functions.
def prob_to_market_odds(prob):
    inflated_prob = prob_to_market_prob(prob)[0]
    market_odds = prob_to_odds(inflated_prob)
    rounded_odds = round(market_odds, -(len(str(abs(market_odds)))-2))
    if abs(rounded_odds) == 100:
        return f'(EVEN)'
    elif rounded_odds > 100:
        return f'(+{round(market_odds, -(len(str(abs(market_odds)))-2))})'
    else:
        return f'({rounded_odds})'

home_prob = expit(model.predict(X_input))[0]
home_odds = prob_to_market_odds(home_prob)
away_prob = 1-home_prob
away_odds = prob_to_market_odds(away_prob)


col1, col2 = st.columns(2)

with col1:
    st.success(f'Home Win Probability: {home_prob*100:.2f}%')
    st.success(f'Home Odds: {home_odds}')

with col2:
    st.success(f'Away Win Probability:  {away_prob*100:.2f}%')
    st.success(f'Away Odds: {away_odds}')

    
    
    