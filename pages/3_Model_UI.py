import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from scipy.special import logit, expit
from utils import logit_func, expit_func

import streamlit as st

import streamlit as st

st.set_page_config(layout="wide")

hide_sidebar_style = """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
        [data-testid="stSidebar"][aria-expanded="false"]{
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)



st.title("Hart's NFL Win Probability Predictor")

model = joblib.load('ui_model.pkl')


st.markdown("""
    This tool uses a machine learning model trained on over 20 years of NFL data to estimate win probabilities in real time. Enter the current game state- possession, score, time, time outs remaining, field position, down and distance, and pregame spread to see the model’s prediction alongside market-implied odds.
""")


feature_columns = ['yardline_100_home', 
                   'down', 
                   'ydstogo', 
                   'home_pos', 
                   'home_score_differential', 
                   'home_spread_line', 
                   'time_weight',
                   'home_timeouts_remaining',
                   'away_timeouts_remaining',
                  ]

DEFAULTS = {
    "qtr": 1,
    "clock": 15,
    "yardline": 50,
    "home_pos": 1,
    "score_differential": 0,
    "home_spread": 0,
    "ydstogo": 10,
    "down": 1,
    "home_timeouts": 3,
    "away_timeouts": 3
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


if st.session_state.get("reset_triggered", False):
    st.session_state['qtr'] = 1
    st.session_state['score_differential'] = 0
    st.session_state['clock'] = 15
    st.session_state['yardline'] = 50
    st.session_state['down'] = 1
    st.session_state['ydstogo'] = 10
    st.session_state['home_spread'] = 0
    st.session_state['reset_triggered'] = False
    st.session_state['home_timeouts'] = 3
    st.session_state['away_timeouts'] = 3
    st.rerun()


# qtr = st.slider("Quarter", 1, 4, key="qtr")
# clock = st.slider("Clock Minutes", 0, 15, key="clock")
# yardline = st.slider("Distance from Goal", 0, 100, key="yardline")
# down = st.radio("Down", options=[1, 2, 3, 4], key="down")
# ydstogo = st.slider("First Down Yards To go", 0, 30, key="ydstogo")
# home_spread = st.slider("Home Team Pregame Spread", -14, 14, key="home_spread")

if st.button("🔄 Reset to Default"):
    st.session_state['reset_triggered'] = True
    st.rerun()

st.subheader("Game State")

home_pos = st.radio(
    "Possession",
    options=[1, 0],
    format_func=lambda x: "Home" if x == 1 else "Away",
    horizontal = True,
    key="home_pos"
)
# st.divider()
# st.subheader("Score Differential")



score_differential = st.slider("Score Differential", 30, 
    -30, 
    key="score_differential")



score_differential = -score_differential


label_html_left = (
    '<span style="display:inline-block; border:1px solid #48FF6A; border-radius:12px; '
    'padding:4px 12px; font-size:1em; color:#48FF6A; white-space:nowrap;">'
    'Home Team Winning</span>'
)
label_html_right = (
    '<span style="display:inline-block; border:1px solid #48FF6A; border-radius:12px;' 
    'padding:4px 12px; font-size:1em; color:#48FF6A; white-space:nowrap;">'
    'Away Team Winning</span>'
)

st.markdown(
    f"""
    <table style="width:100%; border-collapse:collapse; border:none;">
      <tr>
        <td style="text-align:left; border:none;">{label_html_left}</td>
        <td style="text-align:right; border:none;">{label_html_right}</td>
      </tr>
    </table>
    """,
    unsafe_allow_html=True,
)


st.divider()

st.subheader("Clock")

qtr = st.radio(
    "Quarter",
    options=[1, 2, 3, 4], 
    horizontal = True,
    key="qtr")
clock = st.slider("Minutes Remaining", 0, 15, key="clock")


col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    home_timeouts = st.radio(
        "Home Timeouts",
        options=[0, 1, 2, 3],
        horizontal=True,
        key="home_timeouts"
    )

with col3:
    away_timeouts = st.radio(
        "Away Timeouts",
        options=[0, 1, 2, 3],
        horizontal=True,
        key="away_timeouts" 
    ) 





st.divider()

st.subheader("Field Position")

yardline = st.slider("Possession Team Distance From Score (Yards)", 0, 100, key="yardline")

# if home_pos == 0:
#     yardline = 100 - yardline

# st.markdown('<span style="color:48FF6A;">⬆ Goal Line</span>', unsafe_allow_html=True)

# st.markdown(
#         """
#         <div style="text-align:left; font-size:1.0em; color:48FF6A !important">
#          <span style="border:2px solid #48FF6A; padding:4px 18px; border-radius:8px;">⬆ Goal Line</span>
#         </div>
#         """, unsafe_allow_html=True
#     )

# Set goal line marker.
st.markdown(
        """
        <div style="text-align:left; font-size:1.0em; color:48FF6A !important">
        <span style="display:inline-block; border:1px solid #48FF6A; border-radius:12px; 
        padding:4px 12px; font-size:1em; color:#48FF6A; white-space:nowrap;">⬆ Goal Line
        </span>
        </div>
        """, unsafe_allow_html=True
    )

# "display:inline-block; border:1px solid #48FF6A; border-radius:12px; '
#     'padding:4px 12px; font-size:1em; color:#48FF6A; white-space:nowrap;">'
#     'Away Team Winning

# col1, col2, col3 = st.columns([1, 2, 1])
# with col1:
#     st.markdown('<span style="color:48FF6A;">⬆️ Goal Line</span>', unsafe_allow_html=True)
# with col2:
#     st.markdown("<center><span style='color:gray;'></span></center>", unsafe_allow_html=True)
# with col3:
#     st.markdown('<span style="color:48FF6A; float:right;">Away Team Favored</span>', unsafe_allow_html=True)

st.divider()

st.subheader("Down & Distance")

down = st.radio(
    "Down",
    options=[1, 2, 3, 4],
    horizontal = True,
    key="down"
)
ydstogo = st.slider("First Down Distance (Yards)", 0, 30, key="ydstogo")

st.divider()
# st.write("---")

st.subheader("Home Team Pregame Spread")
home_spread = st.slider(" ",min_value = -21.0, 
    max_value = 21.0, 
    step = .5,
format =  '%.1f', key="home_spread")

# col1, col2, col3 = st.columns([2, 8, 2])
# with col1:
#     st.markdown(
#         '<div style="test-align:left;">'
#         '<span style="display:inline-block; border:2px solid #48FF6A; border-radius:50px; '
#         'padding:6px 18px; font-size:1em; color:#48FF6A; '
#         'white-space:nowrap;">Home Team Favored</span>',
#         unsafe_allow_html=True
#     )
# with col3:
#     st.markdown(
#         '<div style="test-align:right;">'
#         '<span style="display:inline-block; border:2px solid #48FF6A; border-radius:50px; '
#         'padding:6px 18px; font-size:1em; color:#48FF6A; '
#         'white-space:nowrap;">Away Team Favored</span>',
#         unsafe_allow_html=True
    # )

# OMG this is so annoying.

label_html_left = (
    '<span style="display:inline-block; border:2px solid #48FF6A; border-radius:12px; '
    'padding:4px 12px; font-size:1em; color:#48FF6A; white-space:nowrap;">'
    'Home Team Favored</span>'
)
label_html_right = (
    '<span style="display:inline-block; border:2px solid #48FF6A; border-radius:12px; '
    'padding:4px 12px; font-size:1em; color:#48FF6A; white-space:nowrap;">'
    'Away Team Favored</span>'
)

st.markdown(
    f"""
    <table style="width:100%; border-collapse:collapse; border:none;">
      <tr>
        <td style="text-align:left; border:none;">{label_html_left}</td>
        <td style="text-align:right; border:none;">{label_html_right}</td>
      </tr>
    </table>
    """,
    unsafe_allow_html=True,
)

game_seconds_remaining = ((4-qtr)*15+clock)*60
time_weight = 1-(game_seconds_remaining/3600)

user_inputs = {
    'yardline_100_home': yardline,
    'down': down,
    'ydstogo': ydstogo,
    'home_pos': home_pos,
    'home_score_differential': score_differential,
    'home_spread_line': home_spread, 
    'time_weight':time_weight,
    'home_timeouts_remaining':home_timeouts,
    'away_timeouts_remaining':away_timeouts,
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

# Convert probability to market probability.
def prob_to_market_prob(prob):
    p1 = prob
    p2 = 1-p1
    hold = 0.0476
    overround = 1 + hold
    fair_total = p1 + p2
    vig_p1 = min((p1 / fair_total) * overround, 0.99999)
    vig_p2 = overround - vig_p1
    return vig_p1, vig_p2

# Convert probability to American odds.
def prob_to_odds(prob):
    if prob > 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)

# Combine functions.
def prob_to_market_odds(prob):
    inflated_prob = prob_to_market_prob(prob)[0]
    market_odds = prob_to_odds(inflated_prob)
    rounded_odds = int(round(market_odds, -(len(str(abs(market_odds)))-2)))
    if rounded_odds > 100:
        rounded_odds = min(rounded_odds, 5000)
    if rounded_odds < 100:
        rounded_odds = max(rounded_odds, -100000)
    if abs(rounded_odds) == 100:
        return f'(EVEN)'
    elif rounded_odds > 100:
        return f'(+{rounded_odds})'
    else:
        return f'({rounded_odds})'

# Instantiate variables.
home_win_prob = model.predict(X_input)[0]
home_odds = prob_to_market_odds(home_win_prob)
away_win_prob = 1-home_win_prob
away_odds = prob_to_market_odds(away_win_prob)


# CSS styling.
st.markdown(
    f"""
    <style>
    
    /* Add padding at bottom */
    .main, .block-container {{
        padding-bottom: 150px;
    }}

    /* Set default colors */
    body, .main, .block-container, .sidebar, .sidebar-content, .stButton > button {{
        background-color: #141e28;
        color: white;
    }}

    /* Set button attributes */
    .stButton > button {{
        border: 1px solid #444;
        border-radius: 1tpx solid #444;
        font-color:white;
    }}

    /* Set radio button color */
    .stSlider label, .stRadio label, .stRadio div {{
        color: white;
    }}

    /* Set slider width */
    .st-c7 {{
        height: .5rem;
    }}

    /* Enlarge slider buttons */
    .st-emotion-cache-1dj3ksd {{ 
       height: 1.5rem;
       width: 1.5rem;
    }}

    /* Floating panel attributes*/
    .fixed-2x2-panel {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        height: 150px;
        background: rgba(20,30,40,0.97);
        z-index: 9999; 
        padding: 2px 0 24px 0;
        display: flex;
        justify-content: center;
    }}

    /* Grid attributes */
    .panel-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: 1fr 1fr;
        gap: 5px;
        width: 100%;
        height: 100px; 
        max-width: 500px;  
    }}

    /* Cell attributes */
    .panel-cell {{
        border-radius: 24px;
        font-size: 1em;  
        font-weight: 500;
        color: #fff;
        padding: 10px 0; # 2px 0;
        text-align: center;
        box-shadow: 0 4px 28px #0006;
        min-width: 0;
        word-break: break-word;
        height: 45px;  
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    /* Set cell background colors */
    .cell-home {{ background: #0525c5; }}
    .cell-away {{ background: #a61616; }}

    
    /* Add text */
    </style>
    <div class="fixed-2x2-panel">
      <div class="panel-grid">
        <div class="panel-cell cell-home">
            Home Win Probability: {home_win_prob*100:.2f}%
        </div>
        <div class="panel-cell cell-away">
            Away Win Probability:  {away_win_prob*100:.2f}%
        </div>
        <div class="panel-cell cell-home">
            Home Odds: {home_odds}
        </div>
        <div class="panel-cell cell-away">
            Away Odds: {away_odds}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

