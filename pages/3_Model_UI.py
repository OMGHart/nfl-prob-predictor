import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from scipy.special import logit, expit

st.title("Hart's NFL Win Probability Predictor")

model = joblib.load('ui_model.pkl')


st.markdown("""
This tool predicts win NFL win probability.
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


if st.session_state.get("reset_triggered", False):
    st.session_state['qtr'] = 1
    st.session_state['score_differential'] = 0
    st.session_state['qtr'] = 1
    st.session_state['clock'] = 15
    st.session_state['yardline'] = 50
    st.session_state['down'] = 1
    st.session_state['ydstogo'] = 10
    st.session_state['home_spread'] = 0
    st.session_state['reset_triggered'] = False
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

# with st.container():
#     st.subheader("Game State")
#     possession = st.radio("Possession", ["Home", "Away"])
#     score_diff = st.slider("Score Differential (Home - Away)", -30, 30, 0)
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col1:
#         st.markdown("Home Team Winning", unsafe_allow_html=True)
#     with col3:
#         st.markdown("Away Team Winning", unsafe_allow_html=True)


label_html_left = (
    '<span style="display:inline-block; border:2px solid #48FF6A; border-radius:12px; '
    'padding:4px 12px; font-size:1em; color:#48FF6A; white-space:nowrap;">'
    'Home Team Winning</span>'
)
label_html_right = (
    '<span style="display:inline-block; border:2px solid #48FF6A; border-radius:12px; '
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

st.divider()

st.subheader("Field Position")

yardline = st.slider("Possession Team Distance From Score (Yards)", 0, 100, key="yardline")

# st.markdown('<span style="color:lightgreen;">⬆ Goal Line</span>', unsafe_allow_html=True)

# st.markdown(
#         """
#         <div style="text-align:left; font-size:1.0em; color:lightgreen">
#          <span style="border:2px solid #48FF6A; padding:4px 18px; border-radius:8px;">GOAL LINE</span>
#         </div>
#         """, unsafe_allow_html=True
#     )

st.markdown(
    ' <div style="text-align:left; font-size:1.0em; color:lightgreen">⬆️   <span style="display:inline-block; border:2px solid #48FF6A; border-radius:50px; '
    'padding:6px 18px; font-size:1em; color:#48FF6A; '
    'white-space:nowrap;">Goal Line</span>',
    unsafe_allow_html=True
)
# col1, col2, col3 = st.columns([1, 2, 1])
# with col1:
#     st.markdown('<span style="color:lightgreen;">⬆️ Goal Line</span>', unsafe_allow_html=True)
# with col2:
#     st.markdown("<center><span style='color:gray;'></span></center>", unsafe_allow_html=True)
# with col3:
#     st.markdown('<span style="color:lightgreen; float:right;">Away Team Favored</span>', unsafe_allow_html=True)

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

st.subheader("Home Team Pregame Spread")
home_spread = st.slider("",min_value = -21.0, 
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
# After your slider


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


home_win_prob = expit(model.predict(X_input))[0]
home_odds = prob_to_market_odds(home_win_prob)
away_win_prob = 1-home_win_prob
away_odds = prob_to_market_odds(away_win_prob)




# col1, col2 = st.columns(2)

# with col1:
#     st.success(f'Home Win Probability: {home_prob*100:.2f}%')
#     st.success(f'Home Odds: {home_odds}')

# with col2:
#     st.success(f'Away Win Probability:  {away_prob*100:.2f}%')
#     st.success(f'Away Odds: {away_odds}')

# st.markdown("&nbsp;", unsafe_allow_html=True)


# if st.button("🔄 Reset to Default"):
#     st.session_state['reset_triggered'] = True
#     st.rerun()


# Define your card style as a Python variable
# card_style = (
#     "background-color:#1e472d;"      # Change this hex to any color you want!
#     "padding:32px 20px;"
#     "border-radius:20px;"
#     "margin-bottom:16px;"
#     "color:white;"
#     "font-size:1em;"
#     "font-family:sans-serif;"
#     "text-align:center;"
# )

# # Render cards using columns
# col1, col2 = st.columns(2)
# with col1:
#     st.markdown(
#         f'<div style="{card_style}">Home Win Probability: {home_win_prob*100:.2f}%</div>',
#         unsafe_allow_html=True
#     )
# with col2:
#     st.markdown(
#         f'<div style="{card_style}">Away Win Probability: {away_win_prob*100:.2f}%</div>',
#         unsafe_allow_html=True
#     )

# col3, col4 = st.columns(2)
# with col3:
#     st.markdown(
#         f'<div style="{card_style}">Home Odds: {home_odds}</div>',
#         unsafe_allow_html=True
#     )
# with col4:
#     st.markdown(
#         f'<div style="{card_style}">Away Odds: {away_odds}</div>',
#         unsafe_allow_html=True
#     )

# card_style = (
#     # "background-color:#244533;"
#     "padding:28px 12px;"
#     "border-radius:20px;"
#     "margin:8px 0px;"
#     "color:white;"
#     # "font-size:1.5em;"
#     "font-family:sans-serif;"
#     "text-align:center;"
#     "width:96%;"  # slight shrink to prevent overflow on small screens
# )

# card_home = f'<div style="{card_style};\
#     background-color:darkblue; \
#     font-size:1.5em; \
#     ">Home Win Probability: {home_win_prob*100:.2f}%</div>'
# card_away = f'<div style="{card_style}; \
#     background-color:darkred; \
#     font-size:1.5em; \
#     ">Away Win Probability: {away_win_prob*100:.2f}%</div>'

# st.markdown(
#     f"""
#     <table style="width:100%; border-collapse:collapse; border:none;">
#       <tr>
#         <td style="vertical-align:top; border:none; width:50%;">{card_home}</td>
#         <td style="vertical-align:top; border:none; width:50%;">{card_away}</td>
#       </tr>
#     </table>
#     """,
#     unsafe_allow_html=True,
# )

# card_home_odds = f'<div style="{card_style}; \
# background-color:darkblue; \
# ">Home Odds: {home_odds}</div>'
# card_away_odds = f'<div style="{card_style}; \
# background-color:darkred; \
# ">Away Odds: {away_odds}</div>'

# st.markdown(
#     f"""
#     <table style="width:100%; border-collapse:collapse; border:none;">
#       <tr>
#         <td style="vertical-align:top; border:none; width:50%;">{card_home_odds}</td>
#         <td style="vertical-align:top; border:none; width:50%;">{card_away_odds}</td>
#       </tr>
#     </table>
#     """,
#     unsafe_allow_html=True,
# )


# st.markdown(
#     f"""
#     <style>
#     .floating-panel {{
#         position: fixed;
#         left: 0;
#         bottom: 0;
#         width: 100vw;
#         background: rgba(20,30,40,0.98);
#         color: #fff;
#         padding: 20px 0 10px 0;
#         box-shadow: 0 -2px 24px 0 #0007;
#         z-index: 9999;
#         display: flex;
#         justify-content: center;
#         gap: 60px;
#     }}
#     .floating-panel > div {{
#         background: #244533;
#         border-radius: 18px;
#         padding: 16px 38px;
#         font-size: 1.4em;
#         font-weight: 700;
#         box-shadow: 0 2px 16px #0003;
#     }}
#     </style>
#     <div class="floating-panel">
#         <div>Home Win: {home_win_prob*100:.2f}%</div>
#         <div>Away Win: {away_win_prob*100:.2f}%</div>
#         <div>Home Odds: {home_odds}</div>
#         <div>Away Odds: {away_odds}</div>
#     </div>
#     """, unsafe_allow_html=True
# )


### WORKING

# st.markdown(
#     f"""
#     <style>
#     .fixed-2x2-panel {{
#         position: fixed;
#         left: 0;
#         bottom: 0;
#         width: 100vw;
#         background: rgba(20,30,40,0.97);
#         z-index: 9999;
#         padding: 24px 0 12px 0;
#         display: flex;
#         justify-content: center;
#     }}
#     .panel-grid {{
#         display: grid;
#         grid-template-columns: 1fr 1fr;
#         grid-template-rows: 1fr 1fr;
#         gap: 24px;
#         width: 90vw;
#         min-width: 400px;
#         max-width: 760px;
#     }}
#     .panel-cell {{
#         border-radius: 36px;
#         font-size: 1em;
#         font-weight: t00;
#         color: #fff;
#         padding: 28px 0 20px 0;
#         text-align: center;
#         box-shadow: 0 4px 28px #0006;
#         min-width: 0;
#         word-break: break-word;
#     }}
#     .cell-home {{ background: #0525c5; }}
#     .cell-away {{ background: #a61616; }}
#     @media (max-width: 400px) {{
#         .panel-grid {{
#             display: flex;
#             flex-direction: column;
#             gap: 12px;
#             width: 98vw;
#         }}
#         .panel-cell {{
#             font-size: 1em;
#             padding: 18px 0 14px 0;
#         }}
#     }}
#     </style>
#     <div class="fixed-2x2-panel">
#       <div class="panel-grid">
#         <div class="panel-cell cell-home">
#             Home Win Probability:<br>{home_win_prob*100:.2f}%
#         </div>
#         <div class="panel-cell cell-away">
#             Away Win Probability:<br>{away_win_prob*100:.2f}%
#         </div>
#         <div class="panel-cell cell-home">
#             Home Odds: {home_odds}
#         </div>
#         <div class="panel-cell cell-away">
#             Away Odds: {away_odds}
#         </div>
#       </div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

###NOT
st.markdown(
    f"""
    <style>
    .fixed-2x2-panel {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        height: 15vh;
        background: rgba(20,30,40,0.97);
        z-index: 9999;
        padding: 24px 0 12px 0;
        display: flex;
        justify-content: center;
    }}
    .panel-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: 1fr 1fr;
        gap: 10px;
        width: 190vw;
        height: 10px;
        max-width: 600px;

    }}
    .panel-cell {{
        border-radius: 24px;
        font-size: 1em;
        font-weight: 500;
        color: #fff;
        padding: 6px 0 6px 0;
        text-align: center;
        box-shadow: 0 4px 28px #0006;
        min-width: 0;
        word-break: break-word;
        height: 10;
    }}
    .cell-home {{ background: #0525c5; }}
    .cell-away {{ background: #a61616; }}
    @media (max-width: 400px) {{
        .panel-grid {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            width: 98vw;
            height: 10vh;
        }}
        .panel-cell {{
            font-size: 1.1em;
            padding: 18px 0 14px 0;
            height: 100px;
        }}
    }}
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

