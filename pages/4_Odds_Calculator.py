import streamlit as st
import numpy as np
import pandas as pd


st.title("Odds Calculator")
st.markdown("""
    Use this tool to convert between probability and sportsbook odds, with or without a the sportsbook's built-in margin (vig).
            """)

prob = st.number_input("Probability", 
                       value = .5,
                       step = .1e-4, 
                       min_value = 1e-5, 
                       max_value = 1-1e-5, 
                       format = "%.5f", 
                       width = 200)
house = st.number_input("House Advantage (Default: .04545)", 
                       value = .04545,
                       step = .1e-4,
                       min_value = 0.0, 
                       max_value = .99,
                       format = "%.5f", 
                       width = 200)

hold = ((2*house)/(1-house))/2

def prob_to_odds(prob):
    if prob > 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)


def prob_to_market_prob(prob, hold):
    p1 = prob
    p2 = 1-p1
    # hold = 0.0476
    # hold = float(hold)
    overround = 1 + hold
    fair_total = p1 + p2
    vig_p1 = min((p1 / fair_total) * overround, 0.99999)
    vig_p2 = overround - vig_p1
    return vig_p1#vig_p2

# Combine functions.
def prob_to_market_odds(prob, hold):
    # prob = vig_p1
    inflated_prob = prob_to_market_prob(prob, hold)
    market_odds = prob_to_odds(inflated_prob)
    rounded_odds = int(round(market_odds, -(len(str(abs(market_odds)))-2)))
    # return rounded_odds
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


# home_win_prob = model.predict(X_input)[0]
# home_odds = prob_to_market_odds(home_win_prob)
# away_win_prob = 1-home_win_prob
# away_odds = prob_to_market_odds(away_win_prob)

# prob = st.text_input("test")


if st.button("Calculate"):
    # st.write(type(prob)
    result = prob_to_market_odds(prob, hold)
    st.info(f'Odds: {result}')

    


