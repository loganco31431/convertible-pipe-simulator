"""PIPE simulator dashboard.

    python -m streamlit run app.py        (or double-click Simulator.bat)

Two pages over the `pipesim` package:
- Trade-out: the term sheet and five selling strategies replayed on real intraday bars (Yahoo, a
  Bloomberg export, or a live Bloomberg terminal), scored against VWAP, plus an execution plan
  for a notice on the next trading day and the stock's liquidity.
- Deal economics: Monte Carlo on daily prices with terms read from filed agreements,
  frictionless vs volume- and impact-aware, warrants, replay, best setup, deal-size sweep.
"""
import streamlit as st

st.set_page_config(page_title="PIPE Desk", layout="wide", initial_sidebar_state="collapsed")
st.navigation([
    st.Page("views/tradeout.py", title="Trade-out", default=True),
    st.Page("views/deal.py", title="Deal economics"),
], position="top").run()
