"""PIPE simulator dashboard.

    python -m streamlit run app.py        (or double-click Simulator.bat)

Two pages over the `pipesim` package:
- Deal economics: Monte Carlo on daily prices, frictionless vs volume- and impact-aware, replay,
  best setup, deal-size sweep.
- Trade-out vs VWAP: the term sheet and five selling strategies replayed on real intraday bars
  (Yahoo, a Bloomberg export, or a live Bloomberg terminal), scored against VWAP.
"""
import streamlit as st

st.set_page_config(page_title="PIPE Simulator", layout="wide")
st.navigation([
    st.Page("views/deal.py", title="Deal economics", default=True),
    st.Page("views/tradeout.py", title="Trade-out vs VWAP"),
]).run()
