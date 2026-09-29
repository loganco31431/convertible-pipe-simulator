@echo off
rem Double-click to open the PIPE simulator dashboard in your browser.
cd /d "%~dp0"
python -m streamlit run app.py --server.headless false
