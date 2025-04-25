@echo off
REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Run the Streamlit app
streamlit run app.py

REM Keep the command window open
pause
