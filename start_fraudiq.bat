@echo off
cd /d C:\Users\TINASHE\fraud_intelligence_system
call venv\Scripts\activate
start "" http://localhost:8501
python -m streamlit run app.py