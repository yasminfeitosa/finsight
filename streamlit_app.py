import streamlit as st
import app
import alerts_page  # Your alerts page module with run_alerts_page()

pages = {
    "Main App": app.run_main_app,
    "Watchlist & Alerts": alerts_page.run_alerts_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
