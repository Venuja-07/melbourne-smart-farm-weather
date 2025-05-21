
import streamlit as st
import pandas as pd
from melbourne_weather_forecast_module import (
    fetch_weather_data,
    preprocess_data,
    linear_regression_forecast,
    arima_forecast,
    prophet_forecast,
    plot_forecasts
)

st.set_page_config(page_title="Smart Farm: Melbourne Weather Forecast", layout="centered")

st.title("🌾 Melbourne Weather Forecast Dashboard")
st.markdown("Get 7-day weather predictions using Machine Learning models for Smart Farming.")

api_key = st.secrets["general"]["OPENWEATHER_API_KEY"]

try:
    df = fetch_weather_data(api_key=api_key)
    df = preprocess_data(df)

    st.subheader("📋 Forecast Data")
    st.dataframe(df)

    with st.spinner("Running models..."):
        lr_pred, mse_lr, r2_lr = linear_regression_forecast(df)
        arima_pred = arima_forecast(df)
        prophet_pred = prophet_forecast(df)

    st.subheader("📈 Forecast Comparison")
    fig = plot_forecasts(df, lr_pred, arima_pred, prophet_pred)
    st.pyplot(fig)

    st.markdown(
        f'''
        **Model Performance**
        - Linear Regression MSE: `{mse_lr:.2f}`
        - Linear Regression R² Score: `{r2_lr:.2f}`
        '''
    )

except Exception as e:
    st.error(f"❌ Error: {e}")
