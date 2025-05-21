
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt

LAT, LON = -37.8136, 144.9631

def fetch_weather_data(api_key, lat=LAT, lon=LON):
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&units=metric&appid={api_key}"
    response = requests.get(url)
    data = response.json()

    if "daily" in data:
        records = []
        for day in data["daily"]:
            dt = datetime.fromtimestamp(day["dt"]).strftime('%Y-%m-%d')
            temp = day["temp"]["day"]
            humidity = day["humidity"]
            condition = day["weather"][0]["description"]
            records.append([dt, temp, humidity, condition])
        return pd.DataFrame(records, columns=["Date", "Day Temp (°C)", "Humidity (%)", "Condition"])
    else:
        raise ValueError("Failed to fetch weather data")

def preprocess_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayIndex"] = np.arange(len(df))
    return df

def linear_regression_forecast(df):
    X = df[["DayIndex", "Humidity (%)"]]
    y = df["Day Temp (°C)"]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    return predictions, mse, r2

def arima_forecast(df):
    y = df["Day Temp (°C)"]
    model = ARIMA(y, order=(2, 1, 1))
    fitted = model.fit()
    forecast = fitted.predict(start=1, end=len(y)-1, typ='levels')
    return forecast

def prophet_forecast(df):
    prophet_df = df[["Date", "Day Temp (°C)"]].rename(columns={"Date": "ds", "Day Temp (°C)": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

def plot_forecasts(df, lr_pred, arima_pred, prophet_pred):
    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Day Temp (°C)"], label="Actual Temp", marker="o")
    plt.plot(df["Date"], lr_pred, label="Linear Regression", linestyle="--")
    plt.plot(df["Date"][1:], arima_pred, label="ARIMA", linestyle="-.")
    plt.plot(prophet_pred["ds"], prophet_pred["yhat"], label="Prophet", linestyle=":")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Melbourne Temperature Forecast - Comparison")
    plt.grid(True)
    plt.tight_layout()
    return plt
