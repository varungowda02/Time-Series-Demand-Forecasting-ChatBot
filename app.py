
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels as sms
import matplotlib.pyplot as plt

#------------------------------------------
# LOAD & PREPARE DATA
#------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dwoo-work/time-series-demand-forecasting/main/src/sales_data_sample_utf8.csv"
    sales = pd.read_csv(url)
    sales = sales.drop_duplicates()

    sales['ORDERDATE'] = pd.to_datetime(sales['ORDERDATE'])
    sales['date'] = pd.to_datetime(sales['ORDERDATE'].dt.strftime("%Y-%m-%d"))
    sales['week'] = sales.date.dt.isocalendar().week
    sales['month'] = sales.date.dt.month
    sales['year'] = sales.date.dt.year

    sales['motorcycles_QUANTITYORDERED'] = sales.loc[sales['PRODUCTLINE'] == 'Motorcycles', 'QUANTITYORDERED']
    time_series = sales.groupby(['week','month','year']).agg(
        date=('date','first'),
        motorcycles_total_qty_ordered=('motorcycles_QUANTITYORDERED', np.sum)
    ).reset_index().sort_values('date')

    time_series['date'] = pd.to_datetime(time_series['date'])
    time_series = time_series.set_index('date')

    monthly_series = time_series.motorcycles_total_qty_ordered.resample('M').sum()
    return monthly_series

monthly_series = load_data()

#------------------------------------------
# BUILD FORECAST MODELS
#------------------------------------------
def forecast_arima(series, steps=12):
    model = sm.tsa.statespace.SARIMAX(series, order=(2,1,0), seasonal_order=(0,2,0,12))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=steps)
    return results, forecast

def forecast_expo(series, steps=12):
    model = sms.tsa.holtwinters.ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=12
    )
    results = model.fit()
    forecast = results.forecast(steps)
    return results, forecast

#------------------------------------------
# STREAMLIT CHATBOT
#------------------------------------------
st.set_page_config(page_title="Demand Forecasting Chatbot", layout="centered")

st.title("📈 Demand Forecasting Chatbot")
st.write("Ask me about motorcycle demand forecasts!")

# Session state for conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat input
if prompt := st.chat_input("Ask something like: Forecast next 12 months"):
    st.session_state["messages"].append({"role": "user", "content": prompt})

# Process messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle last user query
if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    query = st.session_state["messages"][-1]["content"].lower()

    if "arima" in query:
        with st.chat_message("assistant"):
            st.write("Running ARIMA forecast for next 12 months...")
            results, forecast = forecast_arima(monthly_series, steps=12)

            fig, ax = plt.subplots(figsize=(10,4))
            monthly_series.plot(ax=ax, label="Actual")
            forecast.predicted_mean.plot(ax=ax, label="Forecast")
            ax.legend()
            st.pyplot(fig)

            # Show forecasted values
            st.write("### 📊 Predicted Values (ARIMA)")
            forecast_df = forecast.predicted_mean.reset_index().rename(
                columns={"index":"Date", "predicted_mean":"Forecasted_Demand"}
            )
            st.dataframe(forecast_df)

            # Download button
            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Forecast (CSV)", csv, "arima_forecast.csv", "text/csv")

    elif "expo" in query or "exponential" in query:
        with st.chat_message("assistant"):
            st.write("Running Exponential Smoothing forecast for next 12 months...")
            results, forecast = forecast_expo(monthly_series, steps=12)

            fig, ax = plt.subplots(figsize=(10,4))
            monthly_series.plot(ax=ax, label="Actual")
            forecast.plot(ax=ax, label="Forecast")
            ax.legend()
            st.pyplot(fig)

            # Show forecasted values
            st.write("### 📊 Predicted Values (Exponential Smoothing)")
            forecast_df = forecast.reset_index().rename(
                columns={"index":"Date", 0:"Forecasted_Demand"}
            )
            st.dataframe(forecast_df)

            # Download button
            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Forecast (CSV)", csv, "expo_forecast.csv", "text/csv")

    elif "forecast" in query:
        with st.chat_message("assistant"):
            st.write("I can use ARIMA or Exponential Smoothing. Please type:")
            st.write("- `forecast using arima`")
            st.write("- `forecast using expo`")

    else:
        with st.chat_message("assistant"):
            st.write("Sorry, I didn't understand. Try asking: *Forecast using ARIMA* or *Forecast using Expo*")

