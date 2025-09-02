# 📈 Demand Forecasting Chatbot

This project is a **Streamlit-based chatbot** that forecasts motorcycle demand using **time series models**.
It is powered by **ARIMA** and **Exponential Smoothing (Holt-Winters)** models, with interactive charts and downloadable forecast results.

---

## 🚀 Features

* Load and preprocess **motorcycle sales data**.
* Train **ARIMA** and **Exponential Smoothing** models.
* Interactive **Streamlit chatbot** interface.
* Generate **12-month forecasts**.
* Display **plots + forecasted values** in a table.
* **Download forecast results** as CSV.

---

## 📂 Project Structure

```
.
├── app.py                       # Streamlit chatbot application
├── Time_Series_Demand_Forecasting.ipynb   # Jupyter notebook with model exploration
├── README.md                    # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:

   ```bash
   pip install streamlit pandas numpy statsmodels matplotlib
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 💬 How to Use

1. Open the chatbot in your browser (Streamlit will show a local URL).
2. Type messages such as:

   * `forecast using arima`
   * `forecast using expo`
3. The app will:

   * Plot actual vs forecasted demand.
   * Show forecast values in a table.
   * Provide a **CSV download button** for forecasts.

---

## 📊 Models Used

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   Captures trend + seasonality with SARIMAX.

2. **Exponential Smoothing (Holt-Winters)**
   Handles seasonality with additive trend and seasonality.

---

## 📦 Data Source

Data comes from the [Sales Data Sample](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data).
We filter it to forecast **Motorcycle product line demand**.

---

## 🔮 Example Outputs

* Forecast plots (Actual vs Predicted)
* Forecasted values table (with dates & demand)
* Downloadable CSV file

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas / NumPy**
* **Statsmodels**
* **Matplotlib**

---

## 📌 Next Steps

* Add more models (Prophet, LSTM).
* Compare models with evaluation metrics (MAE, RMSE, MAPE).
* Extend chatbot to answer more natural questions.

--- 
Output:

<img width="1902" height="1426" alt="2025-09-02_164718" src="https://github.com/user-attachments/assets/9332f4f4-d1be-4edd-aa89-5c1a82c4b478" />



