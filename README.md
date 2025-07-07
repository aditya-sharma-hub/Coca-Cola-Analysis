# 📈 Coca-Cola Stock Price Prediction using Machine Learning

A comprehensive machine learning project that predicts future stock prices of **Coca-Cola (KO)** using historical time-series data and financial indicators. This end-to-end project covers data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## 🚀 Project Highlights

- 📊 **EDA & Visualization**  
  Analyze trends, moving averages, and volatility with rich visual insights.

- 🛠️ **Feature Engineering**  
  Constructed lag features, rolling statistics, and returns for model learning.

- 🤖 **Modeling**  
  Trained and evaluated:
  - Linear Regression
  - Random Forest Regressor

- 📉 **Evaluation Metrics**  
  Root Mean Squared Error (RMSE) used for performance comparison.

- 🧪 **Ready for Expansion**  
  Prepared for deep learning (LSTM), integration with financial indicators, and dashboard deployment (Streamlit).

---

## 📁 Dataset

- `Coca-Cola_stock_history.csv`: Daily OHLCV stock data
- `Coca-Cola_stock_info.csv`: Company financial attributes (for future use)

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming Language |
| Pandas, NumPy | Data Manipulation |
| Matplotlib, Seaborn | Visualization |
| Scikit-learn | Machine Learning |
| Jupyter Notebook | Interactive Development |

---

## 📌 Key Features Engineered

- `Close_Lag1`: Previous day closing price  
- `Daily_Return`: Daily return percentage  
- `Volatility`: 7-day rolling standard deviation  
- `MA20`, `MA50`: 20 and 50-day moving averages  

---

## 📉 Results

| Model | RMSE |
|-------|------|
| Linear Regression | ~🧮 (Displayed in Notebook) |
| Random Forest | ~🧮 (Displayed in Notebook) |

📌 *Random Forest outperformed Linear Regression in this project.*

---

## 🛠️ Future Work

- Integrate LSTM or GRU models for sequence prediction  
- Add macroeconomic or financial ratio features  
- Build a Streamlit dashboard for interactive predictions  
- Automate data fetching using APIs (e.g., Yahoo Finance)

---

## 🧠 Author

**Aditya Sharma**  
_Machine Learning Enthusiast | Data Science Explorer_

---
