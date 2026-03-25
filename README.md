# 📈 Coca-Cola Stock Price Prediction & Time-Series Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Python-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/Data%20Analysis-Pandas-green?style=for-the-badge&logo=pandas">
  <img src="https://img.shields.io/badge/Visualization-Matplotlib-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Project%20Status-Completed-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge">
</p>

---

### *End-to-End Machine Learning Pipeline for Financial Forecasting & Market Signal Analysis*

> **A scalable and extensible analytics system designed to model, interpret, and predict stock price movements of Coca-Cola (KO) using time-series feature engineering and machine learning techniques.**

---

## 🚀 Project Overview

Financial markets generate massive volumes of time-series data, but raw price data alone does not reveal:

- Underlying trends and momentum  
- Volatility patterns  
- Predictive signals from historical behavior  
- Model-driven future price estimation  

This project transforms historical stock data into a **predictive intelligence system** using:

- Feature engineering  
- Statistical indicators  
- Machine learning models  

👉 Delivering a **data-driven approach to stock price forecasting and analysis**

---

## 🎯 Objectives

- Perform **exploratory time-series analysis (EDA)**  
- Engineer meaningful **lag-based and rolling features**  
- Train and compare **machine learning models**  
- Evaluate predictive performance using **robust metrics**  
- Build a **foundation for advanced financial forecasting systems**

---

## 📊 Sample Visualizations

<p align="center">
  <img src="assets/price_trend.png" width="85%">
  <br>
  <em>Stock price trends with moving averages highlighting long-term patterns</em>
</p>

<p align="center">
  <img src="assets/volatility_plot.png" width="85%">
  <br>
  <em>Volatility clustering and rolling standard deviation analysis</em>
</p>

---

## 🗂️ Dataset Description

### 📌 Data Sources

- `Coca-Cola_stock_history.csv`
  - Daily OHLCV data (Open, High, Low, Close, Volume)

- `Coca-Cola_stock_info.csv`
  - Company financial metadata (for future integration)

---

### 📌 Data Characteristics

- Time-series structured data  
- Daily frequency  
- Minimal missing values  
- Suitable for supervised learning after transformation  

---

## 🛠️ Tech Stack & Tools

| Category | Tools |
|------|------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development | Jupyter Notebook |

---

## 🏗️ System Architecture

```text
Raw Stock Data
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Time-Series Transformation
     ↓
Model Training
     ↓
Evaluation (RMSE)
     ↓
Prediction & Insights
```

## 🔄 Methodology & Pipeline

### 1️⃣ Data Preprocessing
- Date parsing and indexing  
- Handling missing values  
- Sorting data chronologically  

---

### 2️⃣ Feature Engineering

Constructed predictive features from raw price signals:

| Feature | Description |
|--------|------------|
| `Close_Lag1` | Previous day closing price |
| `Daily_Return` | Percentage change in price |
| `Volatility` | 7-day rolling standard deviation |
| `MA20` | 20-day moving average |
| `MA50` | 50-day moving average |

👉 These features transform raw prices into:
- Trend indicators  
- Momentum signals  
- Risk/volatility measures  

---

### 3️⃣ Exploratory Data Analysis (EDA)

- Trend visualization of closing prices  
- Moving average crossover patterns  
- Volatility clustering analysis  
- Return distribution insights  

✔ Helps identify:
- Market trends  
- Noise vs signal  
- Temporal dependencies  

---

### 4️⃣ Modeling

#### 🔹 Linear Regression
- Baseline model  
- Captures linear relationships  

#### 🔹 Random Forest Regressor
- Ensemble-based model  
- Captures non-linear dependencies  
- Handles feature interactions effectively  

---

### 5️⃣ Evaluation

- Metric Used: **Root Mean Squared Error (RMSE)**  

| Model | Performance |
|------|------------|
| Linear Regression | Baseline |
| Random Forest | ✅ Better performance |

---

## 📊 Key Insights

- Stock prices show **strong temporal dependency**
- Moving averages capture **trend direction effectively**
- Volatility is **clustered, not random**
- Tree-based models outperform linear models in financial prediction  

---

## ⚠️ Assumptions & Limitations

- Based only on **historical price data**
- No macroeconomic or sentiment indicators included  
- Market behavior assumed to have learnable patterns  
- Predictions are **probabilistic, not guaranteed**

---

## 📈 Why This Project Matters

- Financial forecasting is complex and noisy  
- Demonstrates:
  - Time-series data handling  
  - Feature engineering expertise  
  - Practical ML model application  

👉 Bridges the gap between:  
**Raw financial data → Predictive intelligence**

---

## 🧩 Project Structure

```text
├── notebooks/
│   └── Coca_Cola_Project.ipynb
├── data/
│   ├── Coca-Cola_stock_history.csv
│   └── Coca-Cola_stock_info.csv
├── assets/
│   ├── price_trend.png
│   └── volatility_plot.png
├── README.md
```


### 💪 Strengths
- Strong feature engineering for time-series
- Scalable for advanced forecasting systems

---

### 🔮 Future Enhancements
- Deep learning models (LSTM / GRU)
- Integration of:
   - Macroeconomic indicators
   - News sentiment analysis
- Hyperparameter tuning

---

## 🎯 Key Takeaways

> **This project demonstrates how historical stock data can be transformed into a predictive modeling system using feature engineering and machine learning techniques.**

- Lag features capture temporal dependencies
- Rolling statistics encode trend and volatility
- Ensemble models improve prediction performance
- Financial data benefits from non-linear modeling approaches
---

## 🧠 Author
**Aditya Sharma**
Machine Learning Enthusiast | Data Science Explorer

---
⭐ If you found this project insightful, consider starring the repository.
---


