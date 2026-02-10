# Medical-Appointment-No-Show-Prediction-Demand-Forecasting

# 🏥 Healthcare Intelligence Platform

An end-to-end analytics platform for predicting **appointment no-shows** and **hospital demand forecasting** using machine learning and Streamlit.

---

## 🚀 Features

### 1. No-Show Prediction
- Single patient risk estimation  
- Batch scoring via CSV upload  
- Model explainability (feature importance)  
- Risk categorization: HIGH / LOW

### 2. Demand Forecasting
- Daily appointment volume forecasting  
- Specialty-wise filtering  
- Recursive time-series prediction using ML model  
- Business metrics: peak, average, total demand

---
## 🌐 Access the Application

The application is deployed on Streamlit Cloud:

👉 **https://medical-appointment-no-show-prediction-demand-forecasting-synz.streamlit.app/**

No installation required – open the link and start using.

## 🧠 Models Used

| Module | Model Type | Purpose |
|------|------------|---------|
| No-Show Prediction | Classification (Tree-based) | Predict probability of patient missing appointment |
| Demand Forecasting | Regression (XGBoost / Tree) | Predict daily appointment count |

---

## 📂 Project Structure
├── streamlit_dashboard.py
├── demand_model_daily.pkl
├── no_show_model.pkl
├── encoders.pkl
├── Medical_appointment_data.csv
├── README.md

---

## 📊 Data Dictionary

| Column | Description |
|------|-------------|
| gender | Patient gender |
| age | Age in years |
| specialty | Department visited |
| place | Hospital location |
| rainy_day_before | Weather indicator |
| storm_day_before | Severe weather indicator |
| appointment_date_continuous | Date of appointment |
| demand | Daily appointment count |
| lag_1, lag_7 | Previous demand values |
| weekday | Day of week |
| month | Month of year |

---

## 🧩 How Forecast Works

• Forecast is TOTAL appointments per day  
• Model is seeded from the last real observed demand  
• Specialty filter enables department-level planning  
• Uses training features: weekday, month, day, lag variables  

---
## 📌 Usage Workflow

1. Select module from sidebar  
2. Enter patient details → Predict risk  
3. Upload CSV for bulk scoring  
4. Navigate to Demand Forecast  
5. Select horizon & specialty  
6. Download forecast CSV

---

## 🛠 Future Improvements

- Add SHAP explainability  
- Real weather API integration  
- Doctor-wise forecasting  
- Appointment rescheduling optimizer  

---

## 👩‍💻 Author

**Jyoti Bharadwaj**  
Data Analytics & Machine Learning Enthusiast

---
