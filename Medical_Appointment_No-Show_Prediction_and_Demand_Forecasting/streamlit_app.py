
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Healthcare Intelligence Platform", layout="wide")

st.title("🏥 Healthcare Intelligence Platform")

menu = st.sidebar.radio("Navigation",
    ["No‑Show Prediction","CSV Batch Scoring","Models Explainability",
     "Demand & Specialty Forecast"])

@st.cache_resource
def load_all():
    clf = joblib.load("/mount/src/medical-appointment-no-show-prediction-demand-forecasting/no_show_model.pkl")
    enc = joblib.load("/mount/src/medical-appointment-no-show-prediction-demand-forecasting/encoders.pkl")
    demand = joblib.load("/mount/src/medical-appointment-no-show-prediction-demand-forecasting/demand_forecast_model_v2.pkl")
    demand_features = joblib.load("/mount/src/medical-appointment-no-show-prediction-demand-forecasting/demand_features_v2.pkl")
    return clf, enc, demand, demand_features

clf, encoders, demand_model, demand_features = load_all()



# ---------- utilities ----------
def build_row(user_vals):
    cols = list(clf.feature_names_in_)
    df = pd.DataFrame(np.zeros((1,len(cols))), columns=cols)
    for k,v in user_vals.items():
        if k in df.columns:
            df.loc[0,k] = v
    return df

def encode_frame(base):
    for c in base.columns:
        if base[c].dtype == object and c in encoders:
            try:
                base[c] = encoders[c].transform(base[c].astype(str))
            except:
                base[c] = 0
    return base

# ---------- MODULE 1 ----------
if menu=="No‑Show Prediction":
    st.markdown('<div class="card">Single Patient Risk Estimation</div>',unsafe_allow_html=True)
    st.write("")
    gender_opts = list(encoders.get("gender",[]).classes_) if "gender" in encoders else ["F","M","I"]
    spec_opts = list(encoders.get("specialty",[]).classes_) if "specialty" in encoders else ["general"]
    place_opts = list(encoders.get("place",[]).classes_) if "place" in encoders else ["center"]

    gender = st.selectbox("Gender",gender_opts)
    age = st.slider("Age",0,100,30)
    specialty = st.selectbox("Specialty",spec_opts)
    place = st.selectbox("Place",place_opts)
    rainy = st.selectbox("Rainy Day Before",[0,1])
    storm = st.selectbox("Storm Day Before",[0,1])

    vals={"gender":gender,"age":age,"specialty":specialty,
          "place":place,"rainy_day_before":rainy,"storm_day_before":storm}

    base = encode_frame(build_row(vals))

    if st.button("Predict Risk"):
        proba = clf.predict_proba(base)[:,1][0]
        risk = "HIGH RISK" if proba>0.45 else "LOW RISK"
        st.metric("No‑Show Probability",round(float(proba),3))
        st.success(risk)

# ---------- MODULE 2 ----------
elif menu == "CSV Batch Scoring":
    st.markdown('<div class="card">Upload CSV with same columns as training</div>', unsafe_allow_html=True)

    f = st.file_uploader("Upload patient file", type=["csv"])
    if f:
        df = pd.read_csv(f)
        st.write("Preview", df.head())

        # --------- VERIFIED PREPARATION LOGIC ---------
        # Align exactly to model schema
        X = df.reindex(columns=clf.feature_names_in_, fill_value=0)

        # Handle categorical encoding safely (same as your working script)
        for c in X.columns:
            if c in encoders:
                X[c] = X[c].fillna("Unknown")
                X[c] = X[c].astype(str)

                known = set(encoders[c].classes_)
                X.loc[~X[c].isin(known), c] = encoders[c].classes_[0]

                X[c] = encoders[c].transform(X[c])

        # Final numeric safety
        X = X.fillna(0)

        # --------- BULK PREDICTION ---------
        probs = clf.predict_proba(X)[:, 1]

        df["no_show_probability"] = probs
        df["risk"] = df["no_show_probability"].apply(lambda x: "HIGH" if x > 0.45 else "LOW")

        # --------- PAGINATION ---------
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        total = len(df)
        pages = int(np.ceil(total / page_size))

        page = st.number_input("Page", 1, pages, 1)

        start = (page - 1) * page_size
        end = start + page_size

        st.dataframe(df.iloc[start:end], height=500)
        st.caption(f"Showing {start+1}-{min(end,total)} of {total}")

        # Single clean download button
        st.download_button(
            "Download Scored File",
            df.to_csv(index=False),
            "scored_output.csv"
        )

# ---------- MODULE 3 ----------
# ---------- MODULE 3 ----------
elif menu == "Models Explainability":

    st.markdown('<div class="section-header">📊 Model Explainability & Insights</div>', unsafe_allow_html=True)

    # ======================================================
    # 1️⃣ NO-SHOW: FEATURE IMPORTANCE
    # ======================================================

    st.subheader("Top Drivers of No-Show")

    if hasattr(clf, "feature_importances_"):

        imp = pd.DataFrame({
            "Feature": clf.feature_names_in_,
            "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        imp = imp.set_index("Feature")
        st.bar_chart(imp)

        st.caption("Higher importance indicates stronger influence on no-show prediction.")

    # ======================================================
    # 2️⃣ INTERACTIVE AGE × SPECIALTY HEATMAP
    # ======================================================

    import plotly.express as px

    st.subheader("Patient Distribution by Age Group & Specialty")

    try:
        df = pd.read_csv("Medical_appointment_data.csv")

        df['age_group'] = pd.cut(
            df['age'],
            bins=[0,18,30,45,60,120],
            labels=["0-18","19-30","31-45","46-60","60+"]
        )

        pivot = pd.crosstab(df['age_group'], df['specialty'])

        pivot_reset = pivot.reset_index().melt(
            id_vars="age_group",
            var_name="Specialty",
            value_name="Appointments"
        )

        fig = px.density_heatmap(
            pivot_reset,
            x="Specialty",
            y="age_group",
            z="Appointments",
            color_continuous_scale="Reds",
            text_auto=True
        )

        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.info("Age or specialty data unavailable.")

    # ======================================================
    # 3️⃣ DEMAND MODEL FIT (OFFICIAL TEST METRICS)
    # ======================================================

    st.subheader("Demand Model Performance (Test Set)")

    try:
        # Load saved official metrics
        metrics = joblib.load("demand_model_metrics.pkl")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("R² (Test Set)", f"{metrics['R2']:.3f}")

        with c2:
            st.metric("MAE (Test Set)", f"{metrics['MAE']:.2f}")

        with c3:
            st.metric("RMSE (Test Set)", f"{metrics['RMSE']:.2f}")

    except:
        st.warning("Test metrics file not found.")

    # ======================================================
    # 4️⃣ FULL HISTORY FIT (VISUAL ONLY)
    # ======================================================

    st.subheader("Historical Model Fit")

    try:
        df = pd.read_csv("Medical_appointment_data.csv")
        df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])

        daily = (
            df.groupby('appointment_date_continuous')
            .size()
            .reset_index(name='demand')
            .sort_values('appointment_date_continuous')
            .reset_index(drop=True)
        )

        if len(daily) > 40:

            daily['trend'] = np.arange(len(daily))
            daily['weekday'] = daily['appointment_date_continuous'].dt.weekday
            daily['weekday_sin'] = np.sin(2*np.pi*daily['weekday']/7)
            daily['weekday_cos'] = np.cos(2*np.pi*daily['weekday']/7)
            daily['month'] = daily['appointment_date_continuous'].dt.month

            daily['lag_1']  = daily['demand'].shift(1)
            daily['lag_7']  = daily['demand'].shift(7)
            daily['lag_14'] = daily['demand'].shift(14)
            daily['lag_30'] = daily['demand'].shift(30)

            daily['roll_mean_7']   = daily['demand'].rolling(7).mean().shift(1)
            daily['roll_median_7'] = daily['demand'].rolling(7).median().shift(1)
            daily['roll_std_7']    = daily['demand'].rolling(7).std().shift(1)
            daily['roll_mean_14']  = daily['demand'].rolling(14).mean().shift(1)
            daily['roll_std_14']   = daily['demand'].rolling(14).std().shift(1)
            daily['roll_std_30']   = daily['demand'].rolling(30).std().shift(1)

            daily['ema_7']  = daily['demand'].ewm(span=7, adjust=False).mean().shift(1)
            daily['ema_14'] = daily['demand'].ewm(span=14, adjust=False).mean().shift(1)

            daily['diff_1'] = daily['demand'].shift(1) - daily['demand'].shift(7)
            daily['diff_7'] = daily['demand'].shift(7) - daily['demand'].shift(14)

            daily = daily.dropna().reset_index(drop=True)

            X = daily[demand_model.feature_names_in_]
            y_actual = daily['demand']

            pred_log = demand_model.predict(X)
            y_pred = np.expm1(pred_log)

            plot_df = pd.DataFrame({
                "Actual": y_actual.values,
                "Predicted": y_pred
            }, index=daily['appointment_date_continuous'])

            st.line_chart(plot_df)


    except:
        st.info("Demand visualization unavailable.")
    # ======================================================
    # 4️⃣ LAG RELATIONSHIP
    # ======================================================

    st.subheader("Lag Relationship (Yesterday vs Today Demand)")

    try:
        scatter_df = pd.DataFrame({
            "Lag_1": daily["lag_1"],
            "Today_Demand": daily["demand"]
        })

        st.scatter_chart(scatter_df)

        st.caption("Strong linear trend indicates temporal dependency.")

    except:
        st.info("Lag analysis unavailable.")

# ---------- MODULE 4 ----------
elif menu == "Demand & Specialty Forecast":

    st.markdown('<div class="section-header">📈 Demand Forecasting Engine</div>', unsafe_allow_html=True)

    df = pd.read_csv("Medical_appointment_data.csv")
    df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])

    col1, col2 = st.columns([1,1])

    with col1:
        days = st.slider("Forecast Horizon (Days)", 7, 60, 14)

    with col2:
        specialty_choice = st.selectbox(
            "Filter by Specialty",
            ["All"] + sorted(df['specialty'].dropna().unique().tolist())
        )

    data = df.copy()

    if specialty_choice != "All":
        data = data[data['specialty'] == specialty_choice]

    # =============================
    # BUILD DAILY SERIES
    # =============================
    daily = (
        data.groupby('appointment_date_continuous')
        .size()
        .reset_index(name='demand')
        .sort_values('appointment_date_continuous')
        .reset_index(drop=True)
    )

    if len(daily) < 40:
        st.warning("Not enough historical data for forecasting.")
        st.stop()

    daily['trend'] = np.arange(len(daily))

    # =============================
    # FORECAST LOGIC
    # =============================
    history = daily.copy()
    current_date = history['appointment_date_continuous'].iloc[-1]

    preds = []
    dates = []

    model_features = demand_model.feature_names_in_

    for i in range(days):

        current_date += pd.Timedelta(days=1)
        weekday = current_date.weekday()

        new_row = {
            'weekday_sin': np.sin(2*np.pi*weekday/7),
            'weekday_cos': np.cos(2*np.pi*weekday/7),
            'month': current_date.month,
            'trend': history['trend'].iloc[-1] + 1,
            'lag_1': history['demand'].iloc[-1],
            'lag_7': history['demand'].iloc[-7],
            'lag_14': history['demand'].iloc[-14],
            'lag_30': history['demand'].iloc[-30],
            'roll_mean_7': history['demand'].iloc[-7:].mean(),
            'roll_median_7': history['demand'].iloc[-7:].median(),
            'roll_std_7': history['demand'].iloc[-7:].std(),
            'roll_mean_14': history['demand'].iloc[-14:].mean(),
            'roll_std_14': history['demand'].iloc[-14:].std(),
            'roll_std_30': history['demand'].iloc[-30:].std(),
            'ema_7': history['demand'].ewm(span=7, adjust=False).mean().iloc[-1],
            'ema_14': history['demand'].ewm(span=14, adjust=False).mean().iloc[-1],
            'diff_1': history['demand'].iloc[-1] - history['demand'].iloc[-7],
            'diff_7': history['demand'].iloc[-7] - history['demand'].iloc[-14],
        }

        X_input = pd.DataFrame([new_row])
        X_input = X_input.reindex(columns=model_features, fill_value=0)

        pred_log = demand_model.predict(X_input)[0]
        pred = max(0, int(round(np.expm1(pred_log))))

        preds.append(pred)
        dates.append(current_date)

        history = pd.concat([
            history,
            pd.DataFrame({
                'appointment_date_continuous':[current_date],
                'demand':[pred],
                'trend':[new_row['trend']]
            })
        ], ignore_index=True)

    forecast_df = pd.DataFrame({
        "Date": dates,
        "Predicted_Appointments": preds
    })

    # =============================
    # CHART 1: HISTORICAL + FORECAST
    # =============================

    st.markdown('<div class="section-header">Demand Trend Visualization</div>', unsafe_allow_html=True)

    recent_actual = daily.tail(60).copy()
    recent_actual = recent_actual.set_index('appointment_date_continuous')
    recent_actual = recent_actual[['demand']]
    recent_actual.columns = ['Actual']

    forecast_plot = forecast_df.set_index('Date')
    forecast_plot = forecast_plot[['Predicted_Appointments']]
    forecast_plot.columns = ['Forecast']

    combined_chart = pd.concat([recent_actual, forecast_plot], axis=0)

    st.line_chart(combined_chart)

    # =============================
    # CHART 2: FORECAST ONLY (DAY-BY-DAY CLEAR VIEW)
    # =============================

    st.markdown('<div class="section-header">Detailed Forecast Curve</div>', unsafe_allow_html=True)

    forecast_only = forecast_df.set_index("Date")
    forecast_only.columns = ["Forecasted Appointments"]

    st.line_chart(forecast_only)

    # =============================
    # KPI CARDS
    # =============================

    st.markdown('<div class="section-header">Capacity Planning Insights</div>', unsafe_allow_html=True)

    avg = int(forecast_df['Predicted_Appointments'].mean())
    peak = int(forecast_df['Predicted_Appointments'].max())
    total = int(forecast_df['Predicted_Appointments'].sum())

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f'<div class="metric-card"><h3>{avg}</h3><p class="small-text">Avg Daily Demand</p></div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="metric-card"><h3>{peak}</h3><p class="small-text">Peak Expected</p></div>', unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="metric-card"><h3>{total}</h3><p class="small-text">Total Forecast Period</p></div>', unsafe_allow_html=True)

    # =============================
    # TABLE + DOWNLOAD
    # =============================

    st.markdown('<div class="section-header">Forecast Table</div>', unsafe_allow_html=True)
    st.dataframe(forecast_df)

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False),
        file_name="demand_forecast.csv"
    )
    
