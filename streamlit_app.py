
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
    clf = joblib.load("no_show_model.pkl")
    enc = joblib.load("encoders.pkl")
    demand = joblib.load("demand_forecast_model.pkl")
    return clf, enc, demand

clf, encoders, demand_model = load_all()

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
elif menu=="Models Explainability":

    st.header("📊 Model Explainability & Insights")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # ======================================================
    # 1. NO-SHOW: FEATURE IMPORTANCE
    # ======================================================
    with col1:
        st.subheader("No-Show Drivers")

        if hasattr(clf, "feature_importances_"):

            imp = pd.DataFrame({
                "feature": clf.feature_names_in_,
                "importance": clf.feature_importances_
            }).sort_values("importance", ascending=False).head(8)

            fig = plt.figure(figsize=(5,4))
            plt.barh(imp["feature"], imp["importance"])
            plt.title("Top Factors Influencing No-Show")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            st.caption("Higher = stronger impact on missing appointment")


    # ======================================================
    # 2. NO-SHOW: RISK BY AGE × SPECIALTY
    # ======================================================
    with col2:
        st.subheader("Risk Pattern by Age & Specialty")

        try:
            df = pd.read_csv("Medical_appointment_data.csv")

            # ----- CREATE AGE GROUPING INSIDE STREAMLIT -----
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0,18,30,45,60,120],
                labels=[
                    "Child (0-18)",
                    "Young Adult (19-30)",
                    "Adult (31-45)",
                    "Middle Age (46-60)",
                    "Senior (60+)"
                ]
            )

            # ----- CREATE PIVOT TABLE -----
            pivot = pd.crosstab(df['age_group'], df['specialty'])

            # ----- PLOT -----
            fig = plt.figure(figsize=(8,5))
            sns.heatmap(
                pivot,
                cmap='Reds',
                annot=True,
                fmt='d'
            )

            plt.title("Age Group vs Specialty Volume")
            plt.ylabel("Age Group")
            plt.xlabel("Specialty")

            st.pyplot(fig)

        except Exception as e:
            st.info("Age or specialty data not available")


    # ======================================================
    # 3. DEMAND: ACTUAL VS PREDICTED
    # ======================================================
    with col3:
        st.subheader("Demand Model Fit")

        try:
            import joblib

            demand_model = joblib.load("demand_forecast_model.pkl")

            df = pd.read_csv("Medical_appointment_data.csv")
            df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])

            daily = df.groupby('appointment_date_continuous').size().reset_index(name='demand')
            daily = daily.sort_values('appointment_date_continuous')

            daily['weekday'] = daily['appointment_date_continuous'].dt.dayofweek
            daily['month']   = daily['appointment_date_continuous'].dt.month
            daily['day']     = daily['appointment_date_continuous'].dt.day

            daily['lag_1'] = daily['demand'].shift(1)
            daily['lag_7'] = daily['demand'].shift(7)

            daily = daily.dropna()

            X = daily[["weekday","month","day","lag_1","lag_7"]]
            y = daily["demand"]

            # ---- FULL RANGE PLOT ----
            pred = demand_model.predict(X)

            fig3 = plt.figure(figsize=(7,4))

            plt.plot(daily['appointment_date_continuous'], y.values, label="Actual")
            plt.plot(daily['appointment_date_continuous'], pred, label="Predicted")

            plt.title("Demand Forecast vs Actual")
            plt.legend()
            plt.xticks(rotation=30)

            st.pyplot(fig3)

        except Exception as e:
            st.info("Train data required for comparison")


    # ======================================================
    # 4. DEMAND: LAG RELATIONSHIP
    # ======================================================
    with col4:
        st.subheader("Lag Relationship")

        try:
            fig4 = plt.figure(figsize=(5,4))
            plt.scatter(daily["lag_1"], daily["demand"], alpha=0.5)
            plt.title("Yesterday vs Today Demand")
            plt.xlabel("Lag 1")
            plt.ylabel("Today")
            st.pyplot(fig4)

        except:
            st.info("Lag analysis requires dataset")


    st.markdown("""
    **How to interpret**

    • No-Show charts → help target reminders  
    • Demand charts → validate forecasting logic  
    • Lag view → proves time dependency  
    """)
# elif menu=="Visualizations":
#     st.markdown('<div class="card">Model Drivers</div>',unsafe_allow_html=True)

#     if hasattr(clf,"feature_importances_"):
#         imp=pd.DataFrame({
#             "feature":clf.feature_names_in_,
#             "importance":clf.feature_importances_
#         }).sort_values("importance",ascending=False).head(10)

#         fig=plt.figure()
#         plt.barh(imp["feature"],imp["importance"])
#         plt.title("Top Influencing Factors")
#         st.pyplot(fig)

#         st.write("""
#         Interpretation Guide  
#         • Higher importance = stronger influence on no‑show  
#         • Weather & Location dominates  
#         • Use for targeted reminders
#         """)

# ---------- MODULE 4 ----------

elif menu == "Demand & Specialty Forecast":

    st.header("📈 Demand Forecasting")
    df = pd.read_csv("Medical_appointment_data.csv")

    df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])


    # ===========================================
    # USER CONTROLS
    # ===========================================

    col1, col2 = st.columns(2)

    with col1:
        days = st.slider("Forecast Horizon (Days)", 7, 60, 14)

    with col2:
        specialty_choice = st.selectbox(
            "Filter by Specialty",
            ["All"] + sorted(df['specialty'].dropna().unique().tolist())
        )

    # ===========================================
    # LOAD DATA
    # ===========================================

    data = df.copy()
    data['appointment_date_continuous'] = pd.to_datetime(data['appointment_date_continuous'])

    # ----- Specialty Filter -----
    if specialty_choice != "All":
        data = data[data['specialty'] == specialty_choice]

    # ===========================================
    # BUILD TRUE DAILY DEMAND SERIES
    # ===========================================

    daily = (
        data
        .groupby('appointment_date_continuous')
        .size()
        .reset_index(name='demand')
        .sort_values('appointment_date_continuous')
    )

    st.subheader("Recent Actual Demand")
    st.dataframe(daily.tail(10))

    # ===========================================
    # FEATURE ENGINEERING (SAME AS TRAINING)
    # ===========================================

    daily['weekday'] = daily['appointment_date_continuous'].dt.dayofweek
    daily['month']   = daily['appointment_date_continuous'].dt.month
    daily['day']     = daily['appointment_date_continuous'].dt.day

    daily['lag_1'] = daily['demand'].shift(1)
    daily['lag_7'] = daily['demand'].shift(7)

    daily = daily.dropna()

    # ===========================================
    # SEED FROM REAL LAST VALUES
    # ===========================================

    last_row = daily.iloc[-1]
    last_7   = daily.iloc[-7]

    last = pd.DataFrame({
        "weekday": [last_row['weekday']],
        "month":   [last_row['month']],
        "day":     [last_row['day']],
        "lag_1":   [last_row['demand']],
        "lag_7":   [last_7['demand']]
    })

    # ===========================================
    # RECURSIVE FORECASTING
    # ===========================================

    preds = []
    dates = []

    current_date = last_row['appointment_date_continuous']

    for i in range(days):

        p = demand_model.predict(last)[0]
        p = max(0, int(round(p)))        # no negative demand

        preds.append(p)

        current_date = current_date + pd.Timedelta(days=1)
        dates.append(current_date)

        # --- Update lags ---
        last["lag_7"] = last["lag_1"]
        last["lag_1"] = p

        last["weekday"] = (last["weekday"] + 1) % 7
        last["month"]   = current_date.month
        last["day"]     = current_date.day

    # ===========================================
    # RESULTS DISPLAY
    # ===========================================

    forecast_df = pd.DataFrame({
        "Date": dates,
        "Predicted_Appointments": preds
    })

    st.subheader("🔮 Forecast Output")
    st.dataframe(forecast_df)

    # ----- Visualization -----
    st.subheader("Forecast Curve")
    st.line_chart(forecast_df.set_index("Date"))

    # ===========================================
    # BUSINESS METRICS
    # ===========================================

    st.subheader("📊 Capacity Insights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Average Daily Demand", int(forecast_df['Predicted_Appointments'].mean()))

    with c2:
        st.metric("Peak Demand", int(forecast_df['Predicted_Appointments'].max()))

    with c3:
        st.metric("Total Next Period", int(forecast_df['Predicted_Appointments'].sum()))

    # ===========================================
    # DOWNLOAD
    # ===========================================

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False),
        file_name="demand_forecast.csv"
    )

    st.info("""
    How to read this:
    • Forecast is TOTAL appointments per day  
    • Based on real last demand as seed  
    • Specialty filter recalculates demand and helps in department level capacity planning
    
    """)




