import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")

# ==================================================
# CUSTOM FONT + UI STYLING
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

h1 { font-size: 55px !important; font-weight: 800; }
h2 { font-size: 38px !important; font-weight: 700; }
h3 { font-size: 28px !important; font-weight: 600; }

section[data-testid="stSidebar"] label { font-size: 18px !important; }

[data-testid="stMetricValue"] { font-size: 42px !important; }

p { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Employee Attrition Business Intelligence & Prediction Dashboard")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader("Upload Employee Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    original_df = pd.read_csv(uploaded_file)
    df = original_df.copy()

    # ==================================================
    # CREATE BUSINESS COLUMNS
    # ==================================================

    if "MonthlyIncome" in df.columns:
        df["Salary Band"] = pd.cut(
            df["MonthlyIncome"],
            bins=[0, 4000, 8000, df["MonthlyIncome"].max()],
            labels=["Low", "Medium", "High"]
        )

    if "YearsAtCompany" in df.columns:
        df["Tenure Group"] = pd.cut(
            df["YearsAtCompany"],
            bins=[-1, 2, 5, df["YearsAtCompany"].max()],
            labels=["0–2 Years", "3–5 Years", "6+ Years"]
        )

    # ==================================================
    # SIDEBAR — DATA PROCESSING
    # ==================================================

    st.sidebar.header("⚙️ Data Processing")

    if st.sidebar.checkbox("Remove Duplicate Rows"):
        df = df.drop_duplicates()

    missing_option = st.sidebar.selectbox(
        "Handle Missing Values",
        ["None", "Drop Rows with Missing", "Fill Numeric with Mean"]
    )

    if missing_option == "Drop Rows with Missing":
        df = df.dropna()

    elif missing_option == "Fill Numeric with Mean":
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())

    # ==================================================
    # SIDEBAR — CORE BUSINESS FILTERS
    # ==================================================

    st.sidebar.header("🔎 Core Business Filters")

    filter_columns = [
        "Gender","Department","JobRole","EducationField","MaritalStatus",
        "BusinessTravel","OverTime","Salary Band","Tenure Group",
        "WorkLifeBalance","JobSatisfaction","EnvironmentSatisfaction"
    ]

    for col in filter_columns:
        if col in df.columns:
            selected = st.sidebar.multiselect(
                f"{col}",
                sorted(df[col].dropna().unique())
            )
            if selected:
                df = df[df[col].isin(selected)]

    # ==================================================
    # KPI SECTION
    # ==================================================

    st.header("📌 Executive Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Employees (Original)", len(original_df))
    col2.metric("Current Employees", len(df))

    if "Attrition" in df.columns:
        attr_rate = (df["Attrition"] == "Yes").mean() * 100
        col3.metric("Attrition Rate (%)", round(attr_rate, 2))
    else:
        col3.metric("Attrition Rate (%)", "N/A")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==================================================
    # BUSINESS INSIGHT VISUALS
    # ==================================================

    st.header("📊 Key Attrition Drivers")

    colA, colB = st.columns(2)

    if "OverTime" in df.columns and "Attrition" in df.columns:
        overtime_chart = df.groupby("OverTime")["Attrition"].apply(lambda x: (x == "Yes").mean())
        colA.subheader("Attrition by Overtime")
        colA.bar_chart(overtime_chart)

    if "Salary Band" in df.columns and "Attrition" in df.columns:
        salary_chart = df.groupby("Salary Band")["Attrition"].apply(lambda x: (x == "Yes").mean())
        colB.subheader("Attrition by Salary Band")
        colB.bar_chart(salary_chart)

    if "Tenure Group" in df.columns and "Attrition" in df.columns:
        st.subheader("Attrition by Tenure Group")
        tenure_chart = df.groupby("Tenure Group")["Attrition"].apply(lambda x: (x == "Yes").mean())
        st.line_chart(tenure_chart)

    # ==================================================
    # CUSTOM VISUALIZATION BUILDER
    # ==================================================

    st.header("📈 Custom Visualization Builder")

    chart_type = st.selectbox(
        "Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"]
    )

    x_axis = st.selectbox("Select X-Axis", df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        y_axis = st.selectbox("Select Y-Axis (Numeric)", numeric_cols)

        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis)

        st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # CORRELATION HEATMAP
    # ==================================================

    st.header("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ==================================================
    # MACHINE LEARNING SECTION
    # ==================================================

    if "Attrition" in df.columns:

        st.header("🤖 Attrition Prediction Model")

        # Remove non-useful columns
        drop_cols = ["EmployeeCount","Over18","StandardHours","EmployeeNumber"]
        model_df = original_df.drop(columns=[c for c in drop_cols if c in original_df.columns]).copy()

        # Encode categorical
        label_encoders = {}
        for col in model_df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col])
            label_encoders[col] = le

        X = model_df.drop("Attrition", axis=1)
        y = model_df["Attrition"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.metric("Model Accuracy (%)", round(accuracy * 100, 2))

        # Feature Importance
        st.subheader("Top 10 Important Features")
        importance = pd.Series(model.coef_[0], index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8,6))
        importance.tail(10).plot(kind="barh", ax=ax)
        st.pyplot(fig)

        # Prediction Input
        st.subheader("Predict Individual Employee Risk")

        input_data = {}

        for col in X.columns:
            if col in original_df.select_dtypes(include=np.number).columns:
                input_data[col] = st.slider(
                    col,
                    float(original_df[col].min()),
                    float(original_df[col].max()),
                    float(original_df[col].median())
                )
            else:
                input_data[col] = float(X[col].median())

        if st.button("Predict Attrition Risk"):
            input_df = pd.DataFrame([input_data])
            probability = model.predict_proba(input_df)[0][1]
            st.write(f"Attrition Probability: **{round(probability*100,2)}%**")

            if probability > 0.5:
                st.error("High Risk Employee ⚠️")
            else:
                st.success("Low Risk Employee ✅")

    # ==================================================
    # DOWNLOAD
    # ==================================================

    st.header("⬇️ Download Filtered Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV dataset to begin analysis.")
