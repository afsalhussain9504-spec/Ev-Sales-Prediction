import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="EV Sales Dashboard", layout="wide")
st.title("⚡ EV Sales Analysis & Prediction Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("IEA Global EV Data 2024 new.csv")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Clean value column (very important)
    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Clean text columns
    df["parameter"] = df["parameter"].str.strip().str.lower()
    df["region"] = df["region"].str.strip()
    df["mode"] = df["mode"].str.strip()
    df["powertrain"] = df["powertrain"].str.strip()

    df = df.dropna(subset=["value"])

    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filter Options")

region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df["region"].dropna().unique())
)

mode = st.sidebar.selectbox(
    "Select Vehicle Type",
    ["All"] + sorted(df["mode"].dropna().unique())
)

powertrain = st.sidebar.selectbox(
    "Select Powertrain",
    ["All"] + sorted(df["powertrain"].dropna().unique())
)

# ---------------- FILTERING ----------------
filtered_df = df.copy()

if region != "All":
    filtered_df = filtered_df[filtered_df["region"] == region]

if mode != "All":
    filtered_df = filtered_df[filtered_df["mode"] == mode]

if powertrain != "All":
    filtered_df = filtered_df[filtered_df["powertrain"] == powertrain]

# ---------------- PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df.head(20))

st.subheader("📊 Data Summary")
st.write(filtered_df.describe())

# ---------------- EV SALES TREND ----------------
st.subheader("📈 EV Sales Trend Over Years")

sales_df = filtered_df[
    filtered_df["parameter"] == "ev sales"
]

if not sales_df.empty:

    sales_df = (
        sales_df.groupby("year")["value"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    fig, ax = plt.subplots()
    ax.plot(sales_df["year"], sales_df["value"], marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("EV Sales (Units)")
    ax.set_title("EV Sales Growth Trend")

    st.pyplot(fig)

else:
    st.warning("No EV sales data available for selected filters.")

# ---------------- PREDICTION 2030 ----------------
st.subheader("🤖 EV Sales Prediction for 2030")

if not sales_df.empty and len(sales_df) > 1:

    X = sales_df[["year"]].astype(float)
    y = sales_df["value"].astype(float)

    model = LinearRegression()
    model.fit(X, y)

    prediction_2030 = model.predict(np.array([[2030]]))

    st.success(
        f"📌 Predicted EV Sales for 2030: "
        f"**{int(prediction_2030[0]):,} units**"
    )

    # Plot actual + prediction
    future_years = np.arange(
        sales_df["year"].max()+1,
        2031
    ).reshape(-1,1)

    future_preds = model.predict(future_years)

    fig2, ax2 = plt.subplots()
    ax2.plot(sales_df["year"], sales_df["value"], marker='o', label="Actual")
    ax2.plot(future_years, future_preds, marker='x', linestyle='--', label="Predicted")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("EV Sales")
    ax2.set_title("Actual vs Predicted EV Sales")
    ax2.legend()

    st.pyplot(fig2)

else:
    st.warning("Not enough data available for prediction.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed using Streamlit | EV Sales Analysis & Prediction Project")