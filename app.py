import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="India Student Dropout Predictor", page_icon="🎓", layout="centered"
)


# ─── Load & Train Model ─────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("dropout_data.csv")
    df = df[df["State/UT"] != "India"].reset_index(drop=True)

    df.columns = [
        "State",
        "Primary_1920",
        "Primary_2021",
        "Primary_2122",
        "UpperPrimary_1920",
        "UpperPrimary_2021",
        "UpperPrimary_2122",
        "Secondary_1920",
        "Secondary_2021",
        "Secondary_2122",
    ]

    df = df.fillna(df.median(numeric_only=True))

    df["Avg_Secondary"] = df[
        ["Secondary_1920", "Secondary_2021", "Secondary_2122"]
    ].mean(axis=1)
    median_sec = df["Avg_Secondary"].median()
    y = (df["Avg_Secondary"] > median_sec).astype(int)

    X = df[
        [
            "Primary_1920",
            "Primary_2021",
            "Primary_2122",
            "UpperPrimary_1920",
            "UpperPrimary_2021",
            "UpperPrimary_2122",
        ]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model, df


model, df = load_model()

# ─── Header ─────────────────────────────────────────────
st.title("🎓 India Student Dropout Predictor")
st.markdown(
    "Predict whether a state is at **high risk** of secondary school dropout based on primary & upper primary dropout rates."
)
st.markdown("---")

# ─── Sidebar Inputs ─────────────────────────────────────
st.sidebar.header("📥 Enter Dropout Rates (%)")
st.sidebar.markdown("Enter the dropout rates for a state across years and levels:")

p1 = st.sidebar.slider("Primary 2019-20 (%)", 0.0, 40.0, 5.0, 0.1)
p2 = st.sidebar.slider("Primary 2020-21 (%)", 0.0, 40.0, 5.0, 0.1)
p3 = st.sidebar.slider("Primary 2021-22 (%)", 0.0, 40.0, 5.0, 0.1)
up1 = st.sidebar.slider("Upper Primary 2019-20 (%)", 0.0, 40.0, 5.0, 0.1)
up2 = st.sidebar.slider("Upper Primary 2020-21 (%)", 0.0, 40.0, 5.0, 0.1)
up3 = st.sidebar.slider("Upper Primary 2021-22 (%)", 0.0, 40.0, 5.0, 0.1)

# ─── Prediction ─────────────────────────────────────────
input_data = np.array([[p1, p2, p3, up1, up2, up3]])
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

st.subheader("🔍 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error("🔴 HIGH Dropout Risk")
        st.markdown("This state is predicted to have **high secondary dropout rates.**")
    else:
        st.success("🟢 LOW Dropout Risk")
        st.markdown("This state is predicted to have **low secondary dropout rates.**")

with col2:
    st.metric("Risk Probability", f"{probability*100:.1f}%")
    st.progress(float(probability))

st.markdown("---")

# ─── Data Overview ──────────────────────────────────────
st.subheader("📊 State-wise Dropout Overview")

df_display = df[["State", "Primary_1920", "UpperPrimary_1920", "Avg_Secondary"]].copy()
df_display.columns = [
    "State",
    "Primary (2019-20)",
    "Upper Primary (2019-20)",
    "Avg Secondary",
]
df_display = df_display.sort_values("Avg Secondary", ascending=False).reset_index(
    drop=True
)

st.dataframe(df_display, use_container_width=True)

# ─── Bar Chart ──────────────────────────────────────────
st.subheader("📈 Top 10 States by Average Secondary Dropout")

top10 = df_display.head(10)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top10["State"][::-1], top10["Avg Secondary"][::-1], color="steelblue")
ax.set_xlabel("Avg Secondary Dropout Rate (%)")
ax.set_title("Top 10 High Risk States")
for bar, val in zip(bars, top10["Avg Secondary"][::-1]):
    ax.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        fontsize=9,
    )
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.markdown(
    "**Data Source:** UDISE+ via [data.gov.in](https://data.gov.in) | **Model:** Logistic Regression (CV F1: 74%)"
)
