import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="AdaBoost Regressor",
    layout="wide"
)

st.title("AdaBoost Regressor")

df = pd.read_csv("data/insurance.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

model = joblib.load("models/adaboost_regressor.pkl")
scaler = joblib.load("models/scaler.pkl")
training_columns = joblib.load("models/columns.pkl")

X = df.drop("charges", axis=1)

X = pd.get_dummies(X, drop_first=True)

for col in training_columns:
    if col not in X.columns:
        X[col] = 0

X = X[training_columns]

X = X.fillna(X.mean())

X_scaled = scaler.transform(X)

predictions = model.predict(X_scaled)

results = df.copy()
results["Predicted Charges"] = predictions

st.subheader("Prediction Results")
st.dataframe(results)

r2 = r2_score(df["charges"], predictions)

st.metric("R² Score", f"{r2:.4f}")

csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Predictions",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv"
)