import streamlit as st
import pandas as pd
import joblib
import pickle


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")



# Load model and features
model = joblib.load("model.pkl")
features = pickle.load(open("features.pkl", "rb"))

# Create empty dataframe with feature columns
sample_df = pd.DataFrame(columns=features)

# Convert to CSV
csv_template = sample_df.to_csv(index=False).encode("utf-8")


# Sidebar
st.sidebar.title("⚙️ Instructions")
st.sidebar.write("""
1. Upload CSV file
2. Ensure columns match training data
3. View predictions
4. Download results
""")

# Main title
st.title("🏦 Credit Card Fraud Detection Dashboard")

st.subheader("Step 1 — Download Template")

st.download_button(
    label="Download Sample CSV Template",
    data=csv_template,
    file_name="sample_upload_template.csv",
    mime="text/csv"
)



st.markdown("Upload transaction data to detect fraudulent activity.")

uploaded_file = st.file_uploader("📂 Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)    

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    try:
        input_data = data[features]

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        data["Prediction"] = predictions
        data["Fraud Probability"] = probabilities
        data["Prediction Label"] = data["Prediction"].map({0: "Legitimate", 1: "Fraud"})

        fraud_count = (data["Prediction"] == 1).sum()
        legit_count = (data["Prediction"] == 0).sum()

        col1, col2 = st.columns(2)

        col1.metric("🚨 Fraud Transactions", fraud_count)
        col2.metric("✅ Legitimate Transactions", legit_count)

        if fraud_count > 0:
            st.error("⚠️ Fraud detected in uploaded file!")
        else:
            st.success("✔ No fraud detected.")

        st.subheader("🔎 Prediction Results")
        st.dataframe(data, use_container_width=True)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Results", csv, "predictions.csv")

    except Exception as e:
        st.error(f"Error: {e}")
