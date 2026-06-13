import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Set page configuration
st.set_page_config(page_title="NexForge Risk Triage", layout="wide")

# Load models and metadata
@st.cache_resource
def load_assets():
    p = joblib.load('preprocessor.pkl')
    i = joblib.load('model_iso.pkl')
    a = tf.keras.models.load_model('model_autoencoder.keras')
    feats = joblib.load('feature_names.pkl')
    return p, i, a, feats

# Initialize assets
try:
    preprocessor, iso_forest, autoencoder, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

st.title("🛡️ NexForge Risk Triage")
st.markdown("Upload your transaction CSV to perform automated anomaly detection.")

# Interface
uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")

if uploaded_file is not None:
    try:
        # Load and process data
        df = pd.read_csv(uploaded_file)
        
        # Drop risk_level if it exists to prevent prediction leakage
        if 'risk_level' in df.columns:
            df = df.drop(columns=['risk_level'])
            
        # Clean and Align data: Force columns to match the training set exactly
        # This handles the "3199 vs 3161" feature mismatch error
        processed_df = df.reindex(columns=feature_names, fill_value=0)
        
        # Transformation
        x_new = preprocessor.transform(processed_df)
        
        # Predictions
        iso_pred = iso_forest.predict(x_new)
        
        # Add Results
        df['Risk_Alert'] = ["High Risk" if x == -1 else "Safe" for x in iso_pred]
        
        # Display Results
        st.subheader("Analysis Results")
        st.write(f"Processed {len(df)} transactions.")
        st.dataframe(df)
        
        # Download button for results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "risk_analysis_results.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
