import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Set page layout
st.set_page_config(page_title="NexForge Risk Triage", layout="wide")

# Load models and features with caching for speed
@st.cache_resource
def load_assets():
    p = joblib.load('preprocessor.pkl')
    i = joblib.load('model_iso.pkl')
    a = tf.keras.models.load_model('model_autoencoder.keras')
    feats = joblib.load('feature_names.pkl')
    return p, i, a, feats

# Attempt to load, display errors if file is missing
try:
    preprocessor, iso_forest, autoencoder, feature_names = load_assets()
except Exception as e:
    st.error(f"Critical Error: Missing model files. {e}")
    st.stop()

st.title("🛡️ NexForge Risk Triage")
st.markdown("Upload your transaction CSV. The system will automatically align your data to the required 3161 features.")

# File Uploader
uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")

if uploaded_file is not None:
    try:
        # 1. Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # 2. Cleanup: Remove the risk label if it exists in the test file
        if 'risk_level' in df.columns:
            df = df.drop(columns=['risk_level'])
            
        # 3. CRITICAL ALIGNMENT: Force columns to match 3161 training features
        # This solves the '3199 vs 3161' ValueError by dropping extra cols and adding missing ones
        processed_df = df.reindex(columns=feature_names, fill_value=0)
        
        # 4. Transform and Predict
        x_new = preprocessor.transform(processed_df)
        iso_pred = iso_forest.predict(x_new)
        
        # 5. Output results
        df['Risk_Alert'] = ["High Risk" if x == -1 else "Safe" for x in iso_pred]
        
        st.success("Analysis Complete!")
        st.dataframe(df.head(10)) # Show preview
        
        # Download button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Results", csv_data, "analysis_results.csv", "text/csv")
        
    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.write("Ensure your CSV contains the correct column headers.")
