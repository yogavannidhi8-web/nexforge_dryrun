import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Set page layout
st.set_page_config(page_title="NexForge Risk Triage", layout="wide")

# Load models and features
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
    st.error(f"Critical Error: Missing model files. {e}")
    st.stop()

st.title("🛡️ NexForge Risk Triage")
st.markdown("Automated Risk Detection using Hybrid AI (Isolation Forest + Autoencoder).")

uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")

if uploaded_file is not None:
    try:
        # 1. Read and Clean Data
        df = pd.read_csv(uploaded_file)
        if 'risk_level' in df.columns:
            df = df.drop(columns=['risk_level'])
            
        # 2. Align Data (Match training features)
        processed_df = df.reindex(columns=feature_names, fill_value=0)
        
        # 3. Transform
        x_new = preprocessor.transform(processed_df)
        
        # 4. Hybrid Prediction Logic
        # A) Isolation Forest Prediction
        iso_pred = iso_forest.predict(x_new)
        
        # B) Autoencoder Reconstruction Error (Mean Squared Error)
        reconstructed = autoencoder.predict(x_new)
        mse = np.mean(np.power(x_new - reconstructed, 2), axis=1)
        
        # Define threshold (90th percentile of errors = High Risk)
        threshold = np.percentile(mse, 90)
        
        # C) Combine: High Risk if Isolation Forest flags (-1) OR Error > Threshold
        df['Risk_Alert'] = ["High Risk" if (pred == -1 or err > threshold) else "Safe" 
                            for pred, err in zip(iso_pred, mse)]
        
        # 5. Output
        st.success("Analysis Complete using Hybrid AI!")
        st.dataframe(df)
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Results", csv_data, "risk_analysis_results.csv", "text/csv")
        
    except Exception as e:
        st.error(f"Processing Error: {e}")
