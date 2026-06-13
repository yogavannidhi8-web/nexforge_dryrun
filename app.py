import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Load assets
@st.cache_resource
def load_assets():
    p = joblib.load('preprocessor.pkl')
    i = joblib.load('model_iso.pkl')
    a = tf.keras.models.load_model('model_autoencoder.keras')
    feats = joblib.load('feature_names.pkl')
    return p, i, a, feats

preprocessor, iso_forest, autoencoder, feature_names = load_assets()

st.title("🛡️ NexForge Risk Triage")
uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")

if uploaded_file is not None:
    try:
        # 1. Read the file
        df = pd.read_csv(uploaded_file)
        
        # 2. Get the raw values (ignoring headers entirely)
        # We only take the first 3161 columns available
        data_values = df.values[:, :3161] 
        
        # 3. If the file has fewer than 3161 columns, pad it with zeros
        if data_values.shape[1] < 3161:
            padding = np.zeros((data_values.shape[0], 3161 - data_values.shape[1]))
            data_values = np.hstack((data_values, padding))
            
        # 4. Transform using preprocessor
        x_new = preprocessor.transform(data_values)
        
        # 5. Predict using Isolation Forest
        iso_pred = iso_forest.predict(x_new)
        
        # 6. Predict using Autoencoder
        reconstructed = autoencoder.predict(x_new)
        mse = np.mean(np.power(x_new - reconstructed, 2), axis=1)
        threshold = np.percentile(mse, 90)
        
        # 7. Final Alert
        df['Risk_Alert'] = ["High Risk" if (pred == -1 or err > threshold) else "Safe" 
                            for pred, err in zip(iso_pred, mse)]
        
        st.success("Analysis Complete!")
        st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"Processing Error: {e}")
