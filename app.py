import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="NexForge Risk Triage", layout="wide")

@st.cache_resource
def load_assets():
    # Load your assets
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
        df = pd.read_csv(uploaded_file)
        if 'risk_level' in df.columns:
            df = df.drop(columns=['risk_level'])
        
        # --- THE FIX: MANUAL ALIGNMENT ---
        # 1. Create a dataframe with ONLY the expected features
        # 2. .reindex creates the exact columns needed
        # 3. .values converts it to a clean numpy array
        
        # Step A: Align to the feature list
        df_aligned = df.reindex(columns=feature_names, fill_value=0)
        
        # Step B: Transform using preprocessor
        x_new = preprocessor.transform(df_aligned)
        
        # Step C: ENSURE it is the right shape before prediction
        # If it's still 3199, we slice it manually to 3161
        if x_new.shape[1] != 3161:
            x_new = x_new[:, :3161]
        
        # Now predict
        iso_pred = iso_forest.predict(x_new)
        
        df['Risk_Alert'] = ["High Risk" if x == -1 else "Safe" for x in iso_pred]
        
        st.success("Analysis Complete!")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Processing Error: {e}")
