import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Load your model (Keep these in the same folder as app.py)
p = joblib.load('preprocessor.pkl')
i = joblib.load('model_iso.pkl')
a = tf.keras.models.load_model('model_autoencoder.keras')
feats = joblib.load('feature_names.pkl')

st.title("🛡️ NexForge Risk Triage")

# Create the Interface
uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    data = pd.DataFrame(df, columns=feats).fillna(0)
    
    # Analysis
    x_new = p.transform(data)
    iso_pred = i.predict(x_new)
    
    # Show Results
    df['Risk_Alert'] = ["High" if x == -1 else "Safe" for x in iso_pred]
    st.write(df)