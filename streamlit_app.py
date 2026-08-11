import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure src module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.io import load_pipeline
from src.config import MODELS_DIR, TARGET_MAP

# Setup
MODEL_PATH = MODELS_DIR / "cinevault_classifier_v1.joblib"
LABEL_MAP = {v: k for k, v in TARGET_MAP.items()}

st.set_page_config(page_title="Cinevault Classification Model", page_icon="🎬", layout="centered")

st.title("🎬 Cinevault Classification Model")
st.markdown("Enter the metadata of a Netflix title below to predict whether it is a **Movie** or **TV Show**.")

# Cache the model loading to improve performance
@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_pipeline(MODEL_PATH)

model = get_model()

if model is None:
    st.error(f"Model not found at `{MODEL_PATH}`. Make sure to train the model first by running `python train.py`.")
else:
    with st.form("prediction_form"):
        st.subheader("Title Metadata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            director = st.text_input("Director", value="Not Given", help="Use 'Not Given' if unknown.")
            country = st.text_input("Country", value="United States")
            date_added = st.text_input("Date Added (M/D/YYYY)", value="9/25/2021")
            
        with col2:
            release_year = st.number_input("Release Year", min_value=1900, max_value=2100, value=2020)
            rating = st.selectbox("Rating", ["PG-13", "TV-MA", "TV-14", "TV-PG", "R", "TV-Y7", "TV-Y", "PG", "TV-G", "NR", "G", "TV-Y7-FV", "NC-17", "UR"])
            
        duration = st.text_input("Duration", value="90 min", help="E.g., '90 min' or '2 Seasons'. Note: This is an important feature in the dataset.")
        listed_in = st.text_input("Listed In (Genres)", value="Documentaries", help="Comma-separated list of genres.")
        
        submit_button = st.form_submit_button(label="Predict")
        
    if submit_button:
        # Build DataFrame for prediction
        input_data = pd.DataFrame([{
            "show_id": "inference",
            "title": "inference",
            "director": director,
            "country": country,
            "date_added": date_added,
            "release_year": release_year,
            "rating": rating,
            "duration": duration,
            "listed_in": listed_in,
        }])
        
        try:
            pred = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]
            
            label = LABEL_MAP[pred]
            confidence = probs[pred] * 100
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            if label == "Movie":
                st.success(f"**Prediction:** {label} 🎥")
            else:
                st.info(f"**Prediction:** {label} 📺")
                
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            
            st.caption(f"Probabilities: Movie = {probs[1]*100:.2f}%, TV Show = {probs[0]*100:.2f}%")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
