import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("gru_model.h5")
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Main title
st.title("🎬 Movie Review Sentiment Analysis")

# Description
st.write("""
This AI model predicts whether a movie review is:

✅ Positive  
❌ Negative
""")

# Text input
review = st.text_area(
    "Enter Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)

# Predict button
predict_button = st.button("Predict Sentiment")

if predict_button:

    # Step 1: Input validation
    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review before predicting.")
    else:

        # Convert text into sequence
        sequence = tokenizer.texts_to_sequences([review])

        # Apply padding
        padded_sequence = pad_sequences(
            sequence,
            maxlen=200,
            padding='post',
            truncating='post'
        )

        # Make prediction
        prediction = model.predict(padded_sequence)

        # Output result
        if predict_button:
            if review.strip() == "":
                st.warning("⚠️ Please enter a movie review before predicting.")
            else:
                sequence = tokenizer.texts_to_sequences([review])
                padded_sequence = pad_sequences(
                    sequence,
                    maxlen=200,
                    padding='post',
                    truncating='post'
                )
                score = model.predict(padded_sequence)[0][0]

                # --- UI OUTPUT ---
                st.subheader("🔍 Prediction Result")
                
                if score >= 0.5:
                    sentiment = "😊 Positive Review"
                    confidence = score * 100
                    st.success(sentiment)
                else:
                    sentiment = "😡 Negative Review"
                    confidence = (1 - score) * 100
                    st.error(sentiment)
                
                st.write(f"**Confidence Score:** {confidence:.2f}%")
                
                st.subheader("📊 Confidence Level")
                
                st.progress(float(confidence / 100))