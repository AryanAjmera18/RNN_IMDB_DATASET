# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit App
st.set_page_config(page_title="🎬 IMDB Review Sentiment", layout="centered")
st.title('🎥 IMDB Movie Review Sentiment Analysis')
st.caption('Classify a movie review as **Positive** or **Negative** using a pre-trained RNN model!')

st.markdown("---")
st.subheader("📝 Enter your movie review:")

# User input
user_input = st.text_area('Your Review', placeholder="Type your review here...")

if st.button('🚀 Predict Sentiment'):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before submitting!")
    else:
        with st.spinner('Analyzing your review...'):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            sentiment = '😊 Positive' if prediction[0][0] > 0.5 else '😞 Negative'
            score = prediction[0][0]

            st.success(f"### Sentiment Prediction: {sentiment}")
            st.metric(label="Prediction Confidence", value=f"{score:.2f}")

            st.progress(int(score * 100))
else:
    st.info('💬 Waiting for your input...')

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
