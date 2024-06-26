import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model('lstm_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the max sequence length
max_sequence_len = model.input_shape[1]  # This should be the same value as used during training

# Function to predict the next words
def predict_next_words(model, tokenizer, text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted):
                predicted_word = word
                break
        text += " " + predicted_word
    return text

# Streamlit app
st.title("Next Word Prediction")
input_text = st.text_input("Enter the initial text:")
num_words = st.number_input("Enter the number of words to predict:", min_value=1, max_value=100, value=1)

if st.button("Predict"):
    if input_text:
        predicted_text = predict_next_words(model, tokenizer, input_text, num_words)
        st.write(predicted_text)
    else:
        st.write("Please enter some text to start prediction.")



#in terminal
streamlit run app.py


