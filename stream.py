import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(message):
    model=load_model('keras/model.h5')
    with open('keras/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    x_1 = tokenizer.texts_to_sequences([message])
    x_1 = pad_sequences(x_1, maxlen=300)
    predictions = model.predict(x_1)[0][0]
    return predictions

st.title("Sentiment Analyzer")
message = st.text_area("Enter Text","Type Here ..")

if st.button("Predict"):
 with st.spinner('Analyzing the text …'):
     prediction=predict(message)
     if prediction >= 0.6:
         st.success(f"Positive! With {round(prediction*100, 2)}% confidence")
         st.balloons()
     elif prediction <= 0.4:
         st.error(f"Negative! With {round(prediction*100, 2)}% confidence")
     else:
         st.warning("Not sure man ¯\_(ツ)_/¯")
