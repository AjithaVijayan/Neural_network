import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle


st.header('Demo')
task = st.selectbox('Select Task', ["Select One",'Sentiment Classification', 'Tumor Detection'])


if task == "Tumor Detection":
        def cnn(img, model):
            img = Image.open(img)
            img = img.resize((128, 128))
            img = np.array(img)
            input_img = np.expand_dims(img, axis=0)
            res = model.predict(input_img)
            if res:
                return "Tumor Detected"
            else:
                return "No Tumor" 
            
        cnn_model = tf.keras.models.load_model("tumor_detection_model.h5")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Submit"):
                result=cnn(uploaded_file, cnn_model)
                st.write(result)

        
elif task == "Sentiment Classification":
        types = ["Perceptron","BackPropagation", "RNN","DNN", "LSTM"]
        input_text2 = st.radio("Select", types, horizontal=True)

        if input_text2 == "Perceptron":
                with open("ppn_model.pkl",'rb') as file:
                    perceptron = pickle.load(file)
                with open("ppn_tokeniser.pkl",'rb') as file:
                    ppn_tokeniser = pickle.load(file)

                def ppn_make_predictions(inp, model):
                    encoded_inp = ppn_tokeniser.texts_to_sequences([inp])
                    padded_inp = tf.keras.preprocessing.sequence.pad_sequences(encoded_inp, maxlen=500)
                    res = model.predict(padded_inp)
                    if res:
                        return "Negative"
                    else:
                        return "Positive"       
                
                st.subheader('Movie Review Classification using Perceptron')
                inp = st.text_area('Enter message')
                if st.button('Check'):
                    pred = ppn_make_predictions([inp], perceptron)
                    st.write(pred)

        if input_text2 == "BackPropagation":
                with open("bp_model.pkl",'rb') as file:
                    backprop = pickle.load(file)
                with open("bp_tokeniser.pkl",'rb') as file:
                    bp_tokeniser = pickle.load(file)

                def bp_make_predictions(inp, model):
                    encoded_inp = bp_tokeniser.texts_to_sequences([inp])
                    padded_inp = tf.keras.preprocessing.sequence.pad_sequences(encoded_inp, maxlen=500)
                    res = model.predict(padded_inp)
                    if res:
                        return "Negative"
                    else:
                        return "Positive"     
                       
                st.subheader('Movie Review Classification using BackPropagation')
                inp = st.text_area('Enter message')
                if st.button('Check'):
                    pred = bp_make_predictions([inp], backprop)
                    st.write(pred)
        

        elif input_text2 == "RNN":
                with open("spam_model.pkl", 'rb') as model_file:
                    rnn_model=pickle.load(model_file)
                with open("spam_tokeniser.pkl", 'rb') as model_file:
                    rnn_tokeniser=pickle.load(model_file)

                def rnn_make_predictions(inp, model):
                    encoded_inp = rnn_tokeniser.texts_to_sequences(inp)
                    padded_inp = tf.keras.preprocessing.sequence.pad_sequences(encoded_inp, maxlen=10, padding='post')
                    res = (model.predict(padded_inp) > 0.5).astype("int32")
                    if res:
                        return "Spam"
                    else:
                        return "Ham"

                st.subheader('Spam message Classification using RNN')
                input = st.text_area("Give message")
                if st.button('Check'):
                    pred = rnn_make_predictions([input], rnn_model)
                    st.write(pred)



        elif input_text2 == "DNN":
                        with open("dnn_model.pkl",'rb') as file:
                            dnn_model = pickle.load(file)
                        with open("dnn_tokeniser.pkl",'rb') as file:
                            dnn_tokeniser = pickle.load(file)

                        def dnn_make_predictions(inp, model):
                            inp = dnn_tokeniser.texts_to_sequences(inp)
                            inp = tf.keras.preprocessing.sequence.pad_sequences(inp, maxlen=500)
                            res = (model.predict(inp) > 0.5).astype("int32")
                            if res:
                                return "Negative"
                            else:
                                return "Positive"       
                        
                        st.subheader('Movie Review Classification using DNN')
                        inp = st.text_area('Enter message')
                        if st.button('Check'):
                            pred = dnn_make_predictions([inp], dnn_model)
                            st.write(pred)

                            

        elif input_text2 == "LSTM":
                with open("lstm_model.pkl",'rb') as file:
                    lstm_model = pickle.load(file) 

                with open("lstm_tokeniser.pkl",'rb') as file:
                    lstm_tokeniser = pickle.load(file)

                def lstm_make_predictions(inp, model):
                    inp = lstm_tokeniser.texts_to_sequences(inp)
                    inp = tf.keras.preprocessing.sequence.pad_sequences(inp, maxlen=500)
                    res = (model.predict(inp) > 0.5).astype("int32")
                    if res:
                        return "Negative"
                    else:
                        return "Positive"
                st.subheader('Movie Review Classification using LSTM')
                inp = st.text_area('Enter message')
                if st.button('Check'):
                    pred = lstm_make_predictions([inp], lstm_model)
                    st.write(pred)  




                




