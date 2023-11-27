import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy import argmax
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
import pickle

dataset = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\App\IMDB Dataset.csv")

dataset['sentiment'] = dataset['sentiment'].map( {'negative': 1, 'positive': 0} )
X = dataset['review'].values
y = dataset['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
X_train = tokeniser.texts_to_sequences(X_train)
X_test = tokeniser.texts_to_sequences(X_test)

vocab_size = len(tokeniser.word_index)+1

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, padding = 'post')
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding = 'post')

n_features = X_train.shape[1]

#Modelling a sample DNN
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(500,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

opt=Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=50, batch_size=16)
loss, acc = model.evaluate(X_test, y_test)

with open("dnn_model.pkl",'wb') as file:
    pickle.dump(model, file) 

with open("dnn_tokeniser.pkl",'wb') as file:
    pickle.dump(tokeniser, file) 
