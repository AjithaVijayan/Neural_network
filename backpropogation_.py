from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from BackPropogation import BackPropogation
import pickle

dataset = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\App\IMDB Dataset.csv")

dataset['sentiment'] = dataset['sentiment'].map( {'negative': 1, 'positive': 0} )
X = dataset['review'].values
y = dataset['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
X_train = tokeniser.texts_to_sequences(X_train)
X_test = tokeniser.texts_to_sequences(X_test)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

backprop = BackPropogation(learning_rate=0.01, epochs=5, activation_function='sigmoid')
backprop.fit(X_train, y_train)
pred = backprop.predict(X_test)

with open("bp_model.pkl",'wb') as file:
    pickle.dump(backprop, file) 
with open("bp_tokeniser.pkl",'wb') as file:
    pickle.dump(tokeniser, file) 