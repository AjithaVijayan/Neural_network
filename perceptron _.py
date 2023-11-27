from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from Perceptron import  Perceptron
import pickle


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dataset = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\App\IMDB Dataset.csv")

dataset['sentiment'] = dataset['sentiment'].map( {'negative': 1, 'positive': 0} )
X = dataset['review'].values
y = dataset['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
X_train = tokeniser.texts_to_sequences(X_train)
X_test = tokeniser.texts_to_sequences(X_test)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

perceptron = Perceptron(epochs=10,activation_function='sigmoid')

perceptron.fit(X_train, y_train)
pred = perceptron.predict(X_test)

print(f"Accuracy : {accuracy_score(pred, y_test)}")
report = classification_report(pred, y_test, digits=2)

print(report)

with open("ppn_model.pkl",'wb') as file:
    pickle.dump(perceptron, file) 
with open("ppn_tokeniser.pkl",'wb') as file:
    pickle.dump(tokeniser, file)     
