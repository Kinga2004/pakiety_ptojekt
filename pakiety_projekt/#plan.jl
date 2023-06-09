### opis:wstęp o fake news, jak działa machine learning, jak trenuje sie kod, testuje sie dane
    # kod: kod, szukanie słów(?), przygotowanie danych/tekstow, podzia na data for training i data for testing

#START python:

using Pkg
Pkg.add("Python.call")
using PythonCall

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.preprocessing.text import one_hot

tkl = pyimport("tensorflow.keras.layers")
pad_sequences = pyimport("pad_sequences")
tkm = pyimport("tensorflow.keras.models")
Sequential = pyimport("Sequential")
tkpt = pyimpory("tensorflow.keras.preprocessing.text")
one_hot = pyimport("one_hot")

using CSV
using DataFrames

df = CSV.read("C:/Users/admmass/Desktop/test.csv", DataFrame)
test = CSV.read("C:/Users/admmass/Desktop/test.csv", DataFrame)

df = coalesce.(df, "")
test = coalesce.(test, "")

df.total = string.(df.title, " ", df.author)
test.total = string.(test.title, " ", test.author)

X = select!(df, Not(:id))
# y = df.id

voc_size = 5000
msg = copy(X)
msg_test = copy(test)

y=df['label']
print(X.shape)
print(y.shape)

using TextAnalysis
using Pkg

# Pkg.add("PyCall")
# using Pkg
# Pkg.add("Conda")
# using Conda
# Conda.add("nltk")

using PyCall
@pyimport nltk
nltk.download("stopwords")

stop_words = nltk.corpus.stopwords.words("english")

stemmer = nltk.stem.PorterStemmer()

corpus = []

for document in eachrow(msg)
    text = lowercase(replace(document.total, r"\W" => " "))
    words = split(text)
    stemmed_words = [stemmer.stem(word) for word in words if !(word in stop_words)]
    push!(corpus,stemmed_words)
end

corpus_test = []
for document in eachrow(msg_test)
    text = lowercase(replace(document.total, r"\W" => " "))
    words = split(text)
    stemmed_words = [stemmer.stem(word) for word in words if !(word in stop_words)]
    push!(corpus_test, stemmed_words)
end

#poniżej jest python

# Converting to one hot representation
onehot_rep = [one_hot(words,voc_size)for words in corpus]
onehot_rep_test = [one_hot(words,voc_size)for words in corpus_test]


#Padding Sentences to make them of same size
embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=25)
embedded_docs_test = pad_sequences(onehot_rep_test,padding='pre',maxlen=25)

#We have used embedding layers with LSTM
model = Sequential()
model.add(Embedding(voc_size,40,input_length=25))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

#Converting into numpy array
X_final = np.array(embedded_docs)
y_final = np.array(y)
test_final = np.array(embedded_docs_test)
X_final.shape,y_final.shape,test_final.shape

model.fit(X_final,y_final,epochs=20,batch_size=64)

y_pred = model.predict_classes(test_final)

final_sub = pd.DataFrame()
final_sub['id']=test['id']
final_sub['label'] = y_pred
final_sub.to_csv('final_sub.csv',index=False)


final_sub.head()

