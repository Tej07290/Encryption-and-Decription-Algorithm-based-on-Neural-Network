import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import Bidirectional
from keras.models import model_from_json
import pickle
import numpy as np

global char_to_int
global int_to_char
vocab_list = []
dataX = []
dataY = []
global n_vocab
global classifier

def getID(chars,data):
    index = 0
    for i in range(len(chars)):
        if chars[i] == data:
            index = i;
            break
    return index     

sentences = ''
with open('model/input.txt', "r") as file:
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        line.lower()
        sentences+=line
file.close()
sentences = sentences.strip()

vocab_list.clear()
for i in range(len(sentences)):
    vocab_list.append(sentences[i])
raw_text = sentences
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
for i in range(0, n_chars):
    dataX.append(char_to_int.get(raw_text[i]))
    dataY.append(getID(chars,raw_text[i]))
print(dataX)
print(dataY)
print(chars)




n_patterns = len(dataX)
if os.path.exists('model/nn_model1.json'):
    with open('model/nn_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    json_file.close()
    classifier.load_weights("model/nn_model_weights.h5")
    classifier._make_predict_function()
    f = open('model/nn_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    loss = data['loss']
    loss = loss[9]
    print("BI-LSTM Training Model Loss = "+str(loss)+"\n")
else:
    seq_length = 1
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    y = np_utils.to_categorical(dataY)
    print(X.shape)
    print(y.shape)
    model = Sequential()
    model.add(Bidirectional(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    hist = model.fit(X, y, epochs=10000, batch_size=64)
    model.save_weights('model/nn_model_weights.h5')            
    model_json = model.to_json()
    with open("model/nn_model.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()    
    f = open('model/nn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    loss = hist.history['loss']
    loss = loss[9]
    print("BI-LSTM Training Model Loss = "+str(loss)+"\n")
    classifier = model

line = 'welcome to java world'
output = ''
enc = []
for i in range(len(line)):
    data = char_to_int[line[i]]
    temp = []
    temp.append(data)
    temp = np.asarray(temp)
    print(temp.shape)
    x = np.reshape(temp, (1, temp.shape[0], 1))
    x = x / float(n_vocab)
    encrypted = classifier.predict(x, verbose=0)[0]
    enc.append(np.argmax(encrypted))

print("Encrypted text : "+str(enc))
enc = np.asarray(enc)
for i in range(len(enc)):
    index = enc[i]
    result = int_to_char[index]
    output+=result
    
print(output)













