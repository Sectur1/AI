import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 

tokenizer = Tokenizer()
data = ""

with open("lauv.txt","r") as f:
    data = f.read()
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1
#break the each line in the corpus
input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)
#pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding="pre"))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels,num_classes=total_words)

model = Sequential([
    Embedding(total_words,100,input_length=max_sequence_len-1),
    Bidirectional(LSTM(150)),
    Dense(total_words,activation="softmax"),
])

model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.01),metrics=["accuracy"])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=20, verbose=1)

# seed_text = "I've got a bad feeling about this"
next_words = 100
phrases = ["need you", "without you","thinking about you","my everything","kiss","love","me","us","we"]
for seed_text in phrases:
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	with open("lovetexts.txt","a") as f:
		f.write(seed_text+"\n\r")
