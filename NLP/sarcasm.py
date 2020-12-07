import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import json

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

with open("sarcasm.json","r") as f:
    datastore = json.load(f)

headlines = []
labels = []
urls = []

for items in datastore:
    headlines.append(items["headline"])
    labels.append(items["is_sarcastic"])
    urls.append(items["article_link"])

training_sentences = headlines[0:training_size]
testing_sentences = headlines[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size,oov_token="<OOV>")
tokenizer.fit_on_texts(headlines)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(headlines)
padded = pad_sequences(sequences,padding="post")


training_headlines = headlines[0:(int(len(headlines)/2)+1)]
testing_headlines = headlines[int(len(headlines)/2):]

training_labels = labels[0:int(len(labels)/2)+1]
testing_labels = labels[int(len(labels)/2):]

tokenizer.fit_on_texts(training_headlines)

index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_headlines)
training_padded = pad_sequences(training_sequences,maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_headlines)
test_padded = pad_sequences(training_sequences,maxlen=max_length, padding=padding_type, truncating=trunc_type)
import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(test_padded)
testing_labels = np.array(testing_labels)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid"),
])

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()
checkpoint_path = "checkpoints/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.save_weights(checkpoint_path.format(epoch=0))
latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest)
history = model.fit(training_padded, training_labels, epochs=30, validation_data=(test_padded, testing_labels), verbose=2,callbacks=[cp_callback])
model.save("models/model")
sentence = ["granny starting to fear spiders in the garden might be real", 
"Seth might have farted"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))