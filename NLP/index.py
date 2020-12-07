import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = [
    "I love you","I love my dog", "I love me"
]

tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
print(word_index)
print(padded)
print(sequences)
print(tokenizer)