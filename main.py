import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('grammer.csv')
text = [x for x,y in zip(df.input.values, df.labels.values) if y==1] 

corpus = [line.split(" ") for line in text]
corpus = corpus[:2000]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
# print(tokenizer.word_index)

total_words = len(tokenizer.word_index) + 1
print(f"total words:{total_words}")

sequences = tokenizer.texts_to_sequences(corpus)
# padded = pad_sequences(sequences,padding="post")

input_sequences = []
labels = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i]
        input_sequences.append(n_gram_sequence)
        labels.append(token_list[i])

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,maxlen = max_sequence_len, padding = 'pre'))

xs = input_sequences
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# #train Neural Network
model = Sequential()
model.add(Embedding(input_dim =total_words, output_dim=240,input_shape =(max_sequence_len,)))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words,activation = "softmax"))

# print(model.summary())
adam = Adam(learning_rate=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=40)

# # save model
# # model.save('main.keras')
# # del model
# # model = load_model('main.keras')
# model.save_weights('my_model.weights.h5')  # to store
# model.load_weights('my_model.weights.h5')

# plot history
# import matplotlib.pyplot as plt

# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.xlabel("Epochs")
#     plt.ylabel(string)
#     plt.show()

# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')

# #driver code
while True:
    seed_text = input("Enter sentence:")
    if seed_text == "q00": #enter q00 to stop
        break

    next_words = 10

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        seed_text+= " " + output_word
        if output_word == '.':
            break

    print(seed_text)

    
