
from __future__ import print_function
import json
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = 'yelp_100_5.txt'
text = open(path).read().lower()
print('corpus length:', len(text))

char_indices = json.loads(open('char_indices.txt').read())
indices_char = json.loads(open('indices_char.txt').read())
chars = sorted(char_indices.keys())
print(indices_char)
#chars = sorted(list(set(text)))
print('total chars:', len(chars))
#char_indices = dict((c, i) for i, c in enumerate(chars))
#indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 256
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(LSTM(512, return_sequences=False))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.002)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.load_weights("transfer_weights")

def sample(preds, temperature=.6):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    x = np.zeros((1, maxlen, len(chars)))
    preds = model.predict(x, verbose=0)[0]
    
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)
    #start_index = char_indices["{"]

    for diversity in [0.2, 0.4, 0.6, 0.8]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            #print(next_index)
            #print (indices_char)
            next_char = indices_char[str(next_index)]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    model.save_weights("transfer_weights")

