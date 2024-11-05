# scripts/rnn_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def create_rnn_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))  # CNN features
    fe1 = Dense(256, activation='relu')(inputs1)

    inputs2 = Input(shape=(max_length,))  # Sequences of words
    se1 = Embedding(vocab_size, 256)(inputs2)
    se2 = LSTM(256)(se1)

    decoder1 = tf.keras.layers.add([fe1, se2])
    outputs = Dense(vocab_size, activation='softmax')(decoder1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

if __name__ == "__main__":
    vocab_size = 1000  # Change as per your dataset
    max_length = 30  # Change as per your dataset
    model = create_rnn_model(vocab_size, max_length)
    model.summary()
