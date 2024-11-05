# scripts/train.py

import numpy as np
from data_preprocessing import load_data, preprocess_images, preprocess_captions
from cnn_model import create_cnn_model
from rnn_model import create_rnn_model

# Load and preprocess data
images_dir = '../data/Flickr8k_Dataset'
captions_file = '../data/captions.json'
captions = load_data(images_dir, captions_file)
images = preprocess_images(images_dir)
padded_sequences, word_index = preprocess_captions(captions)

# Create CNN and RNN models
cnn_model = create_cnn_model()
features = cnn_model.predict(images)

vocab_size = len(word_index) + 1
max_length = padded_sequences.shape[1]
rnn_model = create_rnn_model(vocab_size, max_length)

# Train the RNN model
# You will need to prepare the input data properly here
# This is a simplified example; you may need to adjust
X1 = features
X2 = padded_sequences
y = # prepare your labels (one-hot encoded if necessary)

rnn_model.fit([X1, X2], y, epochs=20, verbose=1)  # Adjust epochs as needed
