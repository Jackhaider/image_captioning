import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# Define paths
DATASET_PATH = r'D:\AI_Project\image_captioning_project\data\Flickr8k_Tepm_Dataset'  # Updated path
TEXT_PATH = r'D:\AI_Project\image_captioning_project\data\captions_temp.txt'  # Updated path

# Load captions
def load_captions(text_path):
    captions = {}
    try:
        with open(text_path, 'r') as file:
            for line in file.readlines():
                image_id, caption = line.split('\t')
                image_id = image_id.split('.')[0]  # Remove file extension
                caption = caption.strip().lower()  # Normalize captions
                if image_id in captions:
                    captions[image_id].append(caption)
                else:
                    captions[image_id] = [caption]
    except Exception as e:
        print(f"An error occurred while loading captions: {e}")
    return captions

# Load images
def load_images(dataset_path):
    images = []
    try:
        for filename in os.listdir(dataset_path):
            if filename.endswith('.jpg'):
                img_path = os.path.join(dataset_path, filename)
                img = Image.open(img_path)
                images.append(img)
    except Exception as e:
        print(f"An error occurred while loading images: {e}")
    return images

# Preprocess captions
def preprocess_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions.values())  # Fit tokenizer on all captions
    vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
    return tokenizer, vocab_size

# Prepare sequences
def create_sequences(tokenizer, captions, max_length):
    X, y = [], []
    for image_id, caption_list in captions.items():
        for caption in caption_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                X.append(in_seq)
                y.append(out_seq)
    return np.array(X), np.array(y)

# Main execution
if __name__ == '__main__':
    # Load captions and images
    captions = load_captions(TEXT_PATH)
    images = load_images(DATASET_PATH)

    # Proceed only if captions are loaded
    if captions:
        # Preprocess captions
        tokenizer, vocab_size = preprocess_captions(captions)

        # Set max length for captions
        max_length = max(len(caption.split()) for caption_list in captions.values() for caption in caption_list)

        # Create input and output sequences
        X, y = create_sequences(tokenizer, captions, max_length)

        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training data shape:", X_train.shape)
        print("Validation data shape:", X_val.shape)
        print("Vocabulary size:", vocab_size)
        print("Max sequence length:", max_length)
    else:
        print("No captions loaded, exiting.")
