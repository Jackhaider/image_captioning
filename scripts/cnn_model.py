# scripts/cnn_model.py

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def create_cnn_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

if __name__ == "__main__":
    model = create_cnn_model()
    model.summary()
