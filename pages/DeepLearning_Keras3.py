# Import PyTorch

from __future__ import print_function, division
import tensorflow_datasets as tfds
import streamlit as st
import cv2

# Display the first image
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras

import tensorflow as tf

from keras import layers
from keras import layers, models
import os
import time
import copy


def imshow(img, tab):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    tab.pyplot(plt)


#______________________________________________________________________________________________

st.set_page_config(page_title="DeepLearningCV on Streamlit", page_icon="🎛️",layout="wide")
st.sidebar.markdown("### DeepLearningCV Part 2 🎛️")
#______________________________________________________________________________________________
# Streamlit UI components
st.title("Deep Learning CV with Streamlit Part 2")
tab1,tab3,tab4,tab5 = st.tabs(["21. Keras Transfer Learning and Fine Tuning","25. Keras - Google Deep Dream","27. Keras - Neural Style Transfer + TF-HUB Models", "29. Keras Autoencoders"])  # ,"G","R","Hue"

#______________________________________________________________________________________________
# Allow user to input the number of epochs
batch_size = st.sidebar.number_input("Number of batch_size:", min_value=1, max_value=1000, value=128, step=1)
epochs = st.sidebar.number_input("Number of epochs:", min_value=1, max_value=100, value=2, step=1)
momentum = st.sidebar.number_input("Number of momentum:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
lr = st.sidebar.number_input("Number of lr:", min_value=0.000, max_value=0.010, value=0.001, step=0.0001)

# ----------------------------------------------------------------------------------------------------------------------
# "21. Keras Transfer Learning and Fine Tuning"
Learning_Tuning = tab1.selectbox("Keras Transfer Learning and Fine Tuning", ["Trainable Layers", "Step 1. Load a base model with pre-trained weights (ImageNet)","Introduce some random data augmentation","Now let's Train our Top Layer","4. Fine Tuning"])
Learning_Tuning_options = tab1.expander("Keras Transfer Learning and Fine Tuning")
Learning_Tuning_inputs = tab1.expander("Results : ")
placeholder1 = tab1.empty()

# Define a Dense layer
layer = keras.layers.Dense(4)
# Initialize weights for the layer
layer.build((None, 2))

# Load the dataset
tfds.disable_progress_bar()
train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,
)

# Resize and batch the dataset
size = (150, 150)
batch_size = batch_size

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y)).cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y)).cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y)).cache().batch(batch_size).prefetch(buffer_size=10)

# Define data augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
])

# Display some images from the training set
tab1.subheader('Sample Images from Training Dataset')
def plot_images(images, labels):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, (ax, img, label) in enumerate(zip(axes.flat, images, labels)):
        ax.imshow(img)
        ax.set_title('Cat' if int(label) == 0 else 'Dog')
        ax.axis("off")
    tab1.pyplot(fig)

if Learning_Tuning == "Trainable Layers":

    tab1.write(f'Number of weights: {len(layer.weights)}')
    tab1.write(f'Number of trainable_weights: {len(layer.trainable_weights)}')
    tab1.write(f'Number of non_trainable_weights: {len(layer.non_trainable_weights)}')

    # Define a model with two Dense layers
    layer1 = keras.layers.Dense(3, activation="relu")
    layer2 = keras.layers.Dense(3, activation="sigmoid")
    model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])

    # Freeze the first layer
    layer1.trainable = False

    # Store the initial weights of layer1
    initial_layer1_weights_values = layer1.get_weights()

    # Compile and train the model
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

    # Check if the weights of layer1 remain unchanged
    final_layer1_weights_values = layer1.get_weights()

    if np.array_equal(initial_layer1_weights_values[0], final_layer1_weights_values[0]):
        tab1.write('Weights unchanged (weights)')

    if np.array_equal(initial_layer1_weights_values[1], final_layer1_weights_values[1]):
        tab1.write('Weights unchanged (biases)')



if Learning_Tuning == "Step 1. Load a base model with pre-trained weights (ImageNet)":
    # Load the dataset
    tfds.disable_progress_bar()
    train_ds, validation_ds, test_ds = tfds.load(
        "cats_vs_dogs",
        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
        as_supervised=True,  # Include labels
    )

    # Calculate the number of samples in each split
    num_train_samples = tf.data.experimental.cardinality(train_ds).numpy()
    num_validation_samples = tf.data.experimental.cardinality(validation_ds).numpy()
    num_test_samples = tf.data.experimental.cardinality(test_ds).numpy()

    tab1.subheader('Dataset Statistics')
    tab1.write(f'Number of training samples: {num_train_samples}')
    tab1.write(f'Number of validation samples: {num_validation_samples}')
    tab1.write(f'Number of test samples: {num_test_samples}')


if Learning_Tuning == "Introduce some random data augmentation":
    # Get some sample images and labels
    for images, labels in train_ds.take(1):
        tab1.write(f'Number of samples: {len(images)}')
        plot_images(images.numpy().astype("uint8"), labels.numpy())

    # Display augmented images
    tab1.subheader('Augmented Images')


    def plot_augmented_images(image, label):
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        augmented_images = [data_augmentation(tf.expand_dims(image, 0))[0] for _ in range(9)]
        for ax, aug_img in zip(axes.flat, augmented_images):
            ax.imshow(aug_img.numpy().astype("uint8"))
            ax.axis("off")
        tab1.pyplot(fig)


    # Get one image to show augmented versions
    for images, labels in train_ds.take(1):
        plot_augmented_images(images[0], labels[0])


if Learning_Tuning == "Now let's Train our Top Layer":
    # Base model with pre-trained weights
    base_model = tf.keras.applications.Xception(
        weights="imagenet",
        input_shape=(150, 150, 3),
        include_top=False,
    )
    base_model.trainable = False

    # Model definition
    inputs = layers.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)
    scale_layer = layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    tab1.write('Training model...')
    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    # Display training progress
    tab1.line_chart(history.history['loss'])
    tab1.line_chart(history.history['binary_accuracy'])

    tab1.write('Unfreezing base model and continuing training...')
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    history_finetune = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    # Display training progress
    tab1.line_chart(history.history['loss'])
    tab1.line_chart(history.history['binary_accuracy'])

if Learning_Tuning == "4. Fine Tuning":

    tab1.write('Unfreezing base model and continuing training...')
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    history_finetune = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    # Display fine-tuning progress
    tab1.line_chart(history_finetune.history['loss'])
    tab1.line_chart(history_finetune.history['binary_accuracy'])

    tab1.write('Training complete!')
    tab1.write('Model summary:')
    tab1.text(model.summary())

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# "25. Keras - Google Deep Dream"
Deep_Dream = tab3.selectbox("Keras - Google Deep Dream", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Deep_Dream_options = tab3.expander("Keras - Google Deep Dream")
Deep_Dream_inputs = tab3.expander("Results : ")
placeholder3 = tab3.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "27. Keras - Neural Style Transfer + TF-HUB Models"
Cats_Dogs = tab4.selectbox("Keras - Neural Style Transfer + TF-HUB Models", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Cats_Dogs_options = tab4.expander("Keras - Neural Style Transfer + TF-HUB Models")
Cats_Dogs_inputs = tab4.expander("Results : ")
placeholder4 = tab4.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "29. Keras Autoencoders"
Lightening = tab5.selectbox("Keras Autoencoders", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Lightening_options = tab5.expander("Keras Autoencoders")
Lightening_inputs = tab5.expander("Results : ")
placeholder5 = tab5.empty()
