
import json
import streamlit as st
import cv2

# Display the first image
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.scorecam import ScoreCAM
from matplotlib import cm

import io

from os import listdir
from os.path import isfile, join


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
tab1,tab2 = st.tabs(["11. CNN Visualisation GradCAM GradCAMplusplus and FasterScoreCAM",""])  # ,"G","R","Hue"


# Images.image(image_rgb, caption="Processed Image", use_column_width=True, channels="RBG")

#______________________________________________________________________________________________
# Allow user to input the number of epochs
batch_size = st.sidebar.number_input("Number of batch_size:", min_value=1, max_value=1000, value=128, step=1)
epochs = st.sidebar.number_input("Number of epochs:", min_value=1, max_value=100, value=2, step=1)
momentum = st.sidebar.number_input("Number of momentum:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
lr = st.sidebar.number_input("Number of lr:", min_value=0.000, max_value=0.010, value=0.001, step=0.0001)

# ----------------------------------------------------------------------------------------------------------------------
# "11. CNN Visualisation GradCAM GradCAMplusplus and FasterScoreCAM"
CNN_Visualisation = tab1.selectbox("CNN Visualisation GradCAM GradCAMplusplus and FasterScoreCAM", ["Grad-CAM","Grad-CAM++", "Score-CAM"])
CNN_Visualisation_options = tab1.expander("CNN Visualisation GradCAM GradCAMplusplus and FasterScoreCAM")
CNN_Visualisation_inputs = tab1.expander("Results : ")
placeholder1 = tab1.empty()

# Function to load and preprocess images using OpenCV
def load_and_preprocess_images(uploaded_files):
    images = []
    for uploaded_file in uploaded_files:
        # Read the image from the file object
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize the image to the target size
        img_resized = cv2.resize(img, (224, 224))

        # Convert image to float32 and preprocess for the model
        img_preprocessed = img_resized.astype('float32')
        img_preprocessed = preprocess_input(img_preprocessed)

        images.append(img_preprocessed)

    return np.asarray(images)


# Streamlit app
tab1.title("Grad-CAM, Grad-CAM++, and Score-CAM Visualization")

# File uploader for images
uploaded_files = tab1.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)
if uploaded_files:
    images = load_and_preprocess_images(uploaded_files)
    image_titles = [uploaded_file.name for uploaded_file in uploaded_files]

    # Load model
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)

    # Display input images
    tab1.write("### Input Images")
    num_images = len(images)
    fig, ax = plt.subplots(1, num_images, figsize=(15, 5))

    if num_images == 1:
        ax = [ax]

    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=14)
        ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) / 255.0)  # Convert BGR to RGB for display
        ax[i].axis('off')
    tab1.pyplot(fig)

    # Model modification function
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m

    # Loss function
    def loss(output):
        return [output[i][np.argmax(output[i])] for i in range(len(output))]


    if CNN_Visualisation == "Grad-CAM":
        # Grad-CAM
        tab1.write("### Grad-CAM")
        gradcam = Gradcam(model, model_modifier=model_modifier, clone=False)
        cam = gradcam(loss, images, penultimate_layer=-1)
        cam = normalize(cam)

        fig, ax = plt.subplots(1, num_images, figsize=(15, 5))
        if num_images == 1:
            ax = [ax]
        for i, title in enumerate(image_titles):
            heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
            ax[i].set_title(title, fontsize=14)
            ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) / 255.0)  # Convert BGR to RGB for display
            ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[i].axis('off')
        tab1.pyplot(fig)

    if CNN_Visualisation == "Grad-CAM++":
        # Grad-CAM++
        tab1.write("### Grad-CAM++")
        gradcam_pp = GradcamPlusPlus(model, model_modifier=model_modifier, clone=False)
        cam_pp = gradcam_pp(loss, images, penultimate_layer=-1)
        cam_pp = normalize(cam_pp)

        fig, ax = plt.subplots(1, num_images, figsize=(15, 5))
        if num_images == 1:
            ax = [ax]
        for i, title in enumerate(image_titles):
            heatmap = np.uint8(cm.jet(cam_pp[i])[..., :3] * 255)
            ax[i].set_title(title, fontsize=14)
            ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) / 255.0)  # Convert BGR to RGB for display
            ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[i].axis('off')
        tab1.pyplot(fig)

    if CNN_Visualisation == "Score-CAM":
        # Score-CAM
        if tab1.checkbox("Enable Score-CAM (requires GPU)"):
            tab1.write("### Score-CAM")
            scorecam = ScoreCAM(model, model_modifier=model_modifier, clone=False)
            cam_sc = scorecam(loss, images, penultimate_layer=-1)
            cam_sc = normalize(cam_sc)

            fig, ax = plt.subplots(1, num_images, figsize=(15, 5))
            if num_images == 1:
                ax = [ax]
            for i, title in enumerate(image_titles):
                heatmap = np.uint8(cm.jet(cam_sc[i])[..., :3] * 255)
                ax[i].set_title(title, fontsize=14)
                ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) / 255.0)  # Convert BGR to RGB for display
                ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
                ax[i].axis('off')
            tab1.pyplot(fig)
# ----------------------------------------------------------------------------------------------------------------------


