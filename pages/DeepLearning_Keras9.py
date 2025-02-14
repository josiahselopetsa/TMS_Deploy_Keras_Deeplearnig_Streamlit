# Import PyTorch

import streamlit as st
import cv2

# Display the first image
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report

from keras.datasets import fashion_mnist
from keras.utils import to_categorical


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16 as Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from PIL import Image
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.scorecam import ScoreCAM
from matplotlib import cm
from keras.preprocessing.image import load_img

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
tab1,tab2,tab3 = st.tabs(["71. Low Light Image Enhancement using MIRNet", "73. Flask Rest API - Server","75. Flask Webapp"])  # ,"G","R","Hue"

# Check if a GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"GPU available: {torch.cuda.is_available()}")
st.sidebar.info(f"Using device: {device}")
# Images.image(image_rgb, caption="Processed Image", use_column_width=True, channels="RBG")

#______________________________________________________________________________________________
# Allow user to input the number of epochs
batch_size = st.sidebar.number_input("Number of batch_size:", min_value=1, max_value=1000, value=128, step=1)
epochs = st.sidebar.number_input("Number of epochs:", min_value=1, max_value=100, value=2, step=1)
momentum = st.sidebar.number_input("Number of momentum:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
lr = st.sidebar.number_input("Number of lr:", min_value=0.000, max_value=0.010, value=0.001, step=0.0001)

# ----------------------------------------------------------------------------------------------------------------------
# "71. Low Light Image Enhancement using MIRNet"
MIRNet = tab1.selectbox("Low Light Image Enhancement using MIRNet", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
MIRNet_options = tab1.expander("Low Light Image Enhancement using MIRNet")
MIRNet_inputs = tab1.expander("Results : ")
placeholder1 = tab1.empty()

# ----------------------------------------------------------------------------------------------------------------------
# "73. Flask Rest API - Server"
Flask_Rest  = tab2.selectbox("Flask Rest API - Server", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Flask_Rest_options = tab2.expander("Flask Rest API - Server")
Flask_Rest_inputs = tab2.expander("Results : ")
placeholder2 = tab2.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "75. Flask Webapp"
Flask_Webapp = tab3.selectbox("Flask Webapp", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Flask_Webapp_options = tab3.expander("Flask Webapp")
Flask_Webapp_inputs = tab3.expander("Results : ")
placeholder3 = tab3.empty()

