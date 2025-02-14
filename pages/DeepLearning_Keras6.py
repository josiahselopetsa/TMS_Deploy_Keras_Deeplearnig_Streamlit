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
tab1,tab2,tab3,tab4,tab5 = st.tabs(["51. Blood Cell Object Detection - YOLOv5", "53. Image Segmentation - Keras, U-Net and SegNet", "55. Mask R-CNN Demo","57. Train a Mask R-CNN - Shapes","59. DeepFakes - first-order-model-demo"])  # ,"G","R","Hue"

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
# "51. Blood Cell Object Detection - YOLOv5"
YOLOv5 = tab1.selectbox("Blood Cell Object Detection - YOLOv5", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
YOLOv5_options = tab1.expander("Blood Cell Object Detection - YOLOv5")
YOLOv5_inputs = tab1.expander("Results : ")
placeholder1 = tab1.empty()

# ----------------------------------------------------------------------------------------------------------------------
# "53. Image Segmentation - Keras, U-Net and SegNet",
U_Net_SegNet  = tab2.selectbox("Image Segmentation - Keras, U-Net and SegNet", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
U_Net_SegNet_options = tab2.expander("Image Segmentation - Keras, U-Net and SegNet")
U_Net_SegNet_inputs = tab2.expander("Results : ")
placeholder2 = tab2.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "55. Mask R-CNN Demo"
R_CNN = tab3.selectbox("Mask R-CNN Demo", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
R_CNN_options = tab3.expander("Mask R-CNN Demo")
R_CNN_inputs = tab3.expander("Results : ")
placeholder3 = tab3.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "57. Train a Mask R-CNN - Shapes"
Train_R_CNN = tab4.selectbox("Train a Mask R-CNN - Shapes", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Train_R_CNN_options = tab4.expander("Train a Mask R-CNN - Shapes")
Train_R_CNN_inputs = tab4.expander("Results : ")
placeholder4 = tab4.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "59. DeepFakes - first-order-model-demo"
DeepFakes = tab5.selectbox("DeepFakes - first-order-model-demo", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
DeepFakes_options = tab5.expander("DeepFakes - first-order-model-demo")
DeepFakes_inputs = tab5.expander("Results : ")
placeholder5 = tab5.empty()
