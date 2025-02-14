import streamlit as st

st.set_page_config(page_title="DeepLearningCV on Streamlit", page_icon="graph",layout="wide")

image = "./image/OpenCV_Chpt_3.png"

with st.container():
    st.subheader("Deep Learning part1 on Streamlit examples")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./images/Deeplearnig_images1/CoreScan.png"
    image2 = "./images/Deeplearnig_images1/DeepLearningImage.png"
    image3 = "./images/Deeplearnig_images1/DeepLearningImage2.png"
    image4 = "./images/Deeplearnig_images1/Training_Model.png"
    with col1:
        st.image(image1, caption="Training Our Model*")
    with col2:
        st.image(image2, caption="Visualizing some of our sample Data")
    with col3:
        st.image(image3, caption="Our Training Plots")
    with col4:
        st.image(image4, caption="GradCAM, GradCAM++ and Faster-ScoreCAM Visualisations")
    st.divider()

with st.container():
    st.subheader("Deep Learning part2 on Streamlit examples")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./images/Deeplearnig_images2/Deeplearnig_images2_2.png"
    image2 = "./images/Deeplearnig_images2/Deeplearnig_images2_3.png"
    image3 = "./images/Deeplearnig_images2/Deeplearnig_images2_4.png"
    image4 = "./images/Deeplearnig_images2/Deeplearnig_images2_5.png"
    with col1:
        st.image(image1, caption="Predictions of: VGG16, ResNet, Inceptionv3, MobileNetv2, SqueezeNet, WideResNet and MNASNet")
    with col2:
        st.image(image2, caption="Get Rank-5 Accuracy")
    with col3:
        st.image(image3, caption="Kaggle's Cats vs Dogs")
    with col4:
        st.image(image4, caption="Using Callbacks - Early Stopping & Checkpointing")
    st.divider()

with st.container():
    st.subheader("Deep Learning part3 on Streamlit examples")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./images/Deeplearnig_images3/Deeplearnig_images3_1.png"
    image2 = "./images/Deeplearnig_images3/Deeplearnig_images3_2.png"
    image3 = "./images/Deeplearnig_images3/Deeplearnig_images3_3.png"
    image4 = "./images/Deeplearnig_images3/Deeplearnig_images3_4.png"
    with col1:
        st.image(image1, caption="Visualize our Data Augmentations*")
    with col2:
        st.image(image2, caption="Transfer Learning with Ants vs Bees")
    with col3:
        st.image(image3, caption="Fast Style Transfer using TF-Hub")
    with col4:
        st.image(image4, caption="Load and preprocess our MNIST Dataset")
    st.divider()

with st.container():
    st.subheader("Deep Learning part4 on Streamlit examples")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./images/Deeplearnig_images4/Deeplearnig_images4_1.png"
    image2 = "./images/Deeplearnig_images4/Deeplearnig_images4_2.png"
    image3 = "./images/Deeplearnig_images4/Deeplearnig_images4_3.png"
    image4 = "./images/Deeplearnig_images4/Deeplearnig_images4_4.png"
    with col1:
        st.image(image1, caption="A Recap on GANs")
    with col2:
        st.image(image2, caption="Display a single image using the epoch number")
    with col3:
        st.image(image3, caption="Our VGGFaceModel")
    with col4:
        st.image(image4, caption="Verify Facial Similarity")
    st.divider()

with st.container():
    st.subheader("Deep Learning part5 on Streamlit examples")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./images/Deeplearnig_images5/Deeplearnig_images5_1.png"
    image2 = "./images/Deeplearnig_images5/Deeplearnig_images5_2.png"
    image3 = "./images/Deeplearnig_images5/Deeplearnig_images5_3.png"
    image4 = "./images/Deeplearnig_images5/Deeplearnig_images5_4.png"
    with col1:
        st.image(image1, caption="Demonstrate facial landmarks")
    with col2:
        st.image(image2, caption="Training MobileNetSSD Object Detection on a Mask Dataset")
    with col3:
        st.image(image3, caption="TinyYOLOv4 using a Pot Hole Dataset")
    with col4:
        st.image(image4, caption="Infer Custom Objects with Saved YOLOv4 Weights")
    st.divider()

with st.container():
    st.subheader("Deep Learning part6 on Streamlit examples")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./images/Deeplearnig_images6/Deeplearnig_images6_1.png"
    image2 = "./images/Deeplearnig_images6/Deeplearnig_images6_2.png"
    image3 = "./images/Deeplearnig_images6/Deeplearnig_images6_3.png"
    image4 = "./images/Deeplearnig_images6/Deeplearnig_images6_4.png"
    with col1:
        st.image(image1, caption="Semantic Segmentation - U-Net and SegNet")
    with col2:
        st.image(image2, caption="Mask CNN TensorFlow Demo by Matterport")
    with col3:
        st.image(image3, caption="Mask R-CNN - Train on Shapes Dataset")
    with col4:
        st.image(image4, caption="Display with Legend segmentation")
    st.divider()

# with st.container():
#     st.subheader("Deep Learning part7 on Streamlit examples")
#     st.divider()
#     col1,col2,col3,col4 = st.columns(4)
#     image1 = "./images/Deeplearnig_images7/"
#     image2 = "./images/Deeplearnig_images7/"
#     image3 = "./images/Deeplearnig_images7/"
#     image4 = "./images/Deeplearnig_images7/"
#     with col1:
#         st.image(image1, caption="Training Our Model*")
#     with col2:
#         st.image(image2, caption="Visualizing some of our sample Data")
#     with col3:
#         st.image(image3, caption="Our Training Plots")
#     with col4:
#         st.image(image4, caption="GradCAM, GradCAM++ and Faster-ScoreCAM Visualisations")
#     st.divider()
#
# with st.container():
#     st.subheader("Deep Learning part8 on Streamlit examples")
#     st.divider()
#     col1,col2,col3,col4 = st.columns(4)
#     image1 = "./images/Deeplearnig_images8/"
#     image2 = "./images/Deeplearnig_images8/"
#     image3 = "./images/Deeplearnig_images8/"
#     image4 = "./images/Deeplearnig_images8/"
#     with col1:
#         st.image(image1, caption="Training Our Model*")
#     with col2:
#         st.image(image2, caption="Visualizing some of our sample Data")
#     with col3:
#         st.image(image3, caption="Our Training Plots")
#     with col4:
#         st.image(image4, caption="GradCAM, GradCAM++ and Faster-ScoreCAM Visualisations")
#     st.divider()

