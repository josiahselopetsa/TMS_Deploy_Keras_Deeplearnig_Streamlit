import streamlit as st
import cv2
import PIL
# Display the first image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras.models import Sequential

from keras.datasets import mnist
from keras.applications.vgg16 import VGG16 as Model


def imshow(img, tab):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    tab.pyplot(plt)


#______________________________________________________________________________________________

st.set_page_config(page_title="DeepLearningCV on Streamlit", page_icon="👢",layout="wide")
st.sidebar.markdown("### DeepLearningCV Part 1 👢")
#______________________________________________________________________________________________
# Streamlit UI components
st.title("Deep Learning CV with Streamlit Part 1")
tab2,tab3,tab5 = st.tabs(["3. Keras Misclassifications and Model Performance Analysis","5. Keras -Fashion-MNIST Part 1 - No Regularisation","9. CNN Visualisation - Filter and Filter Activation Visualisation"])  # ,"G","R","Hue"


# Allow user to input the number of epochs

batch_size = st.sidebar.number_input("Number of batch_size:", min_value=1, max_value=1000, value=128, step=1)
epochs = st.sidebar.number_input("Number of epochs:", min_value=1, max_value=100, value=2, step=1)
momentum = st.sidebar.number_input("Number of momentum:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
lr = st.sidebar.number_input("Number of lr:", min_value=0.000, max_value=0.010, value=0.001, step=0.0001)



# ----------------------------------------------------------------------------------------------------------------------
Misclassifications = tab2.selectbox("Keras Misclassifications and Model Performance Analysis", ["Load our Keras Model and the MNIST Dataset", "Creating our Confusion Matrix"])
Misclassifications_options = tab2.expander("Keras Misclassifications and Model Performance Analysis")
Misclassifications_inputs = tab2.expander("Results : ")
placeholder2 = tab2.empty()

# Allow user to input the number of epochs

# batch_size = Misclassifications_inputs.number_input("Number of batch_size:", min_value=1, max_value=1000, value=128, step=1)
# epochs = Misclassifications_inputs.number_input("Number of epochs:", min_value=1, max_value=100, value=2, step=1)
# momentum = Misclassifications_inputs.number_input("Number of momentum:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
# lr = Misclassifications_inputs.number_input("Number of lr:", min_value=0.000, max_value=0.010, value=0.001, step=0.0001)

# Load the model
model = load_model('./pages/mnist_simple_cnn_10_Epochs.h5')

Misclassifications_options.info("Load the MNIST dataset")
Misclassifications_options.code("x_train, y_train), (x_test, y_test) = mnist.load_data()")
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

Misclassifications_options.info("Reshape the test data")
Misclassifications_options.code("x_test = x_test.reshape(10000, 28, 28, 1)")
# Reshape the test data
x_test = x_test.reshape(10000, 28, 28, 1)

Misclassifications_options.info("Get predictions")
Misclassifications_options.code("pred = np.argmax(model.predict(x_test), axis=-1)")
# Get predictions
pred = np.argmax(model.predict(x_test), axis=-1)

Misclassifications_options.info("Identify misclassified indices")
Misclassifications_options.code("result = np.absolute(y_test - pred)")
Misclassifications_options.code("misclassified_indices = np.nonzero(result > 0)")
# Identify misclassified indices
result = np.absolute(y_test - pred)
misclassified_indices = np.nonzero(result > 0)

if Misclassifications == "Load our Keras Model and the MNIST Dataset":

    # Define the function to display images with predictions
    def imshow(image, title=""):
        st.image(image, caption=title, use_column_width=True)


    def draw_test(name, pred, input_im):
        BLACK = [0, 0, 0]
        expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, input_im.shape[1], cv2.BORDER_CONSTANT, value=BLACK)
        expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(expanded_image, str(pred), (150, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
        return expanded_image


    # Streamlit app layout
    tab2.warning("MNIST Model Misclassifications")

    num_display = tab2.slider("Number of misclassifications to display", 1, len(misclassified_indices[0]), 10)
    # Define columns in Streamlit
    num_columns = 2
    cols = tab2.columns(num_columns)


    for i in range(num_display):
        input_im = x_test[misclassified_indices[0][i]]
        imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        input_im = input_im.reshape(1, 28, 28, 1)

        res = str(np.argmax(model.predict(input_im), axis=-1)[0])
        misclassified_image = draw_test("Misclassified Prediction", res, np.uint8(imageL))

        # Display the image in Streamlit
        Misclassifications_inputs.image(misclassified_image, caption=f"Misclassified as {res}")

    # Streamlit app layout
    tab2.subheader("MNIST Model Misclassifications")

    L = 5
    W = 5

    # Create a grid of subplots
    fig, axes = plt.subplots(L, W, figsize=(12, 12))
    axes = axes.ravel()

    for i in np.arange(0, L * W):
        input_im = x_test[misclassified_indices[0][i]]
        ind = misclassified_indices[0][i]
        predicted_class = str(np.argmax(model.predict(input_im.reshape(1, 28, 28, 1)), axis=-1)[0])
        axes[i].imshow(input_im.reshape(28, 28), cmap='gray_r')
        axes[i].set_title(f"Pred: {predicted_class}\n True: {y_test[ind]}")
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5)

    # Display the plot in Streamlit
    tab2.pyplot(fig)

if Misclassifications == "Creating our Confusion Matrix":
    # Get predictions
    y_pred = np.argmax(model.predict(x_test), axis=-1)

    # Compute confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)


    # Define the function to plot confusion matrix
    def plot_confusion_matrix(cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        return plt


    # Streamlit app layout
    tab2.subheader("MNIST Model Performance")

    # Display confusion matrix
    tab2.subheader("Confusion Matrix")
    fig_cm = plot_confusion_matrix(conf_mat,
                                   target_names=[str(i) for i in range(10)],
                                   title='Confusion Matrix',
                                   normalize=True)
    tab2.pyplot(fig_cm)

    # Display classification report
    tab2.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
    tab2.text(report)

    # Display per-class accuracy
    tab2.subheader("Per-Class Accuracy")
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    for i, ca in enumerate(class_accuracy):
        tab2.write(f'Accuracy for class {i}: {ca:.3f}%')

# ----------------------------------------------------------------------------------------------------------------------
Regularisation = tab3.selectbox("Keras -Fashion-MNIST Part 1 - No Regularisation", ["Loading, Inspecting and Visualising our data", "Data Preprocessing","Building Our Model","Training Our Model","Data Augmentation Example"])
Regularisation_options = tab3.expander("Keras -Fashion-MNIST Part 1 - No Regularisation")
Regularisation_inputs = tab3.expander("Results : ")
placeholder3 = tab3.empty()

# Load the Fashion-MNIST training and test dataset
Regularisation_options.code("Load the Fashion-MNIST training and test dataset : x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class names
Regularisation_options.code("classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']")
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# Store the shape of a single image
input_shape = (img_rows, img_cols, 1)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

if Regularisation == "Loading, Inspecting and Visualising our data":


    # Display the number of samples in x_train, x_test, y_train, y_test
    tab3.write(f"Initial shape or dimensions of x_train:, {x_train.shape}")

    tab3.write(f"Number of samples in our training data:, {len(x_train)}")
    tab3.write(f"Number of labels in our training data:, {len(y_train)}")
    tab3.write(f"Number of samples in our test data:, {len(x_test)}")
    tab3.write(f"Number of labels in our test data:, {len(y_test)}")

    tab3.write(f"\nDimensions of x_train:, {x_train[0].shape}")
    tab3.write(f"Labels in x_train:, {y_train.shape}")
    tab3.write(f"\nDimensions of x_test:, {x_test[0].shape}")
    tab3.write(f"Labels in y_test:, {y_test.shape}")

    # Set the number of images to display
    num_of_images = tab3.slider("Number of images to display", min_value=1, max_value=100, value=50)

    # Create a grid of images
    fig, axes = plt.subplots(5, 10, figsize=(16, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_of_images:
            # Verify the shape of y_train[i]
            Regularisation_inputs.write(f"y_train[{i}] shape: {y_train[i].shape}")

            # Convert one-hot encoded label back to an integer index
            class_index = np.argmax(y_train[i])

            # Fetch the corresponding class name
            class_name = classes[class_index]

            # class_name = classes[y_train[i]]
            # class_name = classes[y_train[i].item()]  # Ensure labels[0] is an integer
            ax.imshow(x_train[i], cmap='gray_r')
            ax.set_title(class_name)
            ax.axis('off')
        else:
            ax.axis('off')

    # Display the plot in Streamlit
    tab3.pyplot(fig)

if Regularisation == "Data Preprocessing":

    # Store the number of rows and columns
    Regularisation_options.code("Store the number of rows and columns: img_rows = x_train[0].shape[0]")
    Regularisation_options.code("Store the number of rows and columns: img_rows = x_train[0].shape[0]")



    # Store the shape of a single image
    Regularisation_options.code("Store the shape of a single image: input_shape = (img_rows, img_cols, 1)")
    input_shape = (img_rows, img_cols, 1)

    # Display the image shape
    tab3.write(f"Image shape: {input_shape}")




    # Count the number of columns in the one-hot encoded matrix (i.e., number of classes)
    Regularisation_options.code("Count the number of columns: num_classes = y_test.shape[1]")


    # Display the number of classes
    tab3.write(f"Number of Classes: {num_classes}")

if Regularisation == "Building Our Model":
    # Create the model
    model = Sequential()

    # Add layers to the model with Streamlit displaying each step
    tab3.write("### Adding Layers to the Model")

    # First Conv2D layer
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    tab3.write("Added Conv2D layer with 32 filters, kernel size 3x3, ReLU activation.")

    # Second Conv2D layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    tab3.write("Added Conv2D layer with 64 filters, kernel size 3x3, ReLU activation.")

    # MaxPooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    tab3.write("Added MaxPooling2D layer with pool size 2x2.")

    # Flatten layer
    model.add(Flatten())
    tab3.write("Added Flatten layer to flatten the input.")

    # Dense layer with 128 units
    model.add(Dense(128, activation='relu'))
    tab3.write("Added Dense layer with 128 units, ReLU activation.")

    # Output Dense layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))
    tab3.write("Added Dense output layer with softmax activation, corresponding to the number of classes.")

    # Summarize the model
    tab3.write("### Model Summary")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_str = "\n".join(model_summary)
    tab3.text(model_summary_str)

if Regularisation == "Training Our Model":
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])


    # Custom callback to update Streamlit during training
    class StreamlitProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs, ax, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.epochs = epochs
            self.progress_bar = st.progress(0)
            self.epoch_text = st.empty()
            self.ax = ax
            self.train_acc = []
            self.val_acc = []
            self.line_train, = ax.plot([], [], label='Train Accuracy')
            self.line_val, = ax.plot([], [], label='Val Accuracy')

        def on_epoch_end(self, epoch, logs=None):
            self.train_acc.append(logs['accuracy'])
            self.val_acc.append(logs['val_accuracy'])

            # Update progress bar and epoch text
            progress = (epoch + 1) / self.epochs
            self.progress_bar.progress(progress)
            self.epoch_text.text(f"Completed Epoch {epoch + 1}/{self.epochs}")

            # Update the plot
            self.line_train.set_data(range(1, len(self.train_acc) + 1), self.train_acc)
            self.line_val.set_data(range(1, len(self.val_acc) + 1), self.val_acc)
            self.ax.relim()
            self.ax.autoscale_view()
            st.pyplot(self.ax.figure)

    # # Set batch size and epochs using Streamlit inputs
    # batch_size = st.number_input("Batch size", min_value=16, max_value=128, value=32, step=16)
    # epochs = st.number_input("Number of epochs", min_value=5, max_value=50, value=15, step=5)

    # Train the model
    if tab3.button('Train Model'):
        fig, ax = plt.subplots()
        ax.set_title("Training and Validation Accuracy")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        with tab3.spinner('Training in progress...'):
            history = model.fit(x_train,
                                y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=0,  # Set verbose to 0 because we're updating via the callback
                                validation_data=(x_test, y_test),
                                callbacks=[StreamlitProgressCallback(epochs, ax)])

        # Evaluate the model
        score = model.evaluate(x_test, y_test, verbose=0)
        Regularisation_options.write(f'Test loss: {score[0]}')
        Regularisation_options.write(f'Test accuracy: {score[1]}')

if Regularisation == "Data Augmentation Example":
    st.sidebar.subheader("Data Augmentation setup")
    rotation_range = st.sidebar.slider("rotation_range", min_value=0, max_value=180, value=30)
    width_shift_range = st.sidebar.slider("width_shift_range", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    height_shift_range = st.sidebar.slider("height_shift_range", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    zoom_range = st.sidebar.slider("zoom_range", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    # Reshape data to [number of samples, width, height, color_depth]
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Change data type to float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Define Data Generator for Augmentation
    data_aug_datagen = ImageDataGenerator(rotation_range=rotation_range,
                                          width_shift_range=width_shift_range,
                                          height_shift_range=height_shift_range,
                                          shear_range=0.2,
                                          zoom_range=zoom_range,
                                          horizontal_flip=True,
                                          fill_mode='nearest')

    # Create an iterator for a single image
    aug_iter = data_aug_datagen.flow(x_train[0].reshape(1, 28, 28, 1), batch_size=1)


    def show_augmentations(augmentations=6):
        fig, axes = plt.subplots(1, augmentations, figsize=(15, 5))
        for i in range(augmentations):
            img = next(aug_iter)[0].astype('uint8')
            # Convert image from BGR (used by OpenCV) to RGB (used by Matplotlib)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].axis('off')
        tab3.pyplot(fig)


    # Streamlit app to control and display augmentations
    tab3.subheader("Image Augmentation Visualization")

    # Input to control the number of augmentations
    num_augmentations = tab3.slider("Number of Augmentations", min_value=1, max_value=10, value=6)

    # Button to generate and display the augmentations
    if tab3.button("Show Augmentations"):
        show_augmentations(num_augmentations)

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
Filter_Activation = tab5.selectbox("9. CNN Visualisation - Filter and Filter Activation Visualisation", ["Our Data Transform", "Defining Our Model","Training Our Model","Visualize our Trained Fillters"])
Filter_Activation_options = tab5.expander("9. CNN Visualisation - Filter and Filter Activation Visualisation")
Filter_Activation_inputs = tab5.expander("Results : ")
placeholder5 = tab5.empty()

tab5.title("MNIST Dataset Overview")

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

# Reshape the data to (samples, rows, cols, channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Store the shape of a single image
input_shape = (img_rows, img_cols, 1)

# Change the image type to float32 and normalize data
x_train = x_train.astype('float32') # originally uint8
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.001),
              metrics=['accuracy'])



if Filter_Activation == "Our Data Transform":

    # Display the shape and dimensions of the datasets
    tab5.subheader("Dataset Information")
    tab5.write(f"Initial shape or dimensions of x_train:, {x_train.shape}")
    tab5.write(f"Number of samples in our training data:, {len(x_train)}")
    tab5.write(f"Number of labels in our training data:, {len(y_train)}")
    tab5.write(f"Number of samples in our test data:, {len(x_test)}")
    tab5.write(f"Number of labels in our test data:, {len(y_test)}")

    # Print the image dimensions and number of labels
    tab5.write(f"Dimensions of a single image in x_train:, {x_train[0].shape}")
    tab5.write(f"Labels in x_train:, {y_train.shape}")
    tab5.write(f"Dimensions of a single image in x_test:, {x_test[0].shape}")
    tab5.write(f"Labels in y_test:, {y_test.shape}")

    # Display shapes and sizes
    tab5.subheader("Data Shapes and Sizes")
    tab5.write(f'x_train shape:, {x_train.shape}')
    tab5.write(f'{x_train.shape[0]}, train samples')
    tab5.write(f'{x_test.shape[0]}, test samples')

    # One-hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Display number of classes
    num_classes = y_test.shape[1]
    tab5.write(f"Number of Classes:, {num_classes}")

    # Calculate the number of pixels
    num_pixels = x_train.shape[1] * x_train.shape[2]
    tab5.write(f"Number of Pixels per Image:, {num_pixels}")

if Filter_Activation == "Defining Our Model":
    # Display model summary
    tab5.subheader("Model Summary")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    tab5.write(f"   {model_summary}")

if Filter_Activation == "Training Our Model":
    # Training parameters
    batch_size = batch_size
    epochs = epochs

    # Training loop
    if tab5.button('Start Training'):
        # Train the model
        tab5.subheader("Training the Model")
        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))

        # Evaluate the model
        tab5.subheader("Model Evaluation")
        score = model.evaluate(x_test, y_test, verbose=0)
        tab5.write('Test loss:', score[0])
        tab5.write('Test accuracy:', score[1])

        # Plot training & validation accuracy and loss
        tab5.subheader("Training History")

        # Plot accuracy
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Training Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy')
        ax.legend()
        tab5.pyplot(fig)

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss')
        ax.legend()
        tab5.pyplot(fig)


if Filter_Activation == "Visualize our Trained Fillters":
    # Get filter shapes and weights
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        filters, biases = layer.get_weights()
        tab5.write(f"{layer.name} filter shape: {filters.shape}")

    # Retrieve and normalize filters
    filters, biases = model.layers[0].get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # Plot filters
    tab5.subheader("Visualizing Filters")

    fig, ax = plt.subplots(4, 8, figsize=(12, 20))
    for i in range(32):
        f = filters[:, :, :, i]
        ax[i // 8, i % 8].imshow(np.squeeze(f, axis=2), cmap='gray')
        ax[i // 8, i % 8].axis('off')

    tab5.pyplot(fig)

    # Extract activations
    layer_outputs = [layer.output for layer in model.layers[:2]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    img_tensor = x_test[22].reshape(1, img_rows, img_cols, 1)
    activations = activation_model.predict(img_tensor)

    tab5.subheader("Activation Maps")

    first_layer_activation = activations[0]
    second_layer_activation = activations[1]

    tab5.write("First Layer Activation Shape:", first_layer_activation.shape)
    tab5.write("Second Layer Activation Shape:", second_layer_activation.shape)

    # Plot activation maps for the first layer
    fig, ax = plt.subplots(4, 8, figsize=(12, 8))
    for i in range(32):
        ax[i // 8, i % 8].imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        ax[i // 8, i % 8].axis('off')

    tab5.pyplot(fig)


    def display_activation(activations, col_size, row_size, act_index):
        activation = activations[act_index]
        fig, ax = plt.subplots(row_size, col_size, figsize=(col_size * 2.5, row_size * 1.5))
        for row in range(row_size):
            for col in range(col_size):
                if activation.shape[-1] > row * col_size + col:
                    ax[row][col].imshow(activation[0, :, :, row * col_size + col], cmap='viridis')
                    ax[row][col].axis('off')


    tab5.subheader("Activation Maps of Second Layer")
    display_activation(activations, 4, 8, 1)
    tab5.pyplot(plt.gcf())
