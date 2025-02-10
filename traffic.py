import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from keras.src.utils.module_utils import tensorflow
from progress.bar import IncrementalBar
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    if len(sys.argv) == 3:
        if os.path.exists(sys.argv[2]):
            print("Loading Model")
            model = tf.keras.models.load_model(sys.argv[2])
        else:
            print("Invalid Model File Path. Train New Model? (Y/N)\n")
            while (inp := input().upper()) not in ("Y","N"):
                print("Invalid Answer")
                inp = input().upper()
            if inp =="N":
                print("Exiting")
                return 0
            else:
                print("New Model Being Trained")
                model = get_model()
    else:

        print("No Model Provided. New One Being Trained")
        model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images=[]
    labels=[]
    bar=IncrementalBar('Reading Data',max=NUM_CATEGORIES) #create progress bar
    for label in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir,label)):
            img = cv2.imread(os.path.join(data_dir,label,image))
            img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT)) #resizes images to 30x30
            img=img/255 #normalizes rgb values
            images.append(img)
            labels.append(int(label))
        bar.next()
    bar.finish()
    return (images,labels)



def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([ #makes model
        tf.keras.layers.Conv2D(64, (3,3),activation="relu",input_shape=(IMG_HEIGHT,IMG_WIDTH,3)), #uses convolutional layer to find patterns in image
        tf.keras.layers.MaxPool2D(pool_size=(3,3)), #shrinks image by finding max pool in 3x3 area

        tf.keras.layers.Flatten(), #flattens the image

        tf.keras.layers.Dense(1024, activation="relu"), #big hidden layer
        tf.keras.layers.Dropout(0.5), #dropout to prevent overfitting


        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax") #output layer
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.save("model.keras") #saves model as model.keras in same directory
    return model


if __name__ == "__main__":
    main()
