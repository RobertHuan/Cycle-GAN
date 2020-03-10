#!/usr/bin/env python3
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt

np.random.seed(10)

# Read MNIST data
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()
# Translation of data
X_Train40 = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')
X_Test40 = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')
# Standardize feature data
X_Train40_norm = X_Train40 / 255
X_Test40_norm = X_Test40 / 255
# Label Onehot-encoding
y_TrainOneHot = np_utils.to_categorical(y_Train)
#########################################################################################
model = Sequential()
# Create CN layer 1
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
# Create Max-Pool 1
model.add(MaxPooling2D(pool_size=(2, 2)))
# Create CN layer 2
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
# Create Max-Pool 2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add Dropout layer
model.add(Dropout(0.25))

model.add(Flatten())
# Fully connected
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
# model.add(Dense(10, activation='sigmoid'))
model.summary()
print("")

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
train_history = model.fit(x=X_Train40_norm, y=y_TrainOneHot, validation_split=0.2, epochs=8, batch_size=600, verbose=1)
model.save('mnist_model.h5')

##################################

'''
def isDisplayAvl():
    return 'DISPLAY' in os.environ.keys()

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

def plot_images_labels_predict(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "l=" + str(labels[idx])
        if len(prediction) > 0:
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))
        else:
            title = "l={}".format(str(labels[idx]))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()

'''

def show_train_history(train_history, train, validation):
    # plot train set accuarcy / loss function value ( determined by what parameter 'train' you pass )
    # The type of train_history.history is dictionary (a special data type in Python)
    plt.plot(train_history.history[train])
    # plot validation set accuarcy / loss function value
    plt.plot(train_history.history[validation])
    # set the title of figure you will draw
    plt.title('Train History')
    # set the title of y-axis
    plt.ylabel(train)
    # set the title of x-axis
    plt.xlabel('Epoch')
    # Places a legend on the place you set by loc
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# from utils import *
# if isDisplayAvl():
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')