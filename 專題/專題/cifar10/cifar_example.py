from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import os
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras

np.random.seed(10)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_trainOneHot = np_utils.to_categorical(y_train)

# -------------------------------------------------
model = Sequential()
# Create CN layer 1
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(BatchNormalization())
# Create Max-Pool 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# Create CN layer 2
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
# Create Max-Pool 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Create CN layer 3
model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
# Create Max-Pool 3
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add Dropout layer
model.add(Dropout(0.25))
model.add(Flatten())
# Fully connected
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# print(K.get_value(model.optimizer.lr))
K.set_value(model.optimizer.lr, 0.002)
print(K.get_value(model.optimizer.lr))
# 每4個epochs就將learning rate降低0.9

def scheduler(epoch):
    if epoch % 4 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.8)
        print("lr changed to {}".format(lr * 0.8))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)

callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1, mode='min'),
    keras.callbacks.ModelCheckpoint(filepath='save_best_1.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='max', period=1),
    reduce_lr
]

train_history = model.fit(x=x_train, y=y_trainOneHot, validation_split=0.2, epochs=10, batch_size=1000, verbose=1, callbacks= callbacks_list)
model.save('cifar_model_1.h5')

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

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')
