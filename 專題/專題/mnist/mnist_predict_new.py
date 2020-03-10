import keras
import numpy as np
from matplotlib import pyplot
from scipy.misc import toimage
from keras.datasets import mnist

pyplot.figure(1)    # global


def show_imags(X):
    # pyplot.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            pyplot.subplot2grid((5, 4), (i, j))
            pyplot.imshow(toimage(X[k]))
            k = k + 1
    # pyplot.show()


(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()

show_imags(X_Test[20:36])

X_Test40 = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')
X_Test40_norm = X_Test40 / 255

model = keras.models.load_model('mnist_model.h5')
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
indices = np.argmax(model.predict(X_Test40_norm[20:36]), 1)
print([labels[x] for x in indices])

# ----- plot label and show
img_label = [labels[x] for x in indices]
img_label_1 = img_label[0:8]
img_label_2 = img_label[8:16]   # cut 2 line
pyplot.subplot2grid((5, 4), (4, 0), colspan=4)  # plot at grid(4, 0) and span 4 col
pyplot.axis('off')  # don't use axis
pyplot.text(0.5, 0.5, img_label_1, ha='center')     # word at (0.5, 0.5) and center to figure
pyplot.text(0.5, 0.2, img_label_2, ha='center')     # word at (0.5, 0.2) and center to figure
pyplot.show()
