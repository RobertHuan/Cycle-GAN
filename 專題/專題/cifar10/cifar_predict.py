import keras
import numpy as np
from matplotlib import pyplot
from scipy.misc import toimage
from keras.datasets import cifar10

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


(X_Train, y_Train), (X_Test, y_Test) = cifar10.load_data()

show_imags(X_Test[32:48])


X_Test40_norm = X_Test/255

model = keras.models.load_model('cifar_model.h5')
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
indices = np.argmax(model.predict(X_Test40_norm[32:48]), 1)
print([labels[x] for x in indices])

# ----- plot label and show
img_label = [labels[x] for x in indices]
img_label_1 = img_label[0:4]
img_label_2 = img_label[4:8]
img_label_3 = img_label[8:12]
img_label_4 = img_label[12:16]   # cut 4 line
pyplot.subplot2grid((5, 4), (4, 0), colspan=4)  # plot at grid(4, 0) and span 4 col
pyplot.axis('off')  # don't use axis
pyplot.text(0.5, 0.6, img_label_1, ha='center')     # word at (0.5, 0.6) and center to figure
pyplot.text(0.5, 0.3, img_label_2, ha='center')     # word at (0.5, 0.3) and center to figure
pyplot.text(0.5, 0.0, img_label_3, ha='center')     # word at (0.5, 0.0) and center to figure
pyplot.text(0.5, -0.3, img_label_4, ha='center')     # word at (0.5, -0.3) and center to figure
pyplot.show()
print(indices)
print(y_Test[32:48])
a = np.reshape(y_Test[32:48], newshape=1*16)
print(a)
print(len(a))
result = 0
for i in range(len(a)):
    if indices[i] == a[i]:
        result = result+1
print("Hit Rate of 16 Shown Images =", result/len(a))


# Compute the hit rate of entire test set
b = len(X_Test40_norm)
print(b)
new_indices = np.argmax(model.predict(X_Test40_norm[:b]), 1)
new_a = np.reshape(y_Test[:b], newshape=1*b)
result = 0
for i in range(b):
    if new_indices[i] == new_a[i]:
        result = result+1
print("Hit Rate of Total Test Set Images =", result/b)
