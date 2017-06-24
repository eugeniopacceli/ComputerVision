import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #utliza o dataset cifar10 do keras

nb_classes = 10
nb_epoch=200
batch_size=32

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

in_shape=x_train.shape

y_train = np_utils.to_categorical(y_train, nb_classes)

model = Sequential()
# For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv
# By default the stride/subsample is 1 and there is no zero-padding.
# If you want zero-padding add a ZeroPadding layer or, if stride is 1 use border_mode="same"
model.add(Conv2D(12, (5, 5), activation = 'relu', input_shape=(32,32,3), init='he_normal'))

# For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(25, (5, 5), activation = 'relu', init='he_normal'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
model.add(Flatten())
model.add(Dense(180, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation = 'softmax', init='he_normal')) #Last layer with one output per class

# The function to optimize is the cross entropy between the true label and the output (softmax) of the model
# We will use adadelta to do the gradient descent see http://cs231n.github.io/neural-networks-3/#ada
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])

# Make the model learn
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)

# Predict the label for X_test
#yPred = model.predict_classes(X_test)
