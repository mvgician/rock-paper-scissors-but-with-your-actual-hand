import os
import numpy
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import glob
import sys
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

# comment this to enable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Resize all training data to image_size x image_size
image_size = 50

# load training data for the "rock" gesture
rock_files = glob.glob(sys.argv[1])
rock = [cv2.imread(img, 0) for img in rock_files]  # 0 as a parameter = grayscale
y_rock = numpy.empty([len(rock)])
for i in range(0, len(rock)):
    rock[i] = cv2.resize(rock[i],(image_size, image_size))
    y_rock[i] = 0   # Output is arbitrarily set as the "0" category and so on

# load training data for the ""paper" gesture
paper_files = glob.glob(sys.argv[2])
paper = [cv2.imread(img, 0) for img in paper_files]  # 0 as a parameter = grayscale
y_paper = numpy.empty([len(paper)])
for i in range(0, len(paper)):
    paper[i] = cv2.resize(paper[i],(image_size, image_size))
    y_paper[i] = 1

# load training data for the "scissors" gesture
scissors_files = glob.glob(sys.argv[3])
scissors = [cv2.imread(img, 0) for img in scissors_files]  # 0 as a parameter = grayscale
y_scissors = numpy.empty([len(scissors)])
for i in range(0, len(scissors)):
    scissors[i] = cv2.resize(scissors[i],(image_size,image_size))
    y_scissors[i] = 2

x_set = []
x_set.extend(rock)
x_set.extend(paper)
x_set.extend(scissors)

y_set = []
y_set.extend(y_rock)
y_set.extend(y_paper)
y_set.extend(y_scissors)

# Split data into training and test in a 90%-10% ratio (debatable and other values should be tried such as 70-30)
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.1)
x_train = numpy.asarray(x_train)
x_test = numpy.asarray(x_test)
y_train = numpy.asarray(y_train)
y_test = numpy.asarray(y_test)

# AFAIK train_test_split does scramble the data but doing it twice won't hurt either
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

# flatten the image_size * image_size data to a one-dimension vector
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], 1, image_size, image_size).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, image_size, image_size).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# transform to one-hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

# model definition
# It works fine as it is (with my training data - 1000 images per gesture - it topped 99.9% test accuracy on the second epoch)
# but it certainly is an overkill
def baseline_model():
    model = Sequential()
    # convolution layer is most likely neccesary, but consider reducing 32 nodes to 16 or kernel to 3x3
    # according to references I forgot (feel free to ask) ELU worked great on the CIFAR100 dataset, so consider trying that too
    # from my experience parametric RELU works well enough too
    model.add(Conv2D(32, (5, 5), input_shape=(1, image_size, image_size), data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # consider reducing 128 nodes to 64 and changing the activation function to p_relu / elu
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # adadelta works too from my experience
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# oh yeah it's all coming together
model = baseline_model()

# awaken your frankenstein's monster
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=20, verbose=1)

# final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# save the model into a .json for further usage
model_json = model.to_json();
with open("rps_model.json", "w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("rps_model_weights.h5")
print("Done! model saved into current folder with name 'rps_model.json' and 'rps_model_weights.h5'")