import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras import utils
import h5py
import cv2
import numpy as np
import os

#Function extracts images taking input as individual file name and for various category folders
def get_optical_flow(fname, category):
    global count 
    cam = cv2.VideoCapture("./"+category + "/" +fname) #Looks for videos in the relavent folder in a list name after relavent categories
    print("./"+category + "/" +fname)
    ret, prev = cam.read() #Reading the video by utilization of OpenCV
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) #prevgray takes the previous frame I/P and converts into Grayscale
    prevgray = cv2.resize(prevgray, (224, 224)) #Resizes prevgray to a size with length : 224 breadth : 224
    videoflow = [] #List which will hold values of generated flow
    while ret:
        ret, img = cam.read() #Images being read when resized and converted to grayscale
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Similar pre-processing steps of image in the front of the target image
            gray = cv2.resize(gray, (224, 224)) #Resizing image in front to Length : 224 Breadth : 224
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, cv2.OPTFLOW_FARNEBACK_GAUSSIAN, 0.5, 5, 15, 3, 7, 1.5, 0)
            prevgray = gray #Making the frame in the front, acts as the previous image for next line of computation 
            videoflow.append(flow) #appending the flow vector to the matrix
        else:
            break
        array1 = np.concatenate(videoflow, axis=-1) #Stacking the flow vectors in a manner to generate a continuous action recognition sequence, in manner described in the paper
    #print(array1.shape)
    #print(len(array1), len(data[0]), len(data[0][0]), len(data[0][0][0]))
    #print(array1.T.shape)
    data = [] #Matrix consisting of data which would be continuously appended, for preprocessing
    for i in range(array1.shape[2]-597): #Dimnensionality reduction, to ensure that stacking of the layers takes place, to make construct motion form
        data.append(array1[:, :, i:i+600]) 
    print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]))
    return data
cv2.destroyAllWindows()

cat = ["carcrash", "fight", "gun"] #Categories
x_train = []
y_train = []
counter = -1
for category in cat:
    counter += 1
    for file in os.listdir("./"+category)[:1]:
        temp_data = get_optical_flow(file, category)
        for vidfile in temp_data:
            x_train.append(vidfile)
        #x_train.append(get_optical_flow(file, category))
        print(len(x_train), len(x_train[0]), len(x_train[0][0]), len(x_train[0][0][0]))
        y_train.append(np.array(counter))
#Input Definitions
learning_rate = 0.001
batch_size = 256
num_classes = 3
y_train = keras.utils.to_categorical(y_train, num_classes)
model = Sequential()
input_shape = (224, 224, 598) #Definition of Input dimensions for input to Convolutional Neural Network
#First Convolutional layer
model.add(Conv2D(96, (7, 7), padding="same", input_shape=input_shape))
"""
Filter = 96
kernel_size = 7,7
"""
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.normalization.BatchNormalization(axis=1))

#Second Convolutional layer
model.add(Conv2D(256, (5, 5), padding="same", strides=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#Third Convolutional layer
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))

#Fourth Convolutional layer
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))

#Fifth Convolutional layer
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Fully Connected Layer #1
model.add(Flatten()) #Operation to flatten the Convolution layers into a Fully Connected layer
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.9))


#Fully Connected Layer #2
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.9))

#Output Layer
model.add(Dense(3, activation='softmax'))



ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0) #Optimizer Conditionals
model.compile(loss="categorical_crossentropy", optimizer='Adadelta', metrics=['accuracy'])

x_train = np.array(x_train)
#y_train = y_train.reshape((-1, 1))
for i in range(1, 10):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=10, shuffle=True)
    model.save("iteration_"+str(i)+".h5") #Saving weights
