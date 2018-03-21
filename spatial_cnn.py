import keras
from keras.models import Sequential
from keras.layer import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import os
import cv2
import h5py

def get_frame(fname, category):

 arr1=[]#initializing an empty array
 cap=cv2.VideoCapture("./"+category + "/"+fname)#creates a videocapture object using the the given path for the video saved on the computer
 success,frame=cam.read()#the function returns 1 if the frame is read properly, else 0-the value of which is stored in success.frame returns the frame that is read
 frame1=cv2.resize(frame, (140,140))#resizing the frames to a matrix of (140x140)
 success=True#set success to true
 while success:
  success,frame=cap.read()#the loop runs while each frame is read properly
  frame1=cv2.resize(frame, (140,140))#reszing the frames to a matrix of 140x140
  arr1.append(frame1)#appending the frame matrix to the array
 return arr1


cat=["carcrash", "fight", "gun"]#list of categories for classification
X=[]
y_train=[]

"""
The below loop converts converts the stored videos in sepearte categories of classification into several frames using opencv
"""
for cat in category:#loops the categories in the category list
 for file in os.listdir("./"+category)[:1]:#loops through each video in the given category file
  temp=[]
  temp=get_frame(file,cat)#converts given video to matrix of pixel value of each individual frame using function get_file
  X.append(temp)#X now is the input for the CNN

"""
The below loop intializes the y values for all the values in the X matrix as per their their categories
"""
for category in cat:
    counter += 1
    for file in os.listdir("./"+category)[:1]:
        y_train.append(np.array(counter))
     
y_train = keras.utils.to_categorical(y_train, num_classes)
#building the base vgg model such that features are extracted using the vggmodel. till the end of the convulution layers
base_model=VGG16(weights='imagenet', include_top=False)
x=preprocess_input(X)#using keras's built in function


features=base_model.predict(x)
np.save(open('features.npy', 'w'),features)#saving the model obtained after passing the input through the pretrained VGG network 

#building the bottleneck
x_train=np.load(open('features.npy'))#loading the trained model


top_model = Sequential()
top_model.add(Flatten())
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.9))#full layer with dropout- full6
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.8))
top_model.add(Dense(20,activation='softmax'))#full layer with dropout-full7


#compiling the model with the adadelta optimizer
top_model.compile(optimizer='Adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#fitting data to the model as per the parameters of the paper
top_model.fit( x_train, y_train, batch_size=256, epochs=50,validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
top_model.save('spatial_cnn.h5')#saving the weights obtained from the model
