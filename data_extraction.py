import cv2
import os
import cv2
import numpy as np
import pickle
#This code allows for significant sub sampling of the dataset for multiplicative training data.


path_ann = "" # Write the path of the folder which contains the files with text file of start and end frames
path_vid = "" # path of the videos to be extracted
path_pickle = "" #path of the folder where pickles are stored

ann_data = os.listdir(path_ann) #list of files in path_ann

vid_files = [i for i in os.listdir(path_vid) if os.path.isfile(os.path.join(path_vid,i))] #list of videos in path_vid

p = ""
for data in ann_data:
    p = path_ann + data
    
    file = open(p, "r") #opens annotation text file
    vid_name = data.rpartition('_')[0] #extracts whatever is written in front of the underscore from the name of textfile

    #finds the path of corresponding video
    for i in vid_files:
        if vid_name in i:
            vid = i
            break
    path = path_vid + "/" + vid #use backslash if required
    #arr = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and vid in i]
    cap = cv2.VideoCapture(path)
    
    for line in file:
        words = line.split()
        t1 = int(words[0]) #first frame
        t2 = int(words[1]) #last frame
        m = t1
        n = m + 24
        while n <= t2:
            pic = []
            for i in range(m,n):
                #reads ith frame, I am not sure if this will work,
                #if it dosent then we will have to calculate it with 
                #video parameters which are diffrenr for every video :(
                cap.set(1,i) 
                ret, frm = cap.read() 
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY) #Converts frame from Color to Grayscale
                gray = cv2.resize(gray, (224, 224)) #Resizing the frame
                pic.append(gray) #Appending to pickle vector
                A = np.array(pic)
                #saves in path_pickle
                #Creates Pickle for saving data
                 pickle.dump(A, open(path_pickle + "/" + str(counter) + ".p", "wb"))    
            m = m + 24
            n = m + 168
        cap.release()

