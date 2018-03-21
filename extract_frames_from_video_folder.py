import cv2
import os
#import numpy as np


def get_frame(category):
    for cat in category:
        count = 0
        for file in os.listdir("./"+cat):
            cam = cv2.VideoCapture("./" + cat + "/" + file)
            print "./" + cat + "/" + file
            #success = True
            success, frame = cam.read()
            #frame1 = cv2.resize(frame, (640, 480))

            while success:
                success, frame = cam.read()
                if success:
                    frame1 = cv2.resize(frame, (640, 480))
                    #print 'Read a new frame: ', success
                    cv2.imwrite("./images/"+"classifications/"+cat+"/frame%d.jpg" % count, frame1)  # save frame as JPEG file
                    count += 1
                else:
                    break



if __name__ == "__main__":
    get_frame(['carcrash', 'fight', 'gun'])


