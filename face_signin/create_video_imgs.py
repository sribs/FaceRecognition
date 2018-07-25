import cv2
import os

def create_imgs_from_video(img_name):
    import cv2
    print(cv2.__version__)
    os.makedirs("video_data/"+img_name)
    vidcap = cv2.VideoCapture('output.avi')
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      cv2.imwrite("video_data/"+img_name+"/"+img_name+"%d.jpg" % count, image)     # save frame as JPEG file
      success,image = vidcap.read()
      #print ('Read a new frame: ', success)
      count += 1

def make_video():
    import numpy as np
    import cv2
    import time

    # The duration in seconds of the video captured
    capture_duration = 5

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    start_time = time.time()
    while( int(time.time() - start_time) < capture_duration ):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
            cv2.imshow('frame',frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()