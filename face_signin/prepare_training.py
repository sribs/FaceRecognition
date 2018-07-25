import cv2
import numpy as np 
import os

def prepare_training_data(data_folder_path):
 
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = sorted(os.listdir(data_folder_path))
    #print(dirs)
    faces = []
    labels = []
    
    for label,count in zip(dirs,range(len(dirs))):
        subject_dir_path = data_folder_path+"/"+label
        for image_name in os.listdir(subject_dir_path):

            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)

            #display an image window to show the image 
            #print("Training label :",label)
            cv2.waitKey(100)

            #detect face
            face, rect = detect_face(image)

            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(count)
        print("Data Prepared for Training")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    x, y, w, h = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

