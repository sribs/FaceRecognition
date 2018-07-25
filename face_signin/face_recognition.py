import cv2
import os
import numpy as np
from face_signin.prepare_training import detect_face

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def create_trainer(faces,labels):
    #create our LBPH face recognizer 
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))

    return face_recognizer

def predict(test_img,dirs,face_recognizer):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label= face_recognizer.predict(face)
    #print(label)
    #get name of respective label returned by face recognizer
    label_text = dirs[label[0]]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img,label_text

def predict_images(img_str,face_recognizer):
    #predict images
    print("Predicting images...")
    dirs = sorted(os.listdir("video_data"))
    #load test images
    #test_img1 = cv2.imread("video_train/frame84.jpg")
    test_img2 = cv2.imread(img_str)

    #perform a prediction
    #predicted_img1 = predict(test_img1)
    predicted_img2,label_text = predict(test_img2,dirs,face_recognizer)
    print("Prediction complete")

    #display both images
    #cv2.imshow(dirs[0], predicted_img1)
    cv2.imshow(label_text, predicted_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return label_text

def sign_in(face_recognizer):

    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
    ret,frame = cap.read() # return a single frame in variable `frame`
    dirs = sorted(os.listdir("video_data"))
    while(True):
        cv2.imshow('img1',frame) #display the captured image
        cv2.imwrite('video_train/c1.png',frame)
        cv2.destroyAllWindows()
        break

    cap.release()

    person = predict_images("video_train/c1.png",face_recognizer)

    return person