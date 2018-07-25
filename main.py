from face_signin.create_video_imgs import make_video,create_imgs_from_video
from face_signin.prepare_training import prepare_training_data, detect_face
from face_signin.face_recognition import create_trainer,sign_in
'''
x = str(input("Image Name :"))

make_video()
create_imgs_from_video(x)'''
#data will be in two lists of same size
#one list will contain all the faces
#and the other list will contain respective labels for each face
name = str(input("User Name : "))
print("Preparing data...")
faces, labels = prepare_training_data("video_data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = create_trainer(faces,labels)

person = sign_in(face_recognizer)

if name.lower() in person.lower():
    print("Successful Authentication")

else:
    print("You are not "+name+", you are ", person)