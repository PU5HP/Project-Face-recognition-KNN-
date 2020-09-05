import cv2
import numpy as np

#init camera
cap = cv2.VideoCapture(0)

#face detection
face_cascade = cv2.CascadeClassifier('frontalface1.xml')
skip = 0
face_data =[]
dataset_path = '/home/pushp/Desktop/data-webcam/'
file_name =input('enter the person name:')
while True:
    ret , frame =cap.read()

    if ret==False:
        continue
       
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #storing the coordinates of the faces
    faces = face_cascade.detectMultiScale(frame,1.1,5)
    #print(faces)
    faces =sorted(faces,key=lambda f:f[2]*f[3])
     
    #pick the last face cause it is largest
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
       
       #extract (crop the required face ):region of the interest
        offset =10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        cv2.imshow('frame',frame)
        cv2.imshow('face section',face_section)
    #store every tenth face later on
    
    skip+=1
    if skip%10 == 0:
      face_data.append(face_section)
      print(len(face_data))

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#convert our face list to numpy as one d array
face_data = np.asarray(face_data)
face_data =face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
#save
np.save(dataset_path+file_name+'.npy',face_data)
print('data successfully saved')
cap.release()
cv2.destroyAllWindows()   
