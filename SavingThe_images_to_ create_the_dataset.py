  
#Saving the images to create the dataset

import cv2
import os  #to handle the direcoties , checking path, folder availability

alg = 'haarcascade_frontalface_default.xml'
haar = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0) #initialize camera

dataset = 'dataset'
name = 'papa'

path = os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)

(width,height) = (130,100)
count = 1    
while count < 100:
    print(count)
    (_,img) = cam.read() # read camera feed
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting color to gray
    faces = haar.detectMultiScale(grayImg,1.3,4) #obtain face coordinate
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2) # draw rectangle
        onlyFace = grayImg[y:y + h,x:x + w] # will crop only face
        resizeImg = cv2.resize(onlyFace,(width,height))
        cv2.imwrite("%s/%s.jpg" %(path,count),resizeImg)
    count+=1 # (1.jpg, 2.jpg etc)
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(10)
    if key == 27: # compare (27 for Esc button to come out of the loop)
        break
print("Face captured successfully")    
cam.release()
cv2.destroyAllWindows()
