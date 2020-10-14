import cv2
alg = "haarcascade_frontalface_default.xml"

haar = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0) #initialize camera

while True:
    _,img = cam.read() # read camera feed
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting color to gray
    faces = haar.detectMultiScale(grayImg,1.3,4) #obtain face coordinate
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2) # draw rectangle
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(10)
    if key == 27: # compare (27 for Esc button)
        break
cam.release()
cv2.destroyAllWindows()
