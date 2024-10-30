import cv2


picture = cv2.imread("IMG_0659.JPG")

greypicture = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
#turning image grey

haar_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#loading the required haarcascade classifier

#now applying face detection function on the grayscale image
#detectmultiscale is responsible for giving the coordinate for only the face
#1.1 is a scale factor for reducting the scale by 10%
#9 is the minimum neighbour 
face_rect = haar_cascade.detectMultiScale(greypicture,1.1,9)

print(face_rect)

for (x,y,w,h) in face_rect:
    cv2.rectangle(picture,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("faces",picture)
cv2.waitKey(0)



