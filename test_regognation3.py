import cv2

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #img = cv2.imread("IMG_20191012_145410_3.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
    for (x,y,w,h) in faces:
        #set a thresh
        thresh = 100
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        #get threshold image
        ret,thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

        #find contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255,255,255), 1)
        img_gray_face = img_gray[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(img_gray_face, 1.1, 19)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x+ex, y+ey), (x+ex + ew, y+ey + eh), (255, 0, 0), 2)
    cv2.imshow('rez', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()