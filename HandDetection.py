import cv2

hand_cascade = cv2.CascadeClassifier('aGest.xml')
palm_cascade = cv2.CascadeClassifier('palm.xml')
frontal_cascade = cv2.CascadeClassifier('closed_frontal_palm.xml')

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
                
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in hands:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()        