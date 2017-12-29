import cv2
import matplotlib.pyplot as plt
import numpy as np

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
 #just making a copy of image passed, so that passed image is not changed 
 img_copy = colored_img.copy()          

 #convert the test image to gray image as opencv face detector expects gray images
 gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          

 #let's detect multiscale (some images may be closer to camera than others) images
 faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5); 

 #go over list of faces and draw them as rectangles on original colored img
 for (x, y, w, h) in faces:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)  
      
 #detected face
 crop_img = img_copy[y+2:y+w, x+2:x+h]
 
 #find the forhead
 forhead = img_copy[int(y+h/15):int(y+h/10) , int(x+w/4):x+int(w/2)]
 
 return img_copy,forhead

def backproject(ROI,target):
    
    #roi is the object or region of object we need to find
    hsv = cv2.cvtColor(ROI,cv2.COLOR_BGR2HSV)
    
    #target is the image we search in
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    
    # calculating ROI histogram
    #channels [0,1] = hue and saturation
    #[0, 180, 0, 256] h_range and s_range
    roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    
    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    dst = cv2.GaussianBlur(dst,(5,5),0)
    
    #Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
    
    # threshold and binary "AND"
    
    ret,thresh = cv2.threshold(dst,5,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(target,thresh)
    
    #res = np.vstack((target,thresh,res))
    
    return target,thresh,res

def calculate_convexhull(img, erosion_size = 35 ):
    
    #dilate the image to make the isolated shapes overlap
    erosion_size = 35
    kernel = np.ones((erosion_size,erosion_size), np.uint8)
    img = cv2.dilate(res,kernel,iterations = 1)
    
    #applying convexhull
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255,0)
    _, contours, _ = cv2.findContours(thresh,2,1)
    
    return contours
 
#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
haar_face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#load test iamge
test = cv2.imread("test6.png")

#call our function to detect faces 
faces_detected_img,roi = detect_faces(haar_face_cascade, test)  

#calculate the histogram and apply backprojection
target,thresh,res = backproject(roi,test)

#plt.imshow(convertToRGB(res))

#calculate the onvexhull by first dilating the points and then findging the contours
contours = calculate_convexhull(res)

for cnt in contours:
    
    hull = cv2.convexHull(cnt)
    #cv2.drawContours(test,[cnt],0,(0,0,255),2)   #blue
    cv2.drawContours(test,[hull],0,(255,0,0),2)  #red
    #print(cv2.contourArea(hull))

#cv2.imshow('output',test)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 











