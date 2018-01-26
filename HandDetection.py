import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import PIL.Image
from sklearn.cross_validation import train_test_split
#from training_sample import train_model,predict_image

def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('data/Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data  = data[:,0:3]

    return data, labels

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv

def TrainTree(data, labels):

    data= BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(trainData, trainLabels)

    return clf

def ApplyToImage(img, clf):

    
    img = cv2.GaussianBlur(img,(3,3),0)

    data= np.reshape(img,(img.shape[0]*img.shape[1],3))

    data = BGR2HSV(data)

    predictedLabels = clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
    
    imgLabels = ((-(imgLabels-1)+1)*255)
    skin_img = imgLabels.squeeze()
    
    arr = PIL.Image.fromarray(np.uint8(skin_img))
    arr = np.asarray(arr)
    
    return arr
  
def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
 
 #just making a copy of image passed, so that passed image is not changed 
 img_copy = colored_img.copy()          

 #convert the test image to gray image as opencv face detector expects gray images
 gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          

 #let's detect multiscale (some images may be closer to camera than others) images
 faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5); 
 
 if len(faces) != 0:
     return faces
 else:
     empty = []
     return (empty)
 
def backproject(ROI,target, ksize = 1, threshold = 0):
    
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
    #dst = cv2.medianBlur(dst, ksize)
    #dst = cv2.GaussianBlur(dst,(5,5),0)
    
    #Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    cv2.filter2D(dst,-1,disc,dst)
    
    # threshold and binary "AND" 
    ret,thresh = cv2.threshold(dst,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.merge((thresh,thresh,thresh))
    
    dst = cv2.GaussianBlur(dst,(5,5),0)
    
    res = cv2.bitwise_and(target,thresh)
    
    return target,thresh,res

def calculate_contours(img, erosion_size = 0 ):
    
    #dilate the image to make the isolated shapes overlap
#    kernel = np.ones((erosion_size,erosion_size), np.uint8)
#    img = cv2.dilate(img,kernel,iterations = 1)

    #applying convexhull
#    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    ret, thresh = cv2.threshold(img_gray, 127, 255,0)
    ret, thresh = cv2.threshold(img, 127, 255,0)
    _, contours, _ = cv2.findContours(thresh,2,1)
    
    return contours,img

def hard_code_skin_detection(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);
    frame = cv2.GaussianBlur(frame,(7,7), 1, 1);

    for r in range(frame.shape[0]):
        for c in range(frame.shape[1]):
    #        #0<H<0.25  -   0.15<S<0.9    -    0.2<V<0.95
            if( (frame[r,c,0]>5) and (frame[r,c,0] < 17) and (frame[r,c,1]>38) and (frame[r,c,1]<250) and (frame[r,c,2]>51) and (frame[r,c,2]<242) ):
                pass
            else:
                for k in range(0,3):
                    frame[r,c,k] = 0
             
    frame      = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,frame_gray = cv2.threshold(frame_gray, 60, 255, cv2.THRESH_BINARY)
    
    frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_ERODE, (3,3,1))
    frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN,  (7,7,1))
    frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_CLOSE, (9,9,1))
    
    frame_gray = cv2.medianBlur(frame_gray, 3)
    
    return frame_gray

#load cascade classifier training file
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#train the skin classifier
data, labels= ReadData()
clf = TrainTree(data, labels)

#train the gesture classifier
#mlp = train_model('Gesture_Images/image_dataSet/')

#interface
def nothing(x):
    pass

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('res', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)
cv2.resizeWindow('res'  , 600, 600)

cv2.createTrackbar('min_area'     ,'image',1500,50000   ,nothing)
cv2.createTrackbar('max_area'     ,'image',150000,150000,nothing)

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 1080)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH , 1920)

while(cap.isOpened() and 1):
    
    ret, img = cap.read()  
    org      = img.copy()
#    org2     = img.copy()
    
    face_only = detect_faces(haar_face_cascade,img)
    if(len(face_only) != 0):
        #set face to black
        for (x, y, w, h) in face_only:
          cv2.rectangle(org, (x-20, y-20), (x+w, y+h+50), (0, 0, 0),-1)      
    
    skin_img = ApplyToImage(org, clf)

    min_hand_threshold = cv2.getTrackbarPos('min_area','image')
    max_hand_threshold = cv2.getTrackbarPos('max_area','image')
    

    contours,dilated_image = calculate_contours(skin_img)
       
    for cnt in contours:
        hull = cv2.convexHull(cnt)       
        area = cv2.contourArea(hull)
        if( area > min_hand_threshold and area < max_hand_threshold ):
            cv2.drawContours(img,[cnt] ,0,(0,0,255),2)   #red
            cv2.drawContours(img,[hull],0,(255,0,0),2)   #blue
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
#            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
#            p1 = (cX-100, cY-100)
#            p2 = (cX+100, cY+100)
            #cv2.rectangle(org, p1, p2, (255,255, 0), 2)
            x, y, width, height = cv2.boundingRect(cnt)
            roi = skin_img[y:y+height, x:x+width]
            cv2.rectangle(img, (x,y), (x+width,y+height), (255,255, 0), 2)
            #cv2.imshow('cnt' ,org2[y:y+height, x:x+width])
            
            #prepare the image for the gesture classifier
            #gest = cv2.resize(org2, (200,200), cv2.INTER_LINEAR)

            
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    cv2.putText(img,"W: %d" %w,(20,50) , cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0)) 
    cv2.putText(img,"H: %d" %h,(20,100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0)) 
                    
    cv2.imshow('image' ,img)
    cv2.imshow('res'   ,skin_img)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    
cap.release()    
cv2.destroyAllWindows()






