import cv2
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
from hsv_segmenter import HSVSegmenter
from keras.models import load_model
import keras
import tensorflow as tf
#import matplotlib.pyplot as plt

print("numpy:", np.version.version)
print('keras:', keras.__version__)
print('tf:', tf.__version__)
print('cv2:', cv2.__version__)

# %% SKIN DETECTOR BASED ON DECISION TREE
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

    data= BGR2HSV(data)

    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
    
    imgLabels = ((-(imgLabels-1)+1)*255)
    skin_img = imgLabels.squeeze()
    
    #arr = PIL.Image.fromarray(np.uint8(skin_img))
    #arr = np.asarray(arr)
    arr = np.asarray(skin_img)
    arr = arr.astype('uint8')
    
    return arr

# %% HAAR CASCADE FACE DETECTION
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
     #go over list of faces and draw them as rectangles on original colored img
     for (x, y, w, h) in faces:
          cv2.rectangle(img_copy, (x-20, y-20), (x+w+20, y+h+20), (0, 0, 0), -1)  
#          
#     #detected face  
#     crop_img = img_copy[y+2:y+w, x+2:x+h]
#     
#     #find the forhead
#     forhead = img_copy[int(y+h/20):int(y+h/5) , x+5:x+int(w/1.5)]

     return img_copy
 else:
     return colored_img
 
# %% Calculate contour
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

#load cascade classifier
#haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

h = HSVSegmenter()
model = load_model('./Keras_models/FINAL_MODEL.h5')

# %% Hand detection and Recognition loop
def nothing(x):
    pass

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',700,700)
cv2.resizeWindow('HSV',700,700)
cv2.createTrackbar('min_area','image',300 ,7000 ,nothing)


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#for video recording
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out_frame = cv2.VideoWriter('frame.avi',fourcc, 10.0, (640,480))
#out_mask  = cv2.VideoWriter('mask.avi',fourcc, 10.0, (640,480), isColor = False)

Tracking = False
nothing  = False

while(cap.isOpened() and 1):
    
    ret, img = cap.read()
    copy     = img.copy()

    skin_img = h.get_mask(copy)
    min_hand_threshold = cv2.getTrackbarPos('min_area','image')

    if not Tracking:
        cv2.rectangle(img, (50,180), (250,370), (255,0, 0), 2)
        roi = copy[180:370,50:250]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi,(70,70))
        roi = np.array(roi)
        roi = roi.astype('float32')
        roi /= 255
        roi= np.expand_dims(roi, axis=3) 
        roi= np.expand_dims(roi, axis=0)
        y_prob = model.predict(roi) 
        y_classes = y_prob.argmax(axis=-1)
        prob = max(y_prob)[0]*100
        cv2.putText(img,"%f,%s" %(prob,y_classes) ,(50,370), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), thickness= 2)
     
    else:
        contours,dilated_image = calculate_contours(skin_img)
        for cnt in contours:
            hull = cv2.convexHull(cnt)       
            area = cv2.contourArea(hull)
            if( area > min_hand_threshold and area < 900000 ):
#                cv2.drawContours(copy,[cnt] ,0,(0,0,255),2)   #red BGR
                cv2.drawContours(copy,[hull],0,(255,0,0),2)   #blue BGR
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                x, y, width, height = cv2.boundingRect(cnt)  
                                
                if not nothing:
                    while((height/width) > 1.1):
                        height = int(height*0.97)
                    
                    while((width/height)>1.1):
                        width=int(width*0.97)
                    
                roi = img[y:y+height, x:x+width]
                #cv2.rectangle(img, (x,y), (x+width,y+height), (255,255, 0), 2)
                           
                #predict the image 
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi,(70,70))
                roi = np.array(roi)
                roi = roi.astype('float32')
                roi /= 255
                roi= np.expand_dims(roi, axis=3) 
                roi= np.expand_dims(roi, axis=0)
                print(roi.shape)
                y_prob = model.predict(roi) 
                y_classes = y_prob.argmax(axis=-1)

                if not (width < 23 or height< 33):
                    cv2.rectangle(img, (x,y), (x+width,y+height), (255,255, 0), 2)
                    prob = max(y_prob)[0]
                    cv2.putText(img,"%f,%s" %(prob,y_classes) ,(cX,cY), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), thickness= 2)
            
#    out_frame.write(img)
#    out_mask.write(skin_img)
                    
    cv2.imshow('HSV',skin_img)           
    cv2.imshow('image' ,img)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    if k == ord('t'):
        Tracking = not Tracking       
    if k == ord('h'):
        nothing = not nothing

#out_frame.release()
#out_mask.release()
        
cap.release()    
cv2.destroyAllWindows()







