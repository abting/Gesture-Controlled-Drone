import cv2
import numpy as np
import time
from sklearn import svm, metrics
import os
import matplotlib.pyplot as plt

img_path = 'Gesture_Images/'   #relative path to where the pictures should be saved
number_of_gestures = 2         #number of gestures to be differentiated between
image_delay = 2.5                #(seconds)time to wait between taking images
number_of_pictures = 30         #number of pictures to take PER gesture
user_ready = 3                 #(seconds)amount of time for the user to get ready

begin_train = 1                 
begin_predict = 0          
    
img_set = []                    #opencv images are stored in this array
img_target = []                 #[n_sample x 1] array of target class(gestures)
img_data = []                   #flattened and converted numoy array of images

test_set = []
test_data = []
test_target = []

#initialize camera
cap = cv2.VideoCapture(0)

#wait 3 seconds for the user to get ready
for i in range(user_ready,0,-1):
    print("prepare to take pictures in ",i)
    time.sleep(1)
    i+=1


#cycle through the gestures and take pictures
for j in range(number_of_gestures):
    
    #create the folder if they do not exist
    path = img_path + str(j)
    if not os.path.exists(path):
        os.makedirs(path)
    
    i = 0
    now = int(round(time.time() * 1000))
    capture = True
    while(cap.isOpened() & capture ):
                
        ret, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)       
        
        current = int(round(time.time() * 1000))
        delay = current-now
        cv2.putText(img,"gesture: %d" %j,(50,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
        cv2.putText(img,"image: %d" %(i+1),(50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
        cv2.putText(img,"%d" %delay,(50,130), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
        cv2.imshow('Train',img)
        #print(delay)
        if(delay > image_delay*1000):
            print("taking picture %d for gesture %d" %(i,j))
            now = int(round(time.time() * 1000))
            cv2.imwrite(path + '/%d.jpg' %i,gray)
            img_set.append(gray)
            img_target.append(j)
            i+=1
        if cv2.waitKey(1) & 0xff ==ord('q'):
            break
        if(i==number_of_pictures):
           capture = False 

cap.release()
cv2.destroyAllWindows()
#           
##array size & type sanity check 
#print("img_set shape = ",np.array(img_set).shape)
#print("img_target shape = ",np.array(img_target).shape)
##print(img_target)
##print(type(np.array(img_set)))
#
##now that all the pictures are taken lets train the model
##first flatten the images
#n_sample = number_of_gestures*number_of_pictures
#img_data = np.array(img_set).reshape((n_sample,-1))
#img_target = np.array(img_target).reshape((n_sample,))
#
#print("flattened img_data shape = ",img_data.shape)
#print("flattened img_target shape = ",img_target.shape)
#
##Create a classifier: a support vector classifier
#classifier = svm.SVC(gamma=0.001)
##train 
#classifier.fit(img_data, img_target)
#
#print("classification report : \n", classifier)
##    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#
##if(begin_predict == 1):
##    cap = cv2.VideoCapture(0)
##    while(True ):
##        
##        ret, img = cap.read()
##        cv2.imshow('pic',img)
##        test = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##        #if space is pressed take a picture and predict the image
##        #space key
##        if cv2.waitKey(1) & 0xff ==ord('q'):
##            break
##        
##        if cv2.waitKey(1) == 32:
##            print("predicting the gesture...")
##            #plt.imshow(test,cmap=plt.cm.gray_r, interpolation='nearest')
##            test = test.reshape(1,-1)
##            predicted = classifier.predict(test)
##            print("predicted array shape = ", predicted.shape)
##            print(predicted)
##            if(predicted == [0]):
##                print("Palm")
##            if(predicted == [1]):
##                print("Fist")
#
#cap = cv2.VideoCapture(0)
#i = 0
#now = int(round(time.time() * 1000))
#capture = True
#for j in range(number_of_gestures):
#    
#    while(cap.isOpened() & capture):
#                    
#        ret, img = cap.read()
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)       
#        
#        current = int(round(time.time() * 1000))
#        delay = current-now
#        cv2.putText(img,"Test gesture: %d" %j,(50,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
#        cv2.putText(img,"Test image: %d" %(i+1),(50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
#        cv2.putText(img,"%d" %delay,(50,130), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
#        cv2.imshow('Test',img)
#        #print(delay)
#        if(delay > image_delay*1000):
#            print("taking picture %d for test gesture %d" %(i,j))
#            now = int(round(time.time() * 1000))
#            test_set.append(gray)
#            test_target.append(j)
#            i+=1
#        if cv2.waitKey(1) & 0xff ==ord('q'):
#            break
#        if(i==number_of_pictures):
#           capture = False 
#
#test_data = np.array(test_set).reshape((n_sample,-1))
#test_target = np.array(test_target).reshape((n_sample,))
#
##test it on 2 images
##test = []
##pic = cv2.imread('Palm.jpg')
##pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
##test.append(pic)
##pic = cv2.imread('Fist.jpg')
##pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
##test.append(pic)
##test = np.array(test).reshape((2,-1))
##print(test.shape)
##expected = [0,1]
##expected = np.array(expected).reshape((2,-1))
##print(expected.shape)
#
#predicted = classifier.predict(test_data)
#print("predicted: ", predicted)
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_target, predicted))
#
#cap.release()
#cv2.destroyAllWindows()


    
