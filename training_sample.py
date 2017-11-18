import numpy as np
from sklearn import svm, metrics
import os
import cv2
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
import time

n_gestures = 2
n_images = 10
img_set = []

X_train = []
Y_train = []

X_test = []
Y_test = []

path = 'Gesture_Images/image_dataSet/'
files = os.listdir(path)
directory = 0

#load up all the images
for file in files:
    c_path = path+file
    temp = os.listdir(c_path)
    
    for f in temp:
        if( f.endswith('.jpg')): #only take the .jpg files
#            print(join(c_path,f))
            img = cv2.imread(join(c_path,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_set.append(img)
            Y_train.append(file)


#imgplot = plt.imshow(img_set[0])
#cv2.imwrite('BEFORE.jpg',img_set[0])
##resize the images using  cv2.INTER_LINEAR to reduce dimentionality from 640*480 (4:3 AR) to (200,200) image

#i = 0
#for image in img_set:
#    img_set[i] = cv2.equalizeHist(image)
#    i += 1
    
#i = 0
#for image in img_set:
#    img_set[i] = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA )
#    i += 1    
#        
#imgplot = plt.imshow(img_set[0])
#cv2.imwrite('AFTER.jpg',img_set[0])
# 
X_train = np.array(img_set)
n_sample = len(X_train) 
X_train = X_train.reshape((n_sample,-1))    

Y_train = np.array(Y_train) 
Y_train = Y_train.reshape((n_sample,))    
  
print("X_train shape = " , X_train.shape)  
print("Y_train shape = " , Y_train.shape)          

path = 'Gesture_Images/'
files = os.listdir(path)
files.remove('image_dataSet')
img_set = []

for file in files:
    c_path = path+file
    temp = os.listdir(c_path)
    
    for f in temp:
        if( f.endswith('.jpg')): #only take the .jpg files
#            print(join(c_path,f))
            img = cv2.imread(join(c_path,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_set.append(img)
            Y_test.append(file)

#resize the images using  cv2.INTER_LINEAR to reduce dimentionality from 640*480 (4:3 AR) to (200,200) image

#i = 0
#for image in img_set:
#    img_set[i] = cv2.equalizeHist(image)
#    i += 1

#i = 0
#for image in img_set:
#    img_set[i] = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA )
#    i += 1
#    
X_test = np.array(img_set)
n_sample = len(X_test) 
X_test = X_test.reshape((n_sample,-1))    

Y_test = np.array(Y_test) 
Y_test = Y_test.reshape((n_sample,)) 

print("X_test shape = " , X_test.shape)  
print("Y_test shape = " , Y_test.shape)      

#C_range = np.logspace(-2,10,13)
#gamma_range = np.logspace(-9,3,13)
#param_grid = dict(gamma = gamma_range, C = C_range)
#
#cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state=42)
#grid = GridSearchCV(SVC(), param_grid = param_grid, cv=cv)
#
#start =  time.time()
#
#grid.fit(X_train, Y_train)
#
#current = time.time()
#
#delay = current - start
#print("delay = ", delay)
#
#print("the best params are %s with a ascore of %0.2f" %(grid.best_params_, grid.best_score_))
#>>the best params are {'C': 1.0, 'gamma': 1e-08} with a ascore of 0.97

#Train the classifer on the pictures
classifier = svm.SVC(probability = True, C = 1.0 , gamma=1e-08)
classifier.fit(X_train, Y_train)

##Test the classifier
predicted = classifier.predict(X_test)

#print the confusion matrix
print("Predicted shape = " , predicted.shape) 
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, predicted))

#i = 0
#for item, label in zip(X_test, Y_test):
#    result = classifier.predict([item])
#    i += 1
#    if result != label:
#        print ("predicted label %s, but true label is %s for sample number %s" % (result, label,i))
#        imgplot = plt.imshow(img_set[i])

#print the probability of each sample
print(classifier.classes_)
for i in range(len(X_test)):
    prob = classifier.predict_proba(X_test)[i]
    print(prob)   
  





    
    



