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
from sklearn.neural_network import MLPClassifier

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

def get_average(A):
    avg = 0
    rows = len(A)
    columns = len(A[0])
    temp = np.zeros((rows, 1))
    for row in range(rows):
        for column in range(columns):
            avg += A[row][column]
        temp[row][0] = avg/columns
    return temp
  
#load up all the images
for file in files:
    c_path = path+file
    temp = os.listdir(c_path)
    
    for f in temp:
        if( f.endswith('.jpg')): #only take the .jpg files
#            print(join(c_path,f))
            img = cv2.imread(join(c_path,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (200,200), cv2.INTER_LINEAR)
            img = cv2.GaussianBlur(img,(5,5),0)
            img = cv2.Canny(img,100,100)
#            img = get_average(img)
            img_set.append(img)
            Y_train.append(file)

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
            #get the everage of the column pixels to reduce dimentiality
#            img = get_average(img)
            #resize the image
            img = cv2.resize(img, (200,200), cv2.INTER_LINEAR)
            #apply blur
            img = cv2.GaussianBlur(img,(5,5),0)
            #detect the edges
            img = cv2.Canny(img,100,100)
            img_set.append(img)
            Y_test.append(file)

#imgplot = plt.imshow(img_set[0])
#plt.show()
#resize the images using  cv2.INTER_LINEAR to reduce dimentionality from 640*480 (4:3 AR) to (200,200) image
#t_img = cv2.resize(img_set[0], (200,200), cv2.INTER_LINEAR)
#cv2.imshow("new size", t_img)
#cv2.waitKey()

#t_img = cv2.Canny(img_set[0],100,100)
#cv2.imshow("edges", t_img)
#cv2.waitKey()

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
#cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state=42)
#grid = GridSearchCV(SVC(), param_grid = param_grid, cv=cv)
#start =  time.time()
#grid.fit(X_train, Y_train)
#current = time.time()
#delay = current - start
#print("delay = ", delay)
#print("the best params are %s with a ascore of %0.2f" %(grid.best_params_, grid.best_score_))
#>>the best params are {'C': 1.0, 'gamma': 1e-08} with a ascore of 0.97
#>>the best params are {'C': 1.0, 'gamma': 9.9999999999999995e-08} with a ascore of 0.97
#Train the classifer on the pictures
#clf = MLPClassifier(solver='lbfgs', alpha=1e-8, hidden_layer_sizes=(50, ), random_state=1)
#clf = GridSearchCV(SVC(), param_grid = param_grid, cv=cv)
#clf.fit(X_train, Y_train)
 
classifier = svm.SVC(probability = True, C = 1.0 , gamma=1.1e-7)
classifier.fit(X_train, Y_train)
#
##Test the classifier
predicted = classifier.predict(X_test)
##pre = clf.predict(X_test)
##print the confusion matrix
print("SVM Predicted shape = " , predicted.shape) 
print("SVM Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, predicted))
##print("MLP Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, pre))
#
#initialize camera
cap = cv2.VideoCapture(0)

now = int(round(time.time() * 1000))
while (cap.isOpened() & True):
     ret, img = cap.read()
     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     current = int(round(time.time() * 1000))
     #take a guess every 200 ms
     if(current - now > 200 ):
#         print(current - now)
         now = int(round(time.time() * 1000))
#         #get the everage of the column pixels to reduce dimentiality
#         test = get_average(gray)
#         flatten the image 
#         test  = gray.reshape(1,-1)
#         test = test.reshape(1,-1)
         test = cv2.resize(gray, (200,200), cv2.INTER_LINEAR)
         test = cv2.GaussianBlur(test,(5,5),0)
         test = cv2.Canny(test,100,100)
         test = test.reshape(1,-1)
         #predict the gesture
         pre = classifier.predict(test)
         cv2.putText(img,"gesture: %s" %pre[0],(50,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
         cv2.imshow('Training feed',img)          
     if cv2.waitKey(1) & 0xff ==ord('q'):
         break

cap.release()
cv2.destroyAllWindows()

#i = 0
#for item, label in zip(X_test, Y_test):
#    result = classifier.predict([item])
#    i += 1
#    if result != label:
#        print ("predicted label %s, but true label is %s for sample number %s" % (result, label,i))
#        imgplot = plt.imshow(img_set[i])

#print the probability of each sample
#print(classifier.classes_)
#
#for i in range(len(X_test)):
#    prob = classifier.predict_proba(X_test)[i]
#    print(prob)   
#  





    
    



