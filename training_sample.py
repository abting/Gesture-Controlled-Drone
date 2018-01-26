import numpy as np
from sklearn import svm, metrics
import os
import cv2
from os.path import join
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import itertools

n_gestures = 2
n_images = 10
img_set = []
X_train = []
Y_train = []
X_test = []
Y_test = []
height = 70
width  = 70

path = 'Gesture_Images/image_dataSet/'
files = os.listdir(path)

#load up all the images
for file in files:
    c_path = path+file
    temp = os.listdir(c_path)
    
    for f in temp:
        if( f.endswith('.jpg') or f.endswith('.bmp') ): #only take the .jpg files
            img = cv2.imread(join(c_path,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (height,width), cv2.INTER_LINEAR)
#            img = cv2.Canny(img,100,100)           
            flip=cv2.flip(img,1)    
            img_set.append(img)
            img_set.append(flip)
            Y_train.append(file)
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
        if( f.endswith('.jpg') or f.endswith('.bmp')  ): #only take the .jpg files

            img = cv2.imread(join(c_path,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (height,width), cv2.INTER_LINEAR)
#            img = cv2.Canny(img,100,100)           
            flip=cv2.flip(img,1)   
            img_set.append(img)
            img_set.append(flip)
            Y_test.append(file)
            Y_test.append(file)

#plt.imshow(img_set[1])

X_test = np.array(img_set)
n_sample = len(X_test) 
X_test = X_test.reshape((n_sample,-1)) 
Y_test = np.array(Y_test) 
Y_test = Y_test.reshape((n_sample,)) 
print("X_test shape = " , X_test.shape)  
print("Y_test shape = " , Y_test.shape)      
print("******************************************************************")

#C_range = np.logspace(-2,10,13)
#gamma_range = np.logspace(-9,3,13)
#print(len(gamma_range))
#param_grid = dict(gamma = gamma_range)
#cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state=42)
#grid = GridSearchCV(MLPClassifier(), param_grid = param_grid, cv=cv)
#grid.fit(X_train, Y_train)

#params = {'hidden_layer_sizes': [(128,10,10)], 'alpha': gamma_range}
#mlp = MLPClassifier(solver='lbfgs', verbose=10, learning_rate='adaptive')
#clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
#clf.fit(X_train, Y_train)
#        
#print("the best params are %s with a ascore of %0.2f" %(clf.best_params_, clf.best_score_))
#>>the best params are {'C': 1.0, 'gamma': 1e-08} with a ascore of 0.97
#>>the best params are {'C': 1.0, 'gamma': 9.9999999999999995e-08} with a ascore of 0.97

#Train the classifer on the pictures
#clf = MLPClassifier(solver='lbfgs', alpha=1e-8, hidden_layer_sizes=(50, ), random_state=1)
#clf = GridSearchCV(SVC(), param_grid = param_grid, cv=cv)
#clf.fit(X_train, Y_train)

#SVM CLASSIFIER
#classifier = svm.SVC(probability = True, C = 1.0 , gamma=1.2e-9)
#start = time.time()
#classifier.fit(X_train, Y_train)
#delay = time.time() - start
#print("SVM training time: %s" %delay)

#MULTI-LAYER PERCEPTRON CLASSIFIER
scaler = StandardScaler()
scaler.fit(X_train) 
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)  

mlp = MLPClassifier(activation='relu', solver='sgd', learning_rate_init = 0.001, alpha=0.008, max_iter=2000, hidden_layer_sizes=(200,200,200) , random_state=1)
start = time.time()
mlp.fit(X_train, Y_train)
delay = time.time() - start
print("MLP training time: %s" %delay)

#---------------------------------------------------------------------------------
#Test the classifiers
#start = time.time()
#SVMpredicted  = classifier.predict(X_test)
#delay = time.time() - start
#print("SVM fitting time: %s" %delay)

start = time.time()
MLPpredicted  = mlp.predict(X_test)
delay = time.time() - start
print("MLP fitting time: %s" %delay)

##print the confusion matrix
#print("******************************************************************")
#print("SVM Predicted shape = " , SVMpredicted.shape) 
#print("SVM Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, SVMpredicted))
#print("SVM Mean Score : %s" % classifier.score(X_test, Y_test))
print("***************************************")
print("MLP Predicted shape = " , MLPpredicted.shape) 
print("MLP Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, MLPpredicted))
print("MLP Mean Score : %s" % mlp.score(X_test, Y_test))
print(mlp.hidden_layer_sizes)
#print("MLP Confusion matrix:\n%s" % metrics.confusion_matrix(Y_test, pre))

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
cnf_matrix = confusion_matrix(Y_test, MLPpredicted)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=('0','1','2'),title='Confusion matrix, without normalization')    
    
    