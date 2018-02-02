import numpy as np
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from os.path import join
import itertools
import time
import os
import cv2
import pickle

img_set    = []
labels     = []
flat_image = []
height, width  = 70,70
   
path = 'Gesture_Images/'
files = os.listdir(path)

for file in files:
    c_path = path+file
    temp = os.listdir(c_path)
    
    for f in temp:
        if( f.endswith('.jpg') or f.endswith('.bmp') ):
            img = cv2.imread(join(c_path,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (height,width), cv2.INTER_LINEAR)          
            flip=cv2.flip(img,1)   
            img_set.append(img)
            img_set.append(flip)
            labels.append(file)
            labels.append(file)

#plt.imshow(img_set[1])

flat_image = np.array(img_set)
n_sample = len(flat_image) 
flat_image = flat_image.reshape((n_sample,-1)) 
labels = np.array(labels) 
labels = labels.reshape((n_sample,)) 

X_train, X_test, Y_train, Y_test = train_test_split(flat_image, labels, test_size=0.1, random_state=42)
print("X_test shape  = " , X_test.shape)  
print("Y_test shape  = " , Y_test.shape) 
print("X_train shape = " , X_train.shape)  
print("Y_train shape = " , Y_train.shape)      

scaler = StandardScaler()
scaler.fit(X_train) 
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)  

#alpha_range = np.logspace(-4,10,13)
#params = {'alpha': alpha_range}
#mlp = MLPClassifier(activation='relu', solver='sgd', learning_rate_init = 0.03, max_iter=500, 
#                        hidden_layer_sizes=(320,) , random_state=1, verbose= True, learning_rate = 'adaptive', tol=1e-4)
#
#clf = GridSearchCV(mlp, params, verbose=10, n_jobs=1, cv=5)
#clf.fit(X_train, Y_train)
#print("the best params are %s with a ascore of %0.2f" %(clf.best_params_, clf.best_score_))

#random_search = RandomizedSearchCV(mlp, param_distributions=params, n_iter=13, verbose=10, n_jobs=1)
#random_search.fit(X_train, Y_train)
#print("the best params are %s with a ascore of %0.2f" %(random_search.best_params_, random_search.best_score_))

#SVM CLASSIFIER
#classifier = svm.SVC(probability = True, C = 1.0 , gamma=0.008)
#start = time.time()
#classifier.fit(X_train, Y_train)
#delay = time.time() - start
#print("SVM training time: %s" %delay)

#MULTI-LAYER PERCEPTRON CLASSIFIER
Train = False
if(Train):   #(270,270,270) --> 90%
    mlp = MLPClassifier(activation='relu', solver='sgd', learning_rate_init = 0.03, alpha=0.008, max_iter=500, 
                            hidden_layer_sizes=(271,271,271) , random_state=1, verbose = True, learning_rate = 'adaptive', tol=1e-4)
    
    start = time.time()
    mlp.fit(X_train, Y_train)
    delay_training = time.time() - start
    print("***************************************")
    print("MLP training time: %s" %delay_training)

Save_Model = False
Load_Model = True
filename = 'finalized_model_4_gesture.sav'
if(Save_Model):
    pickle.dump(mlp, open(filename, 'wb'))
if(Load_Model):
    mlp = pickle.load(open(filename, 'rb'))
    
#Test the classifiers
#---------------------------------------------------------------------------------
#start = time.time()
#SVMpredicted  = classifier.predict(X_test)
#delay = time.time() - start
#print("SVM fitting time: %s" %delay)

start = time.time()
MLPpredicted  = mlp.predict(X_test)
delay_fitting = time.time() - start
print("MLP fitting time: %s" %delay_fitting)
print("MLP training Score: %s" % mlp.score(X_train, Y_train))

#print("******************************************************************")
#print("SVM Predicted shape = " , SVMpredicted.shape) 
#print("SVM Mean Score : %s" % classifier.score(X_test, Y_test))
#print("SVM training Score : %s" % classifier.score(X_test, Y_test))
#print("***************************************")
print("MLP Predicted shape: " , MLPpredicted.shape) 
print("MLP fitting Score: %s" % mlp.score(X_test, Y_test))
print("MLP hidden layers shape: %s" %(mlp.hidden_layer_sizes,))
print("***************************************")

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Greens):

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
plot_confusion_matrix(cnf_matrix, classes=('0','1','2'),title='Confusion matrix, without normalization', normalize=False)    
    
    