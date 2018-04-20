import numpy as np
import cv2
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import itertools
from os.path import join
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

model = load_model('./Keras_models/model_3.h5')

#%% test on new images
img_set    = []
labels     = []
flat_image = []
path = './Test_Gesture_Images/'
files = os.listdir(path)

for file in files:
    c_path = path+file
    temp = os.listdir(c_path)
    
    for f in temp:
        if( f.endswith('.jpg') or f.endswith('.bmp') ):
            img = cv2.imread(join(c_path,f), 0)
            img = cv2.resize(img, (70,70), cv2.INTER_LINEAR)          
            img_set.append(img)
            labels.append(int(file))

flat_image = np.array(img_set)
flat_image = flat_image.astype('float32')
n_sample = len(flat_image) 
flat_image /= 255
flat_image = flat_image.reshape(n_sample,70,70,1)
labels = np.array(labels) 
labels = labels.reshape((n_sample,)) 
labels = to_categorical(labels)
flat_image, labels = shuffle(flat_image, labels, random_state = 2)

target_names = ['class 0(Palm)', 'class 1(Fist)', 'class 2(Peace)','class 3(OK)']
Y_pred = model.predict(flat_image)
y_pred = np.argmax(Y_pred, axis=1)
cnf_matrix = (confusion_matrix(np.argmax(labels,axis=1), y_pred))
np.set_printoptions(precision=2)
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=target_names,normalize=False,title='Confusion matrix New Images')   
#plt.savefig('CM_DR.SELMIC.png', dpi=300)

score,acc = model.evaluate(flat_image, labels, batch_size=50)
print('Test score:', score)
print('Test accuracy:', acc)