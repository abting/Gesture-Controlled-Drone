from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from os.path import join
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.utils import plot_model
from sklearn.utils import shuffle
import itertools

img_set    = []
labels     = []
flat_image = []
height, width  = 70,70
num_epoch = 20
   
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
            labels.append(int(file))
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

X_train, X_test, Y_train, Y_test = train_test_split(flat_image, labels, test_size=0.1, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (70,70,1)  ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(4))  #number of classes
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

train = True
if(train):
    hist = model.fit(X_train, Y_train, batch_size=250, epochs=num_epoch, verbose=1,validation_data=(X_test, Y_test))

save_model = True
if(save_model):
    model.save('keras_model_3.h5')

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#test new image
test_image = cv2.imread('test.bmp')
test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (height,width))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
test_image= np.expand_dims(test_image, axis=3) 
test_image= np.expand_dims(test_image, axis=0)
#
load = False
if(load):
    model = load_model('keras_model_2.h5')
    
y_prob = model.predict(test_image) 
y_classes = y_prob.argmax(axis=-1)
print(y_classes)
print((model.predict(test_image)))
print(model.predict_classes(test_image))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

target_names = ['class 0(Palm)', 'class 1(Fist)', 'class 2(Peace)','class 3(Call Sign)']
					
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
cnf_matrix = (confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,title='Confusion matrix')    
    
model.summary();
  