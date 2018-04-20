from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
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
num_epoch = 16

path = './Gesture_Images/'
files = os.listdir(path)

for file in files:
    c_path = path+file
    temp = os.listdir(c_path)
    
    for f in temp:
        if( f.endswith('.jpg') or f.endswith('.bmp') ):
            img = cv2.imread(join(c_path,f), 0)
            img = cv2.resize(img, (height,width), cv2.INTER_LINEAR)          
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

X_train, X_test, Y_train, Y_test = train_test_split(flat_image, labels, test_size=0.3, random_state=42)

datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=True,
        fill_mode="nearest",
        shear_range=0.5)

datagen.fit(X_train)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (70,70,1)  ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4))  
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

hist = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = 32),steps_per_epoch = len(X_train)/32,
                               epochs=num_epoch,validation_data=(X_test, Y_test) )

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
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

target_names = ['class 0(Palm)', 'class 1(Fist)', 'class 2(Peace)','class 3(OK)']			
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):

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
#plt.figure(1)
##plt.savefig("loss1.png", dpi=300)
#plt.figure(2)
##plt.savefig("loss2.png", dpi=300)

plot_confusion_matrix(cnf_matrix, classes=target_names,normalize=False,title='Confusion matrix')    
    
user_input = input("Save model? [y/n]")
if(user_input == 'y'):
    model_name = 'model_3.h5'
    model.save(model_name)
    print('Model saved!', model_name)






