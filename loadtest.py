import pickle
import cv2

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (70,70), cv2.INTER_LINEAR) 
img = img.reshape(1,-1)

filename = 'finalized_model.sav'
mlp = pickle.load(open(filename, 'rb'))

predicted = mlp.predict(img)
print("predicted array shape = ", predicted.shape)
print(predicted)

prob = mlp.predict_proba(img)
print(prob)