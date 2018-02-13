import os
import cv2
import time

img_path = 'Gesture_Images/'   #relative path to where the pictures should be saved
number_of_gestures = 2         #number of gestures
image_delay = 1             #(seconds)time to wait between taking images
number_of_pictures = 5        #number of pictures to take PER gesture

for i in range(1000):
    i += 1

#initialize camera
cap = cv2.VideoCapture(0)

#if the folder does not exist make the folders
if not os.path.exists(img_path):
    os.makedirs(img_path)

j = 0
for j in range(number_of_gestures):
    path = img_path + str(j)
    if not os.path.exists(path):
        os.makedirs(path)

#get the list of the directories and remove 'image_dataSet'
files = os.listdir(img_path)

#loop through the folders and get all the files
distance = "1-5m"

j = 0
for file in files:
    temp = os.path.join(img_path, file)
                
    latest_image_name = 0
    print("lates image name: ", latest_image_name)        
    print('current folder', file)    
    
    n_images = 0
    now = int(round(time.time() * 1000))
    capture = True
    while(cap.isOpened() & capture ):
                
        ret, img = cap.read()
        copy = img.copy()
        
        current = int(round(time.time() * 1000))
        delay = current-now
        cv2.putText(copy,"gesture: %d" %j,(50,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), thickness = 2)
        cv2.putText(copy,"image: %d" %(latest_image_name+1),(50,140), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), thickness = 2)
        cv2.putText(copy,"%d" %delay,(50,210), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), thickness = 2)
        cv2.imshow('Train',copy)
        #print(delay)
        if(delay > image_delay*1000):
#            print("taking picture %d for gesture %d" %(latest_image_name,j))
            now = int(round(time.time() * 1000))
            #cv2.imwrite(temp + '/' + distance +'_' +' %d.bmp' %latest_image_name,img)
            latest_image_name+=1
            n_images += 1
        if cv2.waitKey(1) & 0xff ==ord('q'):
            break
        if(n_images==number_of_pictures):
           capture = False 
    j +=1           
    
cap.release()
cv2.destroyAllWindows()
#        

