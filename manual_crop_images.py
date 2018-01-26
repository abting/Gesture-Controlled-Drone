import cv2
import os

#open the images
#crop the regions selected by the mouse ONLY GO FROM TOP LEFT TO BOTTOM RIGHT
#save into another folder
#go to next image in folder until you are out of pictures in the initial folder

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
image = None

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
   global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
   if event == cv2.EVENT_LBUTTONDOWN:
       refPt = [(x, y)]
       cropping = True
 
	# check to see if the left mouse button was released
   elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        cropping = False
   
		# draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
          
source = './0/'
destination = './0_cropped/'

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

latest_image_name = 0
done = False

for file in os.listdir(source):
   if(done != True) :
    if file.endswith(".jpg"):
        
        img_path = os.path.join(source, file)
        print("current image: " + img_path)
        image = cv2.imread(img_path)
        clone = image.copy()
        
        while True:
            
            cv2.imshow("image", image)
            
            k = cv2.waitKey(50) & 0xFF
            
            if k == ord("x"):
                break
            
            elif k == 27: #ESC 
                print('quitting')
                done = True
                break
            
            elif k == ord("z"):
                print("image reloaded!")
                image = clone.copy()
                
            elif k == ord("c") and len(refPt) == 2:                
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]     
                success = cv2.imwrite(destination + '%d.bmp' %latest_image_name,roi) 
                
                if(success == True):
                    print("successful")
                    print("image  %d cropped!" %latest_image_name)
                    latest_image_name += 1
                    image = clone.copy()
                elif success ==False:
                    print("not successful")
                    image = clone.copy()
                    

cv2.destroyAllWindows()
 
    
