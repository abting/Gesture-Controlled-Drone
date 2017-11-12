import os

path = 'Gesture_Images/image_dataSet/'
files = os.listdir(path)
directory = 0

#loop through the directories in the path
for file in files:
    arb_number = 1000
    i = 0
    #loop through the files in each directory
    c_path = path+"%d" %directory
    temp = os.listdir(c_path)
    directory +=1
#    print(temp)
    #rename everything first to avoid duplicates
    for f in temp:
        if( f.endswith('.jpg')): #only change the .jpg files
            os.rename(os.path.join(c_path, f), os.path.join(c_path, str(arb_number)+'.jpg'))
            arb_number += 1
         
#    #now rename everything back to 0,1,2....
    for f in temp:
        if( f.endswith('.jpg')): #only change the .jpg files   
#            print(os.path.join(c_path, f))
            os.rename(os.path.join(c_path, f), os.path.join(c_path, str(i)+'.jpg'))
            i += 1