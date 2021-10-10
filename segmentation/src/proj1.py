import cv2
import numpy as np
import os
from infer import myfunc

parent_dir = '/Users/vishwas/Desktop/whatsapp/segmentation/src/result'

# read and scale down image
x=os.listdir("/Users/vishwas/Desktop/whatsapp/segmentation/data/test")
img = cv2.pyrDown(cv2.imread("/Users/vishwas/Desktop/whatsapp/segmentation/data/test/"+x[0], cv2.IMREAD_UNCHANGED))

# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)

# find contours and get the external one
contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
# with each contour, draw boundingRect in green
l = []
for c in contours:
    # get the bounding rect
    
    x, y, w, h = cv2.boundingRect(c)
    if(x != 0 and y != 0):
        flag = True
        for coord in l:
            if(abs(coord[0]-x) < 10 and abs(coord[1]-y) < 10):
                flag = False
        if(flag):
            if(w>30 and h>30 and w<700 and  h<700 and (w*h > 3000)):
                # print((x,y))
                l.append((x,y))
                #print(x,y)
                #draw a green rectangle to visualize the bounding rect
                #cv2.rectangle(img, (x+6, y+6), (x+w-6, y+h-6), (0, 255, 0), 2)
                #cv2.imshow("contours", img)
                cropped_img = img[y+5:y+h-5, x+5:x+w-5]
                dir_name = '('+str(x)+','+str(y)+','+str(x+w)+','+str(y+h)+')'
                path = os.path.join(parent_dir,dir_name)
                os.mkdir(path)
                cv2.imwrite(os.path.join(path,'rectangle.jpg'),cropped_img)
                myfunc(dir_name)
cv2.imshow("contours", img)


            

#print(len(contours))
#cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
#cv2.imshow("contours", img)
