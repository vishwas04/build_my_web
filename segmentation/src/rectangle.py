import cv2
import numpy as np

# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white
# wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg
img = cv2.pyrDown(cv2.imread('test4.jpg', cv2.IMREAD_UNCHANGED))

# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
# find contours and get the external one

contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
#                cv2.CHAIN_APPROX_SIMPLE)

# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
l = []
for c in contours:
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
                cv2.rectangle(img, (x+6, y+6), (x+w-6, y+h-6), (0, 255, 0), 2)
                #cv2.imshow("contours", img)

# print(len(contours))
#cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imshow("contours", img)

while True:
    key = cv2.waitKey(1)
    if key == 27: #ESC key to break
        break

cv2.destroyAllWindows()