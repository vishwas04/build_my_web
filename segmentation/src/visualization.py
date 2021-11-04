import cv2
import matplotlib.pyplot as plt
import numpy as np
import os



def visualize(img, aabbs,img_dir):
    parent_dir = 'segmentation/src/result/'
    img = ((img + 0.5) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    boxes = []
    for aabb in aabbs:
        aabb = aabb.enlarge_to_int_grid().as_type(int)
        if((aabb.xmax-aabb.xmin)>10 and (aabb.ymax-aabb.ymin)>10 and ((aabb.xmax-aabb.xmin)*(aabb.ymax-aabb.ymin)) > 150 and (aabb.xmax-aabb.xmin)*(aabb.ymax-aabb.ymin)<50000):
            boxes.append((aabb.xmin, aabb.ymin,aabb.xmax, aabb.ymax))
            #cv2.rectangle(img, (aabb.xmin, aabb.ymin), (aabb.xmax, aabb.ymax), (255, 0, 255), 2)
    lines = sort_words(boxes)
    path = os.path.join(parent_dir+img_dir,'detect')
    os.mkdir(path)
    i = 0
    height = img.shape[0]
    width = img.shape[1]
    pw = 2
    ph = 2
    for line in lines:
        j = 0
        for (x1, y1, x2, y2) in line:
            # roi = text[y1:y2, x1:x2]
            cropImg = img[max(0,y1-ph):min(y2+ph,height), max(0,x1-pw):min(x2+pw,width)]
            cv2.imwrite(os.path.join(path,'line_'+str(i)+'_'+str(j)+'.jpg'),cropImg)
            j += 1
        i += 1
    #cv2.imshow("detect",img)


def visualize_and_plot(img, aabbs):
    plt.imshow(img, cmap='gray')
    height = img.shape[0]
    width = img.shape[1]
    pw = 10
    ph = 5
    for aabb in aabbs:
        if((aabb.xmax-aabb.xmin)>10 and (aabb.ymax-aabb.ymin)>10 and ((aabb.xmax-aabb.xmin)*(aabb.ymax-aabb.ymin)) > 150 and (aabb.xmax-aabb.xmin)*(aabb.ymax-aabb.ymin)<50000):
            plt.plot([max(0,aabb.xmin-pw), max(0,aabb.xmin-pw), min(aabb.xmax+pw,width), min(aabb.xmax+pw,width), max(0,aabb.xmin-pw)],
                     [max(0,aabb.ymin-ph), min(aabb.ymax+ph,height), min(aabb.ymax+ph,height), max(0,aabb.ymin-ph), max(0,aabb.ymin-ph)])

    plt.show()
    
def sort_words(boxes):
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    if(len(boxes) == 0):
        return []
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)
    
    #boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []
    for box in boxes:
        if box[1] > current_line + mean_height:
            lines.append(tmp_line)
            tmp_line = [box]
            current_line = box[1]            
            continue
        tmp_line.append(box)
    lines.append(tmp_line)
        
    for line in lines:
        line.sort(key=lambda box: box[0])
    # print(lines)
    return lines
