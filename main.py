exi
import dlib
from imutils import face_utils

img = cv2.imread('img1.jpg')

#--------Model Path---------#
proto_file = 'SSD_MobileNet_prototxt.txt'
model_file = 'SSD_MobileNet.caffemodel'

#------Variables for the Model ---------#
classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 
              3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car',
              8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 
              12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 
              16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 
              19: 'train', 20: 'tvmonitor'}

input_shape = (300, 300)
mean = (127.5, 127.5, 127.5)
scale = 0.007843

#---------Load The Model--------#
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

#----Loading the image--------#
img = cv2.resize(img, (720, 620))
#------image preprocessing----#
blob = cv2.dnn.blobFromImage(img,
                             scalefactor=scale,
                             size=input_shape,
                             mean=mean,
                             swapRB=True)  
# since our image is already in the BGR form

net.setInput(blob)
results = net.forward()
for i in range(results.shape[2]):
  
  	# confidence
    confidence = results[0, 0, i, 2]  
    if confidence > 0.7:
      
      	# class id
        id = int(results[0, 0, i, 1])  
        
        # 3-6 contains the coordinate
        x1, y1, x2, y2 = results[0, 0, i, 3:7]  
        
        # print(x1,y1,x2,y2)
        # scale these coordinates to out image pixel
        ih, iw, ic = img.shape
        x1, x2 = int(x1*iw), int(x2*iw)
        y1, y2 = int(y1 * ih), int(y2 * ih)
        cv2.rectangle(img,
                      (x1, y1),
                      (x2, y2),
                      (0, 200, 0), 2)
        cv2.putText(img, f'{classNames[id]}:{confidence}',
                    (x1+30, y1-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 1)
    # print(results[0,0,i,:])


cv2.imshow('Image', img)
cv2.waitKey()
