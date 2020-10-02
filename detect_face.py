import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='putting in image')
ap.add_argument('-p', '--prototxt', default='detection/deploy.prototxt')
ap.add_argument('-m', '--caffemodel', default='detection/res10_300x300_ssd_iter_140000.caffemodel')
ap.add_argument('-c', '--confidence', default=0.5)
ap.add_argument('-o', '--output', default='output/test.jpg')

args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['caffemodel'])

image = cv2.imread(args['input'])
try:  
    h, w = image.shape[:2]
except:
    raise Exception("Problem with the image")
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

print("object detections")
net.setInput(blob)
detections = net.forward()


for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    if confidence > args['confidence']:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        text = "{:2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 5)
        
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255,0, 0), 5)
        
cv2.imwrite(args['output'], image)
        
        