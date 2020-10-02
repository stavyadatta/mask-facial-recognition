import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-d", "--detector", default="detection")
ap.add_argument("-m", "--embedding-model", default="openface_nn4.small2.v1.t7")
ap.add_argument("-r", "--recognizer", default='output/recognizer.pickle')
ap.add_argument("-l", "--le", default='output/le.pickle')
ap.add_argument('-c', "--confidence", default=0.5, type=float)
ap.add_argument('-o', '--output_image', default="output_image/test.jpg")


args = vars(ap.parse_args())

protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])

embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), 
                                (104.0, 177.0, 123.0), swapRB=False, crop=False)

detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        
        if fW < 20 or fH < 20:
            continue
        
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), 
                                        (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()
        
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        if proba > 0.5:
            name = le.classes_[j]
            
            text = "{}, {:.2f}".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
            cv2.putText(image, text, (startX + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
cv2.imwrite(args["output_image"], image)