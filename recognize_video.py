from detect_mask import mask_detection_image
import numpy as np
import argparse
import imutils
import pickle
import cv2
from tensorflow.keras.models import load_model
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video")
ap.add_argument("-d", "--detector", default="detection")
ap.add_argument("-m", "--embedding-model", default="openface_nn4.small2.v1.t7")
ap.add_argument("-r", "--recognizer", default='output/recognizer.pickle')
ap.add_argument("-l", "--le", default='output/le.pickle')
ap.add_argument('-c', "--confidence", default=0.5, type=float)
ap.add_argument('-o', '--output_video', default="output_video/test.avi")


args = vars(ap.parse_args())

protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])

width = 0
height = 0
if args['video']:
    vid = cv2.VideoCapture(args['video'])
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('filename.avi', codec, 10, (width, height))
else:
    vid = cv2.VideoCapture(0)

if (vid.isOpened() == False):  
    print("Error reading video file")

frame_width = int(vid.get(3)) 
frame_height = int(vid.get(4)) 
print(frame_width, frame_height)
size = (frame_width, frame_height) 
out = cv2.VideoWriter(args['output_video'],  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        10, size) 

embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


prototxtPath = os.path.sep.join(['detection', "deploy.prototxt"])
weightsPath = os.path.sep.join(['detection',
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model('mask_detector_keras_new_dataset.model')
i = 0
while True:

    
    grabbed, frame = vid.read()
    i = i + 1
    print(i)
    cv2.imwrite(f'images/me{i}.jpg', frame)  
    if not grabbed:
        break
    
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = frame
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
            print(proba)
            if proba > 0.5:
                print((float(proba)) * 100)
                name = le.classes_[j]
                
                text = "{}, {:.2f}".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                            (230, 0, 0), 2)
                cv2.putText(image, text, (startX + 10, y + 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 0, 0), 2)
                
            
    

    image = mask_detection_image(image, faceNet, maskNet)
    if args['video']:   
        image = cv2.resize(image,(width,height))
    else:
        image = cv2.resize(image, (640, 480))
    
    out.write(image)
    

cv2.destroyAllWindows()


                
