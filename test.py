
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
folder = "Data/C"
counter = 0


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #3malna objet men classe HandDetector ydectecti maximuim 1 main
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]

while True:
    success, img = cap.read() # mech y7ell caméra
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # 3malna image square ( matrice de 1 de taille n*n ) de size 300
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        # men gher offcet ya5edh dimension enta3 box 7aseb kober idi may5alich chwaya espace
        imgCropShape = imgCrop.shape
        #imgWhite[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop #3malna superposition lil impwhite w li imgcrop
        aspectRatio = h / w
        if aspectRatio > 1: # ken valeur akber men 1 maaneha image mechia bi toul
            k = imgSize / h # ki temchi bi toul nchoufou valeur w jdida
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)-1
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.putText(imgOutput,labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1) #ken ki nenzel 3la  key mech yaamel save
