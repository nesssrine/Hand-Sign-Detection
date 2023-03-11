
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
folder = "Data/Y"
counter = 0


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #3malna objet men classe HandDetector ydectecti maximuim 1 main

offset = 20
imgSize = 300
while True:
    success, img = cap.read() # mech y7ell caméra
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
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) #ken ki nenzel 3la  key mech yaamel save
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)