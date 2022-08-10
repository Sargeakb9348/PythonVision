import cv2
import numpy as np
import HandTrackingModule as htm
import time
import os
import pyautogui as pyauto
from win32api import GetSystemMetrics
import math


camWidth, camHeight = 640, 480
displayWidth, displayHeight = GetSystemMetrics(0), GetSystemMetrics(1)
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
previousTime=0
detector = htm.handDetector(maxHands=1)

while True:
    #find landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)


    #get tip of index and thumb
    if len(lmList)!=0:
        #index finger tip
        x1,y1 = lmList[8][1:]
        #thumb tip
        x2,y2 = lmList[4][1:]

        midx: int = (x2-x1)//2
        midy: int = (y2-y1)//2
        length = math.hypot(x2 - x1, y2 - y1)

    #convert webcam coordinate to display coordinate resolution
    x3 = np.interp(x1, (0,camWidth), (0,displayWidth))
    y3 = np.interp(y1, (0, camHeight), (0, displayHeight))

    x4 = np.interp(x2, (0, camWidth), (0, displayWidth))
    y4 = np.interp(y2, (0, camHeight), (0, displayHeight))

    print(length)
    #print(detector.findDistance(8,4))
    #findDistance() not working for some reason
    #if fingers are apart
    if (length>12.0):
        #move mouse
        pyauto.moveTo(displayWidth-x4,y4)
    else:
        #click
        pyauto.mouseDown(displayWidth-x4,y4)
    #show fps
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    #display
    cv2.imshow("Image", img)
    cv2.waitKey(1)#delay

