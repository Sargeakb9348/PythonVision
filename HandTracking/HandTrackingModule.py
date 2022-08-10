import math

import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=6, modelComplexity=1, minDetectionConfidence=0.75, minTrackingConfidence=0.75):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence=minTrackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.minDetectionConfidence, self.minTrackingConfidence)  # default parameters are fine
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNum=0, draw=True):
        xList = []
        yList= []
        self.lmList=[]
        if self.results.multi_hand_landmarks:

            myHand=self.results.multi_hand_landmarks[handNum]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # center location
                #print(id, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0,255,0),2)
        return self.lmList, bbox

def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
    x1, y1 = self.lmList[p1][1:0]
    x2, y2 = self.lmList[p2][1:0]
    cx, cy = (x1+x2)//2, (y1+y2)//2

    if draw:
        cv2.line(img, (x1,y1),(x2,y2),(0,255,0),t)
        cv2.circle(img,(x1,y1), r, (255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

    length = math.hypot(x2-x1, y2-y1)

    return length,img,[x1,y1,x2,y2,cx,cy]

def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[0])#print location of lmList["landmark"] 0 = wrist, 4 = thumb, 8 = index, 20 = pinky

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
