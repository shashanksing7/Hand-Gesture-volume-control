import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



cap=cv.VideoCapture(0,cv.CAP_DSHOW)

mpHands=mp.solutions.hands
hands=mpHands.Hands(min_detection_confidence=0.7,max_num_hands=1)

mpDraw=mp.solutions.drawing_utils

drawsec=mpDraw.DrawingSpec(color=(0,255,0))

ptm=0




devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol=volume.GetVolumeRange()
maxvol=vol[1]
minvol=vol[0]
# print(maxvol,minvol)


while True:

    ret,frame=cap.read()

    frame=cv.flip(frame,2)
    framergb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    ctm=time.time()

    fps=1/(ctm-ptm)
    ptm=ctm

    listm = []

    h, w, c = frame.shape




    # print(fps)

    text="fps:"+str(int(fps))

    result=hands.process(framergb)

    # print(result.multi_hand_landmarks)

    if  result.multi_hand_landmarks:

        for mhandl in result.multi_hand_landmarks:

            for id,lm in enumerate(mhandl.landmark):









                cx,cy=int(lm.x*w),int(lm.y*h)

                listm.append([id,cx,cy])

                # print(listm[::])
                # print(id,cx,cy)






            mpDraw.draw_landmarks(frame,mhandl,mpHands.HAND_CONNECTIONS,drawsec)

            if len(listm) !=0:

                 x1,y1=listm[4][1],listm[4][2]
                 x2,y2=listm[8][1],listm[8][2]

                 mx,my=int((x1+x2)//2),int((y1+y2)//2)

                 cv.circle(frame,(x1,y1),15,(175,0,0),-1)
                 cv.circle(frame,(x2,y2),15,(175,0,0),-1)
                 cv.line(frame,(x1,y1),(x2,y2),(175,0,0),6)
                 cv.circle(frame,(mx,my),15,(175,0,0),-1)

                 length=math.hypot(x2-x1,y2-y1)

                 if length<=100:
                     cv.circle(frame, (mx, my), 15, (0, 255, 0), -1)



                 range =np.interp(length,[50,180],[minvol,maxvol])
                 print(range)
                 volume.SetMasterVolumeLevel(range, None)



                  # print(length)


    cv.putText(frame,text,(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(128,0,0),3,cv.LINE_AA)



    cv.imshow("video",frame)

    if cv.waitKey(5) & 0xFF==ord("q"):
        break


cap.release()
cv.destroyAllWindows()