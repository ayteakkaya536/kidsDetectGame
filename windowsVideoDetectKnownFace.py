### TESTED WORKING on WINDOWS#####
### libraries downloaded with pip, ubuntu/jetpack is different
# python --version
# XX python version 3.9
# pip --version
# XX pip version 21.0.1
# pip install --upgrade pip
# pip install face-recognition
# pip install opencv-python

## THIS DOES NOT TRAIN -- TRAIN THE PICfirst and save as train.pkl
## use the file windowsTrainSaveKnownFaces

## FPS rate comes out as 0.1 running on my laptop
## WATCH VIDEO 38 --> dlib libraries uses CUDA --> your code does not use GPU



import face_recognition
import cv2
import os
import pickle
import time
# from aytekClass import aytekVStream
from threading import Thread
print(cv2.__version__)

# fpsReport=0
scaleFactor=.3

Encodings=[]
Names=[]

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX

## without vStream, wihtout threading
# cam= cv2.VideoCapture(0) # 0 for built in camera, if USB change it to 1

## with vStream, and threading
class aytekVStream:
    def __init__(self,srcCamera,width,height):
        self.width=width
        self.height=height
        self.capture=cv2.VideoCapture(srcCamera)
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while True:
            _,self.frame=self.capture.read() # self means every instance (object) is difefrent, otherwise it would overwrite for each camera (cam1, cam2)
            self.frame2=cv2.resize(self.frame,(self.width,self.height))
    def getFrame(self):
        return self.frame
    # def setStream(self,srcCamera,width,height):
    #     self.width=width
    #     self.height=height
    #     self.srcCamera=srcCamera
dispW=640
dispH=480
cam=aytekVStream(0,dispW,dispH)

#FPS COUNT
timeMark=time.time()
fpsFilter=0

while True:
    try:
        # time.sleep(10)
        frame=cam.getFrame()
        # _,frame=cam.read()
        frameSmall=cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
        frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
        facePositions=face_recognition.face_locations(frameRGB,model='cnn')
        allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):

            name='Unkown Person'
            matches=face_recognition.compare_faces(Encodings,face_encoding)
            if True in matches:
                first_match_index=matches.index(True)
                name=Names[first_match_index]
            top=int(top/scaleFactor)
            right=int(right/scaleFactor)
            bottom=int(bottom/scaleFactor)
            left=int(left/scaleFactor)
            cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
            cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)
        # FPS TIME CALCUALATION
        dt=time.time()-timeMark
        timeMark=time.time()
        fps=1/dt
        fpsFilter=.95*fpsFilter + .05 *fps
        cv2.rectangle(frame,(0,0),(100, 40),(0,0,255),-1)
        cv2.putText(frame, str(round(fpsFilter,2))+ 'fps',(0,25),font,.75,(0,255,255,2))
        cv2.imshow('Picture',frame)
        cv2.moveWindow('Picture',0,0)
    except:
        print(" frame not availebale ")
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
exit()
