# python --version
# XX python version 3.9
# pip --version
# XX pip version 21.0.1
# pip3 install --upgrade pip
# pip3 install ultralytics
# pip3 install opencv-python

## WORKS tested on windows
## NO FACE DETECT, only person
## THIS uses ultralytics YOLO model to detect person

from ultralytics import YOLO
import cv2
import math
import time



cap=cv2.VideoCapture(0)
frame_width=int(cap.get(3))
frame_height = int(cap.get(4))
font=cv2.FONT_HERSHEY_SIMPLEX

## below is to save the video
# out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model=YOLO("../YOLO-Weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#FPS COUNT
timeMark=time.time()
fpsFilter=0

while True:
    success, img = cap.read()

    results=model(img,stream=True) #stream = True will use the generator and it is more efficient than normal

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls=int(box.cls[0])
            class_name=classNames[cls]
            if class_name == 'person':
                x1,y1,x2,y2=box.xyxy[0] # x1, y1 upper left corner, x2,y2 lower right corner

                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2) # convert from tensor format to int
                print(x1,y1,x2,y2)

                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)

                #print(box.conf[0]) # confidence in tensor
                conf=math.ceil((box.conf[0]*100))/100  # convert confidence tensor to int

                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                #print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
    ## FPS TIME CALCUALATION
    dt=time.time()-timeMark
    timeMark=time.time()
    fps=1/dt
    fpsFilter=.95*fpsFilter + .05 *fps
    ## FPS text
    cv2.rectangle(img,(0,0),(100, 40),(0,0,255),-1)
    cv2.putText(img, str(round(fpsFilter,2))+ 'fps',(0,25),font,.75,(0,255,255,2))

    cv2.imshow("Image", img)
    cv2.moveWindow('Image',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
exit()