from ultralytics import YOLO
import cv2


model=YOLO('yolov8n.pt')  ## yolo model 8n is fastest and less accurate, model 8x is slow but accurate
# results = model('XXXpathTOImages',show=True)
results = model('C:\\Users\\aytea\\Intellij_OpenCV\\JetsonNano\\venv\\known\\AytekinAkkaya.jpg', show=True)
cv2.waitKey(0)

# x --> Speed: 3.5ms preprocess, 314.7ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 640)
# n --> Speed: 3.0ms preprocess, 25.0ms inference, 4.0ms postprocess per image at shape (1, 3, 640, 640)