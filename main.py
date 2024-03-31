from ultralytics import YOLO
import cv2
 
model = YOLO('model/yolov8n.pt')
results = model("images/Images1.jpg", show=True)
cv2.waitKey(0)