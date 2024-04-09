from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *
import numpy as np

model = YOLO("model/yolov8l.pt")

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

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


# # config
# cap = cv2.VideoCapture('Videos/cars.mp4')
# limits = [290, 200, 478, 200]
# totalCount = []
# WIDTH = 950
# HEIGHT = 480
# mask = cv2.imread('mask.png')

cap = cv2.VideoCapture('Videos/cars_22.mp4')
WIDTH = 563
HEIGHT = 499
limits_in = [320, 200, 530, 200]
limits_out = [90, 200, 280, 200]
totalCount_in = []
totalCount_out = []
mask = cv2.imread('mask_22.png')

# webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 720)
# cap.set(4, 720)

# Config write video 
output_video_path = 'output_video.mp4'
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))

while True:
  new_frame_time = time.time()
  success, img = cap.read()
  if not success:
        break

  img = cv2.resize(img, (WIDTH, HEIGHT))

  imgRegion = cv2.bitwise_and(img, mask)
  results = model(imgRegion, stream=True, iou=0.5)

  detections = np.empty((0, 5))

  for result in results:
    boxes = result.boxes
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      # cara pertama menggunakan library opencv (cv2)
      # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

      # Cara kedua menggunakan cvzone
      w, h = x2-x1, y2-y1
      # cvzone.cornerRect(img, (x1, y1, w, h), l=15)

      # menambahkan confidence pada bbox
      conf = math.ceil((box.conf[0] * 100)) / 100 

      # menambahkan classname
      cls = int(box.cls[0])
      currentClass = classNames[cls]
      print(currentClass)
      if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.4:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
      # # menabahkan text ke dalam rect
      # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
      #                     scale=0.6, thickness=1, offset=3)

  resultsTracker = tracker.update(detections)
  cv2.line(img, (limits_in[0], limits_in[1]), (limits_in[2], limits_in[3]), (0,0,255), 3)
  cv2.line(img, (limits_out[0], limits_out[1]), (limits_out[2], limits_out[3]), (0,0,255), 3)
  for result in resultsTracker:
      x1, y1, x2, y2, id = result
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      w, h = x2 - x1, y2 - y1
      print(result)
      cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))
      # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
      #                              scale=2, thickness=3, offset=10)
      cx, cy = x1 + w // 2, y1 + h // 2
      cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

      if limits_in[0] < cx < limits_in[2] and limits_in[1] - 10 < cy < limits_in[1] + 10:
          if totalCount_in.count(id) == 0:
              totalCount_in.append(id)
              cv2.line(img, (limits_in[0], limits_in[1]), (limits_in[2], limits_in[3]), (0, 255, 0), 5)
      if limits_out[0] < cx < limits_out[2] and limits_out[1] - 10 < cy < limits_out[1] + 10:
          if totalCount_out.count(id) == 0:
              totalCount_out.append(id)
              cv2.line(img, (limits_out[0], limits_out[1]), (limits_out[2], limits_out[3]), (0, 255, 0), 5)
  cvzone.putTextRect(img, f' Count out: {len(totalCount_out)}', (30, 50), colorR=(0, 100, 0), scale=2, thickness=2, offset=5) 
  cvzone.putTextRect(img, f' Count in: {len(totalCount_in)}', (300, 50), colorR=(0, 100, 0), scale=2, thickness=2, offset=5) 

  # Write frame to output video
  out.write(img)
  
  cv2.imshow("Image", img)
  # cv2.imshow("Image Region", imgRegion)
  cv2.waitKey(1)