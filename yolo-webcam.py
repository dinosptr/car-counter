from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# cap = cv2.VideoCapture(0)
# cap.set(3, 720)
# cap.set(4, 720)

cap = cv2.VideoCapture('Videos/cars.mp4')
# Set the frame width and height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 950)

model = YOLO("model/yolov8n.pt")

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

prev_frame_time = 0
new_frame_time = 0

mask = cv2.imread('mask.png')
print(mask.shape)
while True:
  new_frame_time = time.time()
  success, img = cap.read()
  img = cv2.resize(img, (950, 480))
  print(img.shape)
  imgRegion = cv2.bitwise_and(img, mask)
  results = model(imgRegion, stream=True)

  for result in results:
    boxes = result.boxes
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      # cara pertama menggunakan library opencv (cv2)
      # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

      # Cara kedua menggunakan cvzone
      w, h = x2-x1, y2-y1
      cvzone.cornerRect(img, (x1, y1, w, h), l=15)

      # menambahkan confidence pada bbox
      conf = math.ceil((box.conf[0] * 100)) / 100 

      # menambahkan classname
      cls = int(box.cls[0])

      # menabahkan text ke dalam rect
      cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                          scale=0.6, thickness=1, offset=3)
    
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    
  cv2.imshow("Image", img)
  cv2.imshow("Image Region", imgRegion)
  cv2.waitKey(0)